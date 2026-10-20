import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller as cc
from sglang.srt.mem_cache.l2_transfer import L2Transfer, TransferCompletion
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    before_mla_forward_layer,
    bind_mla_forward_load,
    forward_context,
    get_forward_context,
)
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Event:
    def __init__(self, **kwargs):
        self.ready = True

    def record(self):
        self.ready = False

    def query(self):
        return self.ready

    def wait(self, stream):
        stream.wait_event(self)

    def synchronize(self):
        self.ready = True


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)

    def wait_stream(self, stream):
        self.waits.append(stream)

    def synchronize(self):
        raise AssertionError("Forward mode must not drain the load stream")


class Device:
    Event = Event
    active = None

    @classmethod
    @contextmanager
    def stream(cls, stream):
        previous, cls.active = cls.active, stream
        try:
            yield
        finally:
            cls.active = previous

    @classmethod
    def current_stream(cls):
        return cls.active

    @staticmethod
    def is_current_stream_capturing():
        return False


class TestMLAForwardLoad(CustomTestCase):
    @contextmanager
    def controller(self, is_source):
        operations = []
        load, forward, fence = Stream(), Stream(), Stream()
        Device.active = forward
        controller = cc.HiCacheController.__new__(cc.HiCacheController)
        controller.transfer_layer_id_max = 3
        controller.io_backend = "kernel"
        controller.enable_storage = False
        controller.load_fence_stream = fence
        controller.ack_load_queue = []
        controller.mem_pool_device_allocator = Mock()
        target = SimpleNamespace(
            name="target", host_pool=SimpleNamespace(_is_dummy=not is_source)
        )
        draft = SimpleNamespace(
            name="draft", host_pool=SimpleNamespace(_is_dummy=False)
        )
        controller._move_op_indices = lambda op: (
            op.host_indices,
            op.device_indices,
            None,
        )
        controller._l2_load_transfers = lambda *args: [target, draft]
        controller._mla_broadcast_transfer = lambda op: L2Transfer(
            target.host_pool, target, op.host_indices, op.device_indices
        )
        controller._num_tokens_by_pool = lambda op: {}
        controller._mla_transfer_num_bytes = lambda op: 128
        controller._load_mla_transfers_layer = lambda transfers, k: operations.extend(
            ("h2d", (transfer.name,), k, Device.active) for transfer in transfers
        )
        controller.mla_dedup = SimpleNamespace(
            broadcaster=SimpleNamespace(
                is_src=is_source,
                layer_num=3,
                prepare_broadcast=lambda indices, stream: (indices, None),
                broadcast_loaded_layer=lambda k, plan: operations.append(
                    ("broadcast", (), k, Device.active)
                ),
            )
        )

        @contextmanager
        def submission(transfers, *, start_event):
            self.assertIs(Device.active, load)
            completion = TransferCompletion(Event(), Event(), False)
            yield transfers, completion
            completion.finish_event.record()

        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=load,
            host_to_device_submission=submission,
            resolve_device_indices=lambda op: op.device_indices,
        )
        with (
            patch.object(cc, "device_module", Device),
            get_context().override_server_args(
                disaggregation_mode="null", pp_size=1, enable_unified_memory=False
            ),
            envs.SGLANG_MLA_DEDUP_LOAD_MODE.override("forward"),
        ):
            controller.layer_done_counter = cc.LayerDoneCounter(3)
            yield controller, operations, load, forward, fence, None

    def test_source_and_peer_prefetch_then_broadcast(self):
        for source in (True, False):
            with self.subTest(source=source), self.controller(source) as setup:
                controller, operations, load, forward, fence, _ = setup
                counter = controller.layer_done_counter
                slot = counter.update_producer()
                indices = torch.arange(4)
                controller._start_loading_mla(
                    slot, cc.CacheOperation(indices, indices, 123)
                )
                state = controller._mla_forward_loads[slot]
                self.assertIn(fence, load.waits)
                self.assertFalse(counter.events[slot].finish_event.query())
                self.assertEqual(controller.ack_load_queue, [])
                controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_not_called()
                self.assertTrue(all(kind == "h2d" for kind, *_ in operations))
                self.assertEqual(
                    sum(ts == ("target",) for _, ts, _, _ in operations),
                    3 if source else 0,
                )
                self.assertEqual(sum(ts == ("draft",) for _, ts, _, _ in operations), 3)

                counter.set_consumer(slot)
                handle = counter.bind_forward_load()
                handle.before_layer(0)
                counter.wait_until(0)
                self.assertEqual(sum(kind == "broadcast" for kind, *_ in operations), 1)
                self.assertEqual(controller.ack_load_queue, [])
                handle.before_layer(2)
                self.assertTrue(
                    all(
                        s is forward
                        for kind, _, _, s in operations
                        if kind == "broadcast"
                    )
                )
                self.assertTrue(
                    all(event in forward.waits for event in state["copy_ready"])
                )
                ack = controller.ack_load_queue[0]
                self.assertIs(ack.finish_event, state["finish"])
                self.assertIsNot(ack.finish_event, state["completion"].finish_event)
                self.assertFalse(ack.finish_event.query())
                self.assertEqual(ack.node_ids, [123])
                controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_called_once_with(
                    (id(controller), "load"), ack.finish_event
                )
                self.assertIsNone(counter.bind_forward_load())

    def test_queued_batches_order_and_recycled_handles(self):
        with self.controller(True) as setup:
            controller, _, _, _, _, _ = setup
            counter = controller.layer_done_counter
            indices = torch.arange(4)
            handles = []
            for node in (10, 11):
                slot = counter.update_producer()
                controller._start_loading_mla(
                    slot, cc.CacheOperation(indices, indices, node)
                )
                counter.set_consumer(slot)
                handles.append(counter.bind_forward_load())
            with self.assertRaisesRegex(RuntimeError, "submission order"):
                handles[1].before_layer(0)
            with self.assertRaisesRegex(RuntimeError, "unfinished"):
                counter.reset()
            handles[0].before_layer(2)
            self.assertEqual(counter.consumer_index, 1)
            handles[1].before_layer(2)
            self.assertEqual(
                [a.node_ids for a in controller.ack_load_queue], [[10], [11]]
            )
            counter.producer_index = 2
            with self.assertRaisesRegex(AssertionError, "ready before being reused"):
                counter.update_producer()
            self.assertEqual(counter.producer_index, 2)
            for event in counter.events[0].load_events:
                event.ready = True
            counter.update_producer()
            with self.assertRaisesRegex(RuntimeError, "recycled slot"):
                handles[0].before_layer(0)

    def test_unsupported_paths_fail_before_submission(self):
        with self.controller(True) as setup:
            controller, operations, _, _, _, _ = setup
            indices = torch.arange(4)
            op = cc.CacheOperation(indices, indices, 10)
            slot = controller.layer_done_counter.update_producer()
            with get_memory().override(enable_unified_memory=True):
                with self.assertRaisesRegex(ValueError, "movable device pages"):
                    controller._start_loading_mla(slot, op)
            for override in (
                get_parallel().override(pp_size=2),
                get_disagg().override(disaggregation_mode="prefill"),
            ):
                with override, self.assertRaisesRegex(ValueError, "non-PD, PP1"):
                    controller._start_loading_mla(slot, op)
            self.assertEqual(operations, [])
            self.assertFalse(controller.layer_done_counter.events[slot].forward_pending)

    def test_storage_enabled_load_is_consumed_by_forward(self):
        """L3 attachment must not reject a subsequent L2-to-GPU reload."""
        with self.controller(False) as setup:
            controller, operations, _, _, _, _ = setup
            controller.enable_storage = True
            indices = torch.arange(4)
            slot = controller.layer_done_counter.update_producer()
            controller._start_loading_mla(slot, cc.CacheOperation(indices, indices, 42))
            self.assertEqual(controller.ack_load_queue, [])
            controller.layer_done_counter.set_consumer(slot)
            controller.layer_done_counter.bind_forward_load().before_layer(2)
            self.assertEqual(
                [ack.node_ids for ack in controller.ack_load_queue], [[42]]
            )
            self.assertEqual(sum(kind == "broadcast" for kind, *_ in operations), 3)

    def test_unsupported_model_and_forward_modes_leave_load_pending(self):
        with self.controller(True) as setup:
            controller, _, _, _, _, _ = setup
            counter = controller.layer_done_counter
            indices = torch.arange(2)
            slot = counter.update_producer()
            controller._start_loading_mla(slot, cc.CacheOperation(indices, indices, 3))
            counter.set_consumer(slot)
            for supported, tbo, split in (
                (False, False, False),
                (True, True, False),
                (True, False, True),
            ):
                with self.subTest(supported=supported, tbo=tbo, split=split):
                    with self.assertRaisesRegex(RuntimeError, "supported model"):
                        bind_mla_forward_load(
                            SimpleNamespace(layer_transfer_counter=counter),
                            model=SimpleNamespace(supports_mla_forward_load=supported),
                            can_run_tbo=tbo,
                            split_prefill=split,
                        )
                    self.assertTrue(counter.events[slot].forward_pending)
                    self.assertEqual(controller.ack_load_queue, [])

    def test_hybrid_reload_keeps_mamba_and_sidecar_draft_rank_local(self):
        """A linear layer has no MLA row; peer state/draft still need H2D."""
        for source in (False, True):
            with self.subTest(source=source), self.controller(source) as setup:
                controller, _, _, _, _, _ = setup
                controller.enable_storage = True
                target = [torch.full((2, 1), -1.0) for _ in range(2)]
                mamba = torch.full((2, 1), -1.0)
                draft = torch.full((2, 1), -1.0)
                rank_value = 100.0 if source else 200.0
                indices = torch.arange(2)

                def transfer(buffers, mapping, value, *, dummy=False):
                    def load(pool, host, device, layer, backend, **kwargs):
                        buffers[layer].fill_(value + layer)

                    host_pool = SimpleNamespace(
                        _is_dummy=dummy, load_to_device_per_layer_physical=load
                    )
                    return L2Transfer(
                        host_pool, object(), indices, indices, layer_mapper=mapping
                    )

                transfers = [
                    transfer(
                        target, lambda i: {0: 0, 2: 1}.get(i), 10, dummy=not source
                    ),
                    transfer([mamba], lambda i: 0 if i == 1 else None, rank_value),
                    transfer([draft], lambda i: 0 if i == 0 else None, rank_value + 1),
                ]
                controller._l2_load_transfers = lambda *args: transfers
                del controller._load_mla_transfers_layer
                controller._mla_broadcast_layer_id = lambda i: {0: 0, 2: 1}.get(i)
                broadcast_layers = []

                def broadcast(layer, plan):
                    expected = torch.full((2, 1), 10.0 + layer)
                    if source:
                        torch.testing.assert_close(target[layer], expected)
                    else:
                        target[layer].copy_(expected)
                    broadcast_layers.append(layer)

                controller.mla_dedup.broadcaster.broadcast_loaded_layer = broadcast
                counter = controller.layer_done_counter
                slot = counter.update_producer()
                controller._start_loading_mla(
                    slot, cc.CacheOperation(indices, indices, 7)
                )
                torch.testing.assert_close(mamba, torch.full((2, 1), rank_value))
                torch.testing.assert_close(draft, torch.full((2, 1), rank_value + 1))
                counter.set_consumer(slot)
                handle = counter.bind_forward_load()
                handle.before_layer(1)
                self.assertEqual(broadcast_layers, [0])
                self.assertEqual(controller.ack_load_queue, [])
                handle.before_layer(2)
                self.assertEqual(broadcast_layers, [0, 1])
                self.assertEqual(len(controller.ack_load_queue), 1)

    def test_load_context_survives_eager_batch_conversion(self):
        """ModelRunner's load handle must survive EagerRunner replacing its batch."""
        from sglang.srt.model_executor import model_runner as mr
        from sglang.srt.model_executor.cuda_graph_buffer_registry import (
            CudaGraphBufferRegistry,
        )
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner

        with self.controller(True) as setup:
            controller, _, _, _, _, _ = setup
            indices = torch.arange(2)
            counter = controller.layer_done_counter
            slot = counter.update_producer()
            controller._start_loading_mla(slot, cc.CacheOperation(indices, indices, 9))
            counter.set_consumer(slot)
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=indices,
                req_pool_indices=indices[:1],
                seq_lens=indices[:1],
                out_cache_loc=indices,
                seq_lens_sum=2,
            )
            eager = EagerRunner.__new__(EagerRunner)
            eager._eager_registry = CudaGraphBufferRegistry(
                device=torch.device("cpu"), max_bs=1, max_num_tokens=2
            )
            converted = []

            def execute(live_batch, **kwargs):
                view = eager.load_batch(live_batch)
                converted.append(view)
                for layer in range(3):
                    before_mla_forward_layer(layer)
                return "restored"

            runner = mr.ModelRunner.__new__(mr.ModelRunner)
            runner.token_to_kv_pool = SimpleNamespace(layer_transfer_counter=counter)
            runner.model = SimpleNamespace(supports_mla_forward_load=True)
            runner.attn_backend = object()
            runner.device = "cuda"
            runner.hisparse_coordinator = None
            runner.prefill_cuda_graph_runner = None
            runner.eager_runner = SimpleNamespace(execute=execute)
            runner._prepare_eager_forward_batch = lambda fb: None
            runner._maybe_execute_deferred_mamba_cow_and_clear = lambda fb: None
            outer = ForwardContext(attn_backend=object())
            with (
                forward_context(outer),
                envs.SGLANG_EAGER_INPUT_NO_COPY.override(False),
                patch.object(mr, "get_global_dwdp_manager", return_value=None),
            ):
                result = runner._forward_raw(batch, None)
                self.assertIs(get_forward_context(), outer)
            self.assertIsNot(converted[0], batch)
            self.assertEqual(result.logits_output, "restored")
            self.assertFalse(result.can_run_graph)
            self.assertFalse(counter.events[slot].forward_pending)
            self.assertEqual([ack.node_ids for ack in controller.ack_load_queue], [[9]])

            def replay(live_batch, **kwargs):
                self.assertIsNone(get_forward_context().mla_forward_load)
                return "graph"

            runner.decode_cuda_graph_runner = SimpleNamespace(
                can_run_graph=lambda fb: True, execute=replay
            )
            batch.forward_mode = ForwardMode.DECODE
            result = runner._forward_raw(batch, None)
            self.assertTrue(result.can_run_graph)
            self.assertEqual(result.logits_output, "graph")

    def test_mamba_cow_waits_for_local_reload_before_copying_state(self):
        """COW precedes the layer loop and must not read an unrestored source."""
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.model_runner import ModelRunner

        with self.controller(False) as setup:
            controller, operations, _, forward, _, _ = setup
            counter = controller.layer_done_counter
            indices = torch.arange(2)
            slot = counter.update_producer()
            controller._start_loading_mla(slot, cc.CacheOperation(indices, indices, 12))
            counter.set_consumer(slot)
            copy_event = counter.events[slot].local_copies_event
            state = torch.tensor([7.0, -1.0])
            pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
            pool.mamba_ckpt_pool = None
            pool.translate_mamba_indices = lambda indices: indices

            def copy(src, dst):
                self.assertIn(copy_event, forward.waits)
                state[dst] = state[src]

            pool.copy_mamba_state = copy
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                mamba_clear_indices=None,
                mamba_cow_src_indices=torch.tensor([0]),
                mamba_cow_dst_indices=torch.tensor([1]),
            )
            runner = SimpleNamespace(req_to_token_pool=pool, is_draft_worker=False)
            with forward_context(
                ForwardContext(
                    attn_backend=object(), mla_forward_load=counter.bind_forward_load()
                )
            ):
                ModelRunner._maybe_execute_deferred_mamba_cow_and_clear(runner, batch)
            torch.testing.assert_close(state, torch.tensor([7.0, 7.0]))
            self.assertIsNone(batch.mamba_cow_src_indices)
            self.assertEqual(controller.ack_load_queue, [])
            self.assertTrue(all(kind == "h2d" for kind, *_ in operations))

    def test_layer_hook_without_reload_remains_fullgraph_compilable(self):
        """The always-present model hook must not break normal compiled forwards."""

        def layer(tensor):
            before_mla_forward_layer(0)
            return tensor + 1

        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        with forward_context(ForwardContext(attn_backend=object())):
            torch.testing.assert_close(
                compiled(torch.tensor([2.0])), torch.tensor([3.0])
            )


if __name__ == "__main__":
    unittest.main()
