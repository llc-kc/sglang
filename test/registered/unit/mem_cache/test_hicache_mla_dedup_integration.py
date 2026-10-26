"""Unit tests for the MLA host-dedup controller integration."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.l2_transfer as l2_transfer_module
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.speculative import base_spec_worker
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _nextn_worker(*, enable_dedup: bool):
    spec_algorithm = SimpleNamespace(
        is_eagle=lambda: True,
        is_eagle3=lambda: False,
        is_dspark=lambda: False,
    )
    target_runner = SimpleNamespace(
        mtp_draft_device_pools=(),
        spec_algorithm=spec_algorithm,
    )
    draft_pool = object()
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(architectures=["Glm4MoeForCausalLM"]),
        ),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        server_args=SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_mla_hicache_host_dedup=enable_dedup,
        ),
        _draft_model_runners=lambda: (draft_runner,),
    )
    return worker, target_runner, draft_pool


class TestHiCacheMLADedupDraftPlan(unittest.TestCase):
    def test_disabled_flag_keeps_original_write_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.write_queue = [op]
        controller.ack_write_queue = []
        controller._move_write_operation = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._move_mla_write_operation = mock.Mock()
        controller._l2_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._mla_l2_transfers = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller._mla_transfer_num_bytes = mock.Mock()
        controller.l2_transfer_engine = SimpleNamespace(
            submit_device_to_host=mock.Mock(return_value=completion)
        )

        controller.start_writing()

        controller._move_write_operation.assert_called_once_with(op)
        controller._l2_transfers.assert_called_once()
        controller._move_mla_write_operation.assert_not_called()
        controller._mla_l2_transfers.assert_not_called()
        controller._mla_transfer_num_bytes.assert_not_called()

    def test_disabled_flag_keeps_original_load_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        start_event = mock.Mock()
        producer_event = SimpleNamespace(start_event=start_event, complete=mock.Mock())
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.transfer_layer_id_max = 2
        controller.load_fence_stream = None
        controller.layer_done_counter = SimpleNamespace(
            update_producer=mock.Mock(return_value=0), events=[producer_event]
        )
        controller._move_op_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_load_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._start_loading_mla = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller.load_fence_stream = None
        controller.l2_transfer_engine = SimpleNamespace(
            submit_host_to_device=mock.Mock(return_value=completion)
        )

        self.assertEqual(controller.start_loading(), 0)

        controller._move_op_indices.assert_called_once_with(op)
        controller._l2_load_transfers.assert_called_once()
        controller._start_loading_mla.assert_not_called()

    def test_peer_l2_transfer_keeps_only_rank_local_sidecars(self):
        target_host = SimpleNamespace(_is_dummy=True)
        draft_host = SimpleNamespace(_is_dummy=False)
        target_device = object()
        draft_device = object()
        anchor = SimpleNamespace(
            host_pool=target_host,
            device_pool=target_device,
            layer_mapper=lambda layer_id: layer_id,
        )
        draft_entry = SimpleNamespace(
            host_pool=draft_host,
            device_pool=draft_device,
            layer_mapper=lambda layer_id: layer_id,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor,
            entry_map={PoolName.DRAFT: draft_entry},
        )
        indices = torch.arange(2)

        transfers = controller._mla_l2_transfers(
            indices,
            indices,
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=indices,
                    device_indices=indices,
                )
            ],
        )

        self.assertEqual(len(transfers), 1)
        self.assertIs(transfers[0].host_pool, draft_host)

    def test_packed_mtp_is_loaded_from_anchor_before_broadcast(self):
        anchor_host = SimpleNamespace(
            _is_dummy=False,
            load_to_device_per_layer_physical=mock.Mock(),
            prepare_transfer_indices=lambda host, device, backend: (host, device),
        )
        target_pool = object()
        draft_pool = object()
        anchor = SimpleNamespace(
            host_pool=anchor_host,
            device_pool=target_pool,
            layer_mapper=lambda layer_id: {0: 0, 1: 1, 2: 2}.get(layer_id),
            packed_draft_device_pools=(draft_pool,),
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor,
            entry_map={PoolName.KV: anchor},
        )
        controller.transfer_layer_id_max = 2
        controller.io_backend = "kernel"
        indices = torch.arange(2)

        transfers = controller._l2_load_transfers(indices, indices)
        engine = L2TransferEngine.__new__(L2TransferEngine)
        engine.host_to_device_stream = mock.Mock()
        engine.io_backend = "kernel"
        with mock.patch.object(
            l2_transfer_module,
            "device_module",
            SimpleNamespace(stream=lambda stream: nullcontext(), Event=mock.Mock),
        ):
            engine.submit_host_to_device(transfers, transfer_layer_id_max=1)

        self.assertEqual(len(transfers), 2)
        self.assertEqual(
            anchor_host.load_to_device_per_layer_physical.call_args_list,
            [
                mock.call(
                    target_pool,
                    indices,
                    indices,
                    0,
                    "kernel",
                    is_draft=False,
                ),
                mock.call(
                    draft_pool,
                    indices,
                    indices,
                    2,
                    "kernel",
                    is_draft=True,
                ),
            ],
        )

    def test_source_load_includes_target_and_rank_local_sidecars(self):
        target_host = object()
        draft_host = object()
        target = L2Transfer(target_host, object(), torch.arange(1), torch.arange(1))
        draft = L2Transfer(draft_host, object(), torch.arange(1), torch.arange(1))
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.KV: SimpleNamespace(host_pool=target_host),
                PoolName.DRAFT: SimpleNamespace(host_pool=draft_host),
            }
        )
        controller._l2_load_transfers = mock.Mock(return_value=[target, draft])
        op = CacheOperation(torch.arange(1), torch.arange(1), node_id=1)
        controller._move_op_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )

        self.assertEqual(controller._prepare_rank_local_load(op), [target, draft])

    def test_source_load_and_broadcast_are_layerwise(self):
        operations = []

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                operations.append(("record", self.name))

            def wait(self, stream):
                operations.append(("wait", self.name))

        class Stream:
            def synchronize(self):
                operations.append(("sync", None))

        broadcaster = SimpleNamespace(
            is_src=True,
            prepare_broadcast=lambda indices, stream: (indices, None),
            broadcast_loaded_layer=lambda layer_id, plan: operations.append(
                ("broadcast", layer_id)
            ),
            broadcast_loaded_mtp_draft=lambda pool, plan: operations.append(
                ("broadcast_mtp", pool)
            ),
        )
        producer_event = SimpleNamespace(
            start_event=Event("producer"),
            complete=lambda layer_id: operations.append(("complete", layer_id)),
        )
        # Include a linear layer before the two attention layers. Packed MTP
        # tails must load at their draft depth, independently of that mapping.
        anchor_host = SimpleNamespace(
            _is_dummy=False,
            load_to_device_per_layer_physical=mock.Mock(
                side_effect=lambda pool, host, device, layer, backend, **kwargs: (
                    operations.append(("load", pool, layer, kwargs["is_draft"]))
                )
            ),
        )
        anchor = SimpleNamespace(
            host_pool=anchor_host,
            device_pool="target",
            layer_mapper=lambda layer_id: {1: 0, 2: 1, 3: 2, 4: 3}.get(layer_id),
            packed_draft_device_pools=("mtp0", "mtp1"),
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor, entry_map={PoolName.KV: anchor}
        )
        controller.mla_dedup = SimpleNamespace(broadcaster=broadcaster)
        controller.transfer_layer_id_max = 3
        controller.load_fence_stream = None
        controller.io_backend = "kernel"
        controller.layer_done_counter = SimpleNamespace(events=[producer_event])
        engine = L2TransferEngine.__new__(L2TransferEngine)
        engine.host_to_device_stream = Stream()
        engine.io_backend = "kernel"
        controller.l2_transfer_engine = engine
        anchor_host.prepare_transfer_indices = lambda host, device, backend: (
            host,
            device,
        )
        controller.ack_load_queue = []
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._mla_transfer_num_bytes = mock.Mock(return_value=0)
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        controller.move_hybrid_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )

        fake_device_module = SimpleNamespace(
            stream=lambda stream: nullcontext(),
        )
        ack_start, ack_finish = Event("ack_start"), Event("ack_finish")
        with (
            mock.patch.object(l2_transfer_module, "device_module", fake_device_module),
            mock.patch.object(
                l2_transfer_module,
                "make_timing_event_pair",
                return_value=(ack_start, ack_finish, True),
            ),
        ):
            controller._start_loading_mla(0, op)

        self.assertEqual(
            operations,
            [
                ("record", "producer"),
                ("wait", "producer"),
                ("record", "ack_start"),
                ("load", "mtp0", 2, True),
                ("broadcast_mtp", "mtp0"),
                ("complete", 0),
                ("load", "target", 0, False),
                ("load", "mtp1", 3, True),
                ("broadcast", 0),
                ("broadcast_mtp", "mtp1"),
                ("complete", 1),
                ("load", "target", 1, False),
                ("broadcast", 1),
                ("complete", 2),
                ("record", "ack_finish"),
                ("sync", None),
            ],
        )
        ack = controller.ack_load_queue[0]
        self.assertIs(ack.start_event, ack_start)
        self.assertIs(ack.finish_event, ack_finish)
        self.assertTrue(ack.timing_enabled)

    def test_peer_load_broadcasts_with_and_without_rank_local_sidecars(self):
        for has_sidecar in (False, True):
            with self.subTest(has_sidecar=has_sidecar):
                operations = []
                dummy = SimpleNamespace(_is_dummy=True)
                sidecar = SimpleNamespace(
                    _is_dummy=False,
                    prepare_transfer_indices=lambda host, device, backend: (
                        host,
                        device,
                    ),
                    load_to_device_per_layer_physical=lambda pool, host, device, layer, backend, **kwargs: (
                        operations.append(("load_sidecar", layer))
                    ),
                )
                controller = HiCacheController.__new__(HiCacheController)
                broadcaster = SimpleNamespace(
                    prepare_broadcast=mock.Mock(return_value=mock.sentinel.plan),
                    broadcast_loaded_layer=lambda layer, plan: operations.append(
                        ("broadcast", layer)
                    ),
                )
                controller.mla_dedup = SimpleNamespace(
                    is_dummy_rank=True, broadcaster=broadcaster
                )
                controller.transfer_layer_id_max = 2
                controller.load_fence_stream = None
                controller.layer_done_counter = SimpleNamespace(
                    events=[
                        SimpleNamespace(
                            start_event=mock.Mock(),
                            complete=lambda layer: operations.append(
                                ("complete", layer)
                            ),
                        )
                    ]
                )
                controller.ack_load_queue = []
                controller._num_tokens_by_pool = mock.Mock(return_value={})
                controller._mla_transfer_num_bytes = mock.Mock(return_value=0)
                indices = torch.arange(2)
                controller._move_op_indices = mock.Mock(
                    return_value=(indices, indices, None)
                )
                transfers = [L2Transfer(dummy, object(), indices, indices)]
                if has_sidecar:
                    transfers.append(L2Transfer(sidecar, object(), indices, indices))
                controller._l2_load_transfers = mock.Mock(return_value=transfers)
                engine = L2TransferEngine.__new__(L2TransferEngine)
                engine.host_to_device_stream = mock.Mock()
                engine.io_backend = "kernel"
                controller.l2_transfer_engine = engine

                with mock.patch.object(
                    l2_transfer_module,
                    "device_module",
                    SimpleNamespace(
                        stream=lambda stream: nullcontext(), Event=mock.Mock
                    ),
                ):
                    controller._start_loading_mla(
                        0, CacheOperation(indices, indices, node_id=1)
                    )

                expected = []
                for layer in range(2):
                    if has_sidecar:
                        expected.append(("load_sidecar", layer))
                    expected.extend((("broadcast", layer), ("complete", layer)))
                self.assertEqual(operations, expected)
                broadcaster.prepare_broadcast.assert_called_once()
                engine.host_to_device_stream.synchronize.assert_called_once()
                self.assertEqual(len(controller.ack_load_queue), 1)

    def test_hybrid_broadcast_skips_linear_layers(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(
                layer_mapper=lambda layer_id: {0: 0, 2: 1}.get(layer_id)
            )
        )

        self.assertEqual(controller._mla_broadcast_layer_id(0), 0)
        self.assertIsNone(controller._mla_broadcast_layer_id(1))
        self.assertEqual(controller._mla_broadcast_layer_id(2), 1)

    def test_dedup_peer_backs_up_every_real_sidecar(self):
        real_pool = SimpleNamespace(_is_dummy=False)
        dummy_pool = SimpleNamespace(_is_dummy=True)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.backup_skip = True
        controller.mla_dedup = object()
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.MAMBA: SimpleNamespace(host_pool=real_pool),
                PoolName.DRAFT_INDEXER: SimpleNamespace(host_pool=real_pool),
                PoolName.INDEXER: SimpleNamespace(host_pool=dummy_pool),
            }
        )

        self.assertTrue(controller.should_backup(PoolTransfer(PoolName.MAMBA)))
        self.assertTrue(controller.should_backup(PoolTransfer(PoolName.DRAFT_INDEXER)))
        self.assertFalse(controller.should_backup(PoolTransfer(PoolName.INDEXER)))

        controller.mla_dedup = None
        controller.storage_backend_type = "mooncake"
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        controller.mem_pool_host.entry_map[PoolName.DRAFT] = SimpleNamespace(
            host_pool=MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        )
        self.assertTrue(controller.should_backup(PoolTransfer(PoolName.DRAFT)))
        self.assertFalse(controller.should_backup(PoolTransfer(PoolName.DRAFT_INDEXER)))

    def test_peer_storage_load_skips_dummy_kv_and_loads_real_draft(self):
        from queue import Queue

        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = SimpleNamespace(is_dummy_rank=True)
        controller.page_size = 2
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.KV: SimpleNamespace(host_pool=SimpleNamespace(_is_dummy=True)),
                PoolName.INDEXER: SimpleNamespace(
                    host_pool=SimpleNamespace(_is_dummy=True)
                ),
                PoolName.DRAFT: SimpleNamespace(
                    host_pool=SimpleNamespace(_is_dummy=False)
                ),
            }
        )
        controller.page_get_func = mock.Mock(
            side_effect=AssertionError("dummy KV read")
        )
        controller.storage_backend = SimpleNamespace(
            batch_get_v2=mock.Mock(return_value={PoolName.DRAFT: [True, False]})
        )
        controller.prefetch_sync_queue = Queue()
        operation = SimpleNamespace(
            request_id="peer",
            hash_value=["page0", "page1"],
            host_indices=torch.arange(4),
            prefix_keys=["prefix"],
            pool_transfers=[
                PoolTransfer(name, indices_from_pool=PoolName.KV)
                for name in (PoolName.INDEXER, PoolName.DRAFT)
            ],
            is_terminated=lambda: False,
        )

        self.assertEqual(controller._page_transfer(operation), 1)
        controller.page_get_func.assert_not_called()
        transfers = controller.storage_backend.batch_get_v2.call_args.args[0]
        self.assertEqual([transfer.name for transfer in transfers], [PoolName.DRAFT])
        self.assertEqual(
            controller.prefetch_sync_queue.get_nowait().completed_tokens, 2
        )

    def test_dedup_keeps_nextn_draft_packed(self):
        for enabled in (False, True):
            with (
                self.subTest(enable_dedup=enabled),
                mock.patch.object(
                    base_spec_worker,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=True,
                        enable_unified_cache_external_linker=False,
                        enable_mla_hicache_host_dedup=enabled,
                    ),
                ),
            ):
                worker, target_runner, draft_pool = _nextn_worker(enable_dedup=enabled)
                plan = BaseSpecWorker._build_hicache_draft_plan(worker)
                self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
                self.assertEqual(target_runner.mtp_draft_device_pools, (draft_pool,))

    def test_dedup_keeps_non_mtp_speculative_caches_as_sidecars(self):
        cases = (
            ("eagle3", True, True, False, "EagleDraft", HiCacheDraftMode.SIDECAR),
            ("dflash", False, False, False, "DFlashDraft", HiCacheDraftMode.SIDECAR),
            ("dspark", False, False, True, "GenericDSpark", HiCacheDraftMode.SIDECAR),
            (
                "dspark_dsv4",
                False,
                False,
                True,
                "DeepseekV4ForCausalLMDSpark",
                HiCacheDraftMode.PACKED,
            ),
        )
        for name, is_eagle, is_eagle3, is_dspark, architecture, expected in cases:
            with (
                self.subTest(name=name),
                mock.patch.object(
                    base_spec_worker,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=True,
                        enable_unified_cache_external_linker=False,
                    ),
                ),
            ):
                algorithm = SimpleNamespace(
                    is_eagle=lambda: is_eagle,
                    is_eagle3=lambda: is_eagle3,
                    is_dspark=lambda: is_dspark,
                )
                target_runner = SimpleNamespace(
                    mtp_draft_device_pools=(), spec_algorithm=algorithm
                )
                draft_pool = object()
                draft_runner = SimpleNamespace(
                    token_to_kv_pool=draft_pool,
                    model_config=SimpleNamespace(
                        num_nextn_predict_layers=0,
                        hf_config=SimpleNamespace(architectures=[architecture]),
                    ),
                )
                worker = SimpleNamespace(
                    target_worker=SimpleNamespace(model_runner=target_runner),
                    _draft_model_runners=lambda: (draft_runner,),
                )

                plan = BaseSpecWorker._build_hicache_draft_plan(worker)

                self.assertEqual(plan.mode, expected)
                packed = (draft_pool,) if expected == HiCacheDraftMode.PACKED else ()
                self.assertEqual(target_runner.mtp_draft_device_pools, packed)


if __name__ == "__main__":
    unittest.main()
