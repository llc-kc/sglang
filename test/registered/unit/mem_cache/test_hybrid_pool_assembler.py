"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _split_hicache_size,
    build_full_draft_pools,
    build_hybrid_mamba_stack,
    build_kv_only_group,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestDeviceAllocEviction(CustomTestCase):
    def test_swa_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 8

        _evict_swa_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(swa_num_tokens=2))
        cache.evict.assert_not_called()

    def test_mamba_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        allocator = cache.req_to_token_pool.mamba_allocator
        allocator.schedulable_available_size.return_value = 8

        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(mamba_num=2))
        cache.evict.assert_not_called()

    def test_sufficient_capacity_skips_eviction(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 10
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size.return_value = 10

        _evict_swa_for_device_alloc(cache, required_size=10)
        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_not_called()
        cache.evict.assert_not_called()


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_kv_only_group_builds_dummy_anchor_on_peer_rank(self):
        host_pool = SimpleNamespace(
            layout="page_first",
            page_size=2,
            device="cpu",
            size=8,
            logical_size=8,
            can_use_write_back_jit=False,
        )
        kv_pool = SimpleNamespace(layer_num=2)
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
            "build_kv_host_pool",
            return_value=host_pool,
        ) as build_host_pool:
            group = build_kv_only_group(
                page_size=2,
                kv_pool=kv_pool,
                full_layer_mapping={0: 0, 1: 1},
                use_mla=True,
                is_dummy=True,
            )

        self.assertIs(group.anchor_entry.host_pool, host_pool)
        self.assertTrue(build_host_pool.call_args.kwargs["is_dummy"])

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_empty_hybrid_linear_pool(self):
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        draft_kv_pool.full_kv_pool = SimpleNamespace(layer_num=0)

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
            )
        )
        # The layout comes from the published configuration.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path="dummy", hicache_mem_layout="page_first")
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ) as build_host_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_kv_pool,
                tree_cache=tree_cache,
            )

        self.assertEqual(build_host_pool.call_args.kwargs["host_to_device_ratio"], 1.0)
        self.assertEqual(len(specs), 1)
        self.assertIs(entries[0].host_pool, draft_host_pool)


class TestHybridMambaDsaAssembly(CustomTestCase):
    def test_dsa_full_layers_add_dedup_aware_indexer_with_packed_mtp(self):
        class FakeDSAPool:
            layer_num = 2

        def host_pool():
            return SimpleNamespace(
                layout="page_first",
                page_size=2,
                device="cpu",
                size=8,
                logical_size=8,
                can_use_write_back_jit=False,
            )

        kv_pool = FakeDSAPool()
        draft_pool = SimpleNamespace(layer_num=1)
        params = SimpleNamespace(
            page_size=2,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(
                    alloc=MagicMock(),
                    free=MagicMock(),
                )
            ),
            mtp_draft_device_pools=(draft_pool,),
            token_to_kv_pool_allocator=MagicMock(),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = SimpleNamespace(
            hicache_size=0,
            hicache_ratio=1,
            hicache_mem_layout="page_first",
            hicache_write_policy="write_back",
            hicache_io_backend="kernel",
            hicache_host_memory_mode="kernel",
        )
        indexer_host = host_pool()

        with (
            patch(
                "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool",
                FakeDSAPool,
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "build_kv_host_pool",
                return_value=host_pool(),
            ) as build_kv_host,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.MambaPoolHost",
                return_value=host_pool(),
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "DSAIndexerPoolHost",
                return_value=indexer_host,
            ) as build_indexer_host,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "HybridCacheController",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
                return_value=memory,
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            group, _ = build_hybrid_mamba_stack(
                params=params,
                kv_pool=kv_pool,
                mamba_pool=SimpleNamespace(),
                full_layer_mapping={0: 0, 2: 1},
                mamba_layer_mapping={1: 0},
                load_cache_event=None,
                storage_backend="mooncake",
                use_mla=True,
                mla_kv_is_dummy=True,
                mla_dedup_context=MagicMock(),
            )

        self.assertIn(PoolName.INDEXER, group.entry_map)
        self.assertTrue(build_kv_host.call_args.kwargs["is_dummy"])
        self.assertTrue(build_indexer_host.call_args.kwargs["is_dummy"])
        indexer_entry = group.entry_map[PoolName.INDEXER]
        self.assertEqual(indexer_entry.layer_mapper(0), 0)
        self.assertIsNone(indexer_entry.layer_mapper(1))
        self.assertEqual(indexer_entry.layer_mapper(2), 1)
        self.assertEqual(indexer_entry.layer_mapper(3), 2)


if __name__ == "__main__":
    unittest.main()
