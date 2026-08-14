"""Unit tests for MLA host-dedup draft-cache planning."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
    HiCacheDraftPlan,
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
    def test_dedup_keeps_nextn_draft_rank_local(self):
        worker, target_runner, draft_pool = _nextn_worker(enable_dedup=True)

        plan = BaseSpecWorker._build_hicache_draft_plan(worker)

        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_runner.mtp_draft_device_pools, ())

    def test_normal_hicache_still_packs_nextn_draft(self):
        worker, target_runner, draft_pool = _nextn_worker(enable_dedup=False)

        plan = BaseSpecWorker._build_hicache_draft_plan(worker)

        self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_runner.mtp_draft_device_pools, (draft_pool,))

    def test_unified_cache_uses_independent_draft_pool_with_dedup(self):
        tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        draft_pool = object()
        plan = HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,
            device_pools=(draft_pool,),
        )
        server_args = SimpleNamespace(enable_mla_hicache_host_dedup=True)

        with mock.patch.object(
            kv_cache_builder, "_register_legacy_hicache_draft"
        ) as register_legacy:
            kv_cache_builder.maybe_register_hicache_draft(
                tree_cache=tree_cache,
                draft_plan=plan,
                server_args=server_args,
                page_size=64,
            )

        register_legacy.assert_called_once_with(
            tree_cache=tree_cache,
            draft_pool=draft_pool,
            server_args=server_args,
            page_size=64,
        )


if __name__ == "__main__":
    unittest.main()
