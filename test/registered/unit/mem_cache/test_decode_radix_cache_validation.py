"""Decode prefix reuse admits GPU DeepSeek HiCache and unified L1-only KV.

Paged layouts require the prefix-only HiCache path; other SWA HiCache stays gated.

    python -m pytest test/registered/unit/mem_cache/test_decode_radix_cache_validation.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import validate_decode_radix_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DSV4 = SimpleNamespace(is_deepseek_v4_arch=True, is_hybrid_swa_compress=False)


def _validate(token_to_kv_pool, *, hicache=False):
    validate_decode_radix_cache(
        model_config=_DSV4,
        token_to_kv_pool=token_to_kv_pool,
        is_hybrid_swa=True,
        enable_hierarchical_cache=hicache,
    )


class TestValidateDecodeRadixCacheDSV4(CustomTestCase):
    def test_unified_kv_bf16_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=False))

    def test_unified_kv_fp8_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=True))

    def test_paged_layout_is_rejected(self):
        for pool in (
            SimpleNamespace(_unified_kv=False, _unified_kv_fp8=False),
            Mock(spec=DeepSeekV4TokenToKVPool, _unified_kv=False),
        ):
            with self.subTest(pool=type(pool).__name__):
                with self.assertRaisesRegex(ValueError, "unified KV layout"):
                    _validate(pool)

    def test_hicache_admits_deepseek_but_keeps_other_swa_layouts_gated(self):
        """PD can restore DeepSeek prefix pages while prefill sends its state."""
        for unified in (False, True):
            with self.subTest(unified=unified):
                _validate(
                    Mock(spec=DeepSeekV4TokenToKVPool, _unified_kv=unified),
                    hicache=True,
                )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _validate(SimpleNamespace(), hicache=True)
        # NPU has an independently allocated C128 tree component, which this
        # GPU prefix-only host stack cannot restore.
        with patch("sglang.srt.mem_cache.kv_cache_builder.is_npu", return_value=True):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _validate(Mock(spec=DeepSeekV4TokenToKVPool), hicache=True)


if __name__ == "__main__":
    unittest.main()
