"""PD decode restores DeepSeek-V4.1 prefix pages from Mooncake HiCache.

Prefill has no HiCache: every request's window and compressor state must arrive
through PD, including after decode flushes both of its memory cache tiers.
"""

import time
import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import is_in_ci, try_cached_model

register_cuda_ci(est_time=650, stage="extra-b", runner_config="4-gpu-gb300")

PAGE_SIZE = 256
SERVER_ARGS = [
    "--page-size",
    str(PAGE_SIZE),
    "--dp-size",
    "2",
    "--ep-size",
    "2",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "megamoe",
    "--dsv4-attn-backend",
    "flashmla",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--enable-decoder-swa-bounded-replay",
    "--mem-fraction-static",
    "0.85",
    "--model-loader-extra-config",
    '{"enable_multithread_load":true}',
]


class TestDSV41DecodeHiCache(PDDisaggregationServerBase):
    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2
    extra_prefill_args = [*SERVER_ARGS, "--chunked-prefill-size", "65536"]
    extra_decode_args = [
        *SERVER_ARGS,
        "--disaggregation-decode-enable-radix-cache",
        "--enable-hierarchical-cache",
        "--hicache-size",
        "1",
        "--hicache-write-policy",
        "write_through",
        "--hicache-io-backend",
        "direct",
        "--hicache-mem-layout",
        "page_first_direct",
        "--hicache-storage-backend",
        "mooncake",
        "--enable-metrics",
        "--enable-cache-report",
        "--cuda-graph-max-bs-decode",
        "24",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model("deepseek-ai/DeepSeek-V4.1-Flash")
        cls.mooncake = MooncakeTestServices(
            protocol="rdma" if is_in_ci() else "tcp",
            store_segment_size=32 * 1024**3,
        )
        cls.mooncake.start()
        env = {
            **cls.mooncake.server_env(),
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
            "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
        }
        cls.extra_prefill_env = env
        cls.extra_decode_env = env
        cls.launch_all()

    @classmethod
    def wait_server_ready(cls, url, timeout=1800, process=None):
        super().wait_server_ready(url, timeout=max(timeout, 1800), process=process)

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if hasattr(cls, "mooncake"):
                cls.mooncake.stop()

    def _metric(self, name):
        response = requests.get(f"{self.decode_url}/metrics", timeout=30)
        response.raise_for_status()
        return sum(
            sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name == name
            and ("pool" not in sample.labels or sample.labels["pool"] == "kv")
        )

    def _generate(self, input_ids):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def test_l3_restore_preserves_output_with_prefill_supplied_state(self):
        input_ids = list(
            sample_random_requests(
                input_len=1500,
                output_len=64,
                num_prompts=1,
                range_ratio=1.0,
                tokenizer=get_tokenizer(self.model),
                dataset_path="",
                return_text=False,
            )[0].prompt
        )
        backed_up = self._metric("sglang:backuped_tokens_total")
        baseline = self._generate(input_ids)
        deadline = time.monotonic() + 60
        while self._metric("sglang:backuped_tokens_total") <= backed_up:
            self.assertLess(time.monotonic(), deadline, "No KV reached Mooncake")
            time.sleep(0.2)

        for _ in range(2):
            # Cold prefill cannot supply a cached prefix. Cold decode can only
            # avoid PD prefix transfer by fetching its compressed pages from L3.
            for url in (self.prefill_url, self.decode_url):
                response = requests.post(f"{url}/flush_cache?timeout=60", timeout=90)
                response.raise_for_status()
            prefetched = self._metric("sglang:prefetched_tokens_total")
            loaded = self._metric("sglang:load_back_tokens_total")
            restored = self._generate(input_ids)
            self.assertEqual(restored["output_ids"], baseline["output_ids"])
            self.assertGreaterEqual(
                self._metric("sglang:prefetched_tokens_total") - prefetched, PAGE_SIZE
            )
            self.assertGreaterEqual(
                self._metric("sglang:load_back_tokens_total") - loaded, PAGE_SIZE
            )


if __name__ == "__main__":
    unittest.main()
