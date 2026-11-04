import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.kv_capacity_estimator_server import (
    KvCapacityEstimatorMetrics,
    _RequiredCapacitySlideWindow,
    _build_online_simulation_config,
    _chat_completion_payload,
    _stream_chat_completion,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvCapacityEstimatorServer(unittest.TestCase):
    def test_online_config_forces_full_trace_and_extracts_output_path(self):
        received = None

        def config_factory(values):
            nonlocal received
            received = values
            return "config"

        config, output_path, slide_window_size = _build_online_simulation_config(
            {"kv_bytes_per_token": 10, "output_path": "/tmp/result.json"},
            config_factory,
        )

        self.assertEqual(config, "config")
        self.assertEqual(output_path, "/tmp/result.json")
        self.assertEqual(slide_window_size, 3000)
        self.assertEqual(
            received,
            {"kv_bytes_per_token": 10, "warm_up_ratio": 0.0},
        )

    def test_online_config_extracts_and_validates_slide_window_size(self):
        config, _, slide_window_size = _build_online_simulation_config(
            {"kv_bytes_per_token": 10, "slide_window_size": 42},
            lambda values: values,
        )

        self.assertEqual(slide_window_size, 42)
        self.assertNotIn("slide_window_size", config)

        for invalid_value in (0, -1, 1.5, True, "3000"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _build_online_simulation_config(
                        {
                            "kv_bytes_per_token": 10,
                            "slide_window_size": invalid_value,
                        },
                        lambda values: values,
                    )

    def test_required_capacity_slide_window_uses_latest_requests(self):
        window = _RequiredCapacitySlideWindow(1000)
        for capacity_bytes in range(1, 1001):
            window.add(capacity_bytes)

        snapshot = window.snapshot()
        self.assertEqual(snapshot["requests"], 1000)
        self.assertEqual(snapshot["reachable_samples"], 1000)
        self.assertEqual(snapshot["p90_bytes"], 900)
        self.assertEqual(snapshot["p95_bytes"], 950)
        self.assertEqual(snapshot["p99_bytes"], 990)
        self.assertEqual(snapshot["p999_bytes"], 999)

        window.add(None)
        snapshot = window.snapshot()
        self.assertEqual(snapshot["requests"], 1000)
        self.assertEqual(snapshot["reachable_samples"], 999)
        self.assertEqual(snapshot["unreachable_samples"], 1)
        self.assertEqual(snapshot["p999_bytes"], 1000)

    def test_online_config_rejects_warm_up_ratio(self):
        with self.assertRaisesRegex(ValueError, "only supported by offline replay"):
            _build_online_simulation_config(
                {"kv_bytes_per_token": 10, "warm_up_ratio": 0.5},
                lambda values: values,
            )

    def test_chat_completion_payload_is_openai_compatible(self):
        request = SimpleNamespace(model="served-model")
        result = _chat_completion_payload(request, {"request_number": 3}, 42)

        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "served-model")
        self.assertEqual(
            result["choices"][0]["message"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": 42,
                "completion_tokens": 0,
                "total_tokens": 42,
            },
        )
        self.assertEqual(
            result["kv_capacity_estimation"], {"request_number": 3}
        )

    def test_stream_chat_completion_has_analysis_and_done_marker(self):
        async def collect():
            request = SimpleNamespace(
                model="served-model",
                stream_options=SimpleNamespace(include_usage=True),
            )
            return [
                chunk
                async for chunk in _stream_chat_completion(
                    request, {"request_number": 1}, prompt_tokens=7
                )
            ]

        chunks = asyncio.run(collect())
        self.assertIn(
            b'"kv_capacity_estimation":{"request_number":1}', chunks[0]
        )
        self.assertIn(b'"prompt_tokens":7', chunks[1])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    def test_prometheus_metrics_include_capacity_and_tokenizer_series(self):
        metrics = KvCapacityEstimatorMetrics(
            model_name="test-model",
            capacities=(("1GiB", 2**30),),
            extra_labels={"deployment": "test"},
            slide_window_size=3000,
        )
        metrics.set_worker_count(4)
        metrics.request_admitted()
        analysis = SimpleNamespace(
            page_accesses=3,
            reusable_accesses=2,
            infinite_prefix_hits=1,
            capacity_page_hits=(2,),
            capacity_prefix_hits=(1,),
            required_capacity_bytes=2**30,
        )
        metrics.record_success(
            is_streaming=False,
            prompt_tokens=123,
            tokenize_seconds=0.02,
            tokenizer_queue_seconds=0.01,
            simulation_seconds=0.001,
            request_seconds=0.031,
            analysis=analysis,
            capacities=(("1GiB", 2**30),),
            cumulative_page_accesses=3,
            cumulative_page_hits=[2],
            cumulative_prefix_hits=[1],
            slide_window={
                "requests": 1,
                "reachable_samples": 1,
                "p90_bytes": 2**30,
                "p95_bytes": 2**30,
                "p99_bytes": 2**30,
                "p999_bytes": 2**30,
            },
        )
        metrics.request_finished()

        output = metrics.render().decode()
        self.assertIn("sglang:prompt_tokens_total", output)
        self.assertIn("sglang:kv_capacity_estimator_tokenizer_workers", output)
        self.assertIn("sglang:kv_capacity_estimator_capacity_page_hit_rate", output)
        self.assertIn('capacity="1GiB"', output)
        self.assertIn('deployment="test"', output)
        self.assertIn(
            "sglang:kv_capacity_estimator_required_capacity_slide_window_bytes",
            output,
        )
        self.assertIn('quantile="0.9"', output)
        self.assertIn('quantile="0.999"', output)

    def test_prometheus_metrics_reject_reserved_extra_labels(self):
        with self.assertRaisesRegex(ValueError, "reserved labels"):
            KvCapacityEstimatorMetrics(
                model_name="test-model",
                capacities=(("1GiB", 2**30),),
                extra_labels={"status": "bad"},
                slide_window_size=3000,
            )


if __name__ == "__main__":
    unittest.main()
