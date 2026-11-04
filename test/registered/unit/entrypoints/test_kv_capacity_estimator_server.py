import asyncio
import unittest
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from prometheus_client.utils import floatToGoString

from sglang.srt.entrypoints.kv_capacity_estimator_server import (
    KvCapacityEstimatorMetrics,
    KvCapacityEstimatorService,
    _build_online_simulation_config,
    _chat_completion_payload,
    _stream_chat_completion,
    create_kv_capacity_estimator_app,
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

        config, output_path = _build_online_simulation_config(
            {"kv_bytes_per_token": 10, "output_path": "/tmp/result.json"},
            config_factory,
        )

        self.assertEqual(config, "config")
        self.assertEqual(output_path, "/tmp/result.json")
        self.assertEqual(
            received,
            {"kv_bytes_per_token": 10, "warm_up_ratio": 0.0},
        )

    def test_online_config_rejects_removed_request_window(self):
        with self.assertRaisesRegex(ValueError, "has been removed"):
            _build_online_simulation_config(
                {"kv_bytes_per_token": 10, "slide_window_size": 3000},
                lambda values: values,
            )

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
        self.assertEqual(result["kv_capacity_estimation"], {"request_number": 3})

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
        self.assertIn(b'"kv_capacity_estimation":{"request_number":1}', chunks[0])
        self.assertIn(b'"prompt_tokens":7', chunks[1])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    def test_prometheus_metrics_include_capacity_and_tokenizer_series(self):
        metrics = KvCapacityEstimatorMetrics(
            model_name="test-model",
            capacities=(("1GiB", 2**30),),
            extra_labels={"deployment": "test"},
            page_size=64,
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
        )
        metrics.request_finished()

        output = metrics.render().decode()
        self.assertIn("sglang:prompt_tokens_total", output)
        self.assertIn("sglang:kv_capacity_estimator_tokenizer_workers", output)
        self.assertIn("sglang:kv_capacity_estimator_capacity_page_hit_rate", output)
        self.assertIn('capacity="1GiB"', output)
        self.assertIn('deployment="test"', output)
        self.assertNotIn("slide_window", output)
        self.assertNotIn("quantile=", output)
        labels = metrics._labels
        capacity_labels = metrics._capacity_labels("1GiB", 2**30)
        self.assertEqual(
            metrics.registry.get_sample_value(
                "sglang:kv_capacity_estimator_capacity_prefix_hit_tokens_total",
                capacity_labels,
            ),
            64,
        )
        self.assertEqual(
            metrics.registry.get_sample_value(
                "sglang:kv_capacity_estimator_required_capacity_bytes_bucket",
                {**labels, "le": floatToGoString(2**30)},
            ),
            1,
        )

    def test_cumulative_histogram_and_tokens_include_zero_partial_and_unreachable(self):
        metrics = KvCapacityEstimatorMetrics(
            model_name="test",
            capacities=(("custom", 123), ("large", 2**46)),
            extra_labels=None,
            page_size=64,
        )
        labels = metrics._labels
        name = "sglang:kv_capacity_estimator_"

        def sample(suffix, **extra):
            return metrics.registry.get_sample_value(name + suffix, {**labels, **extra})

        # These children must exist before traffic, so a scrape can capture zero.
        self.assertEqual(sample("required_capacity_bytes_count"), 0)
        self.assertEqual(sample("required_capacity_unreachable_total"), 0)
        self.assertEqual(sample("required_capacity_bytes_bucket", le="123.0"), 0)
        self.assertEqual(
            sample("required_capacity_bytes_bucket", le=floatToGoString(2**46)), 0
        )
        for requirement in (0, 123, 2**46, None):
            metrics.record_success(
                is_streaming=True,
                prompt_tokens=65,
                tokenize_seconds=0,
                tokenizer_queue_seconds=0,
                simulation_seconds=0,
                request_seconds=0,
                analysis=SimpleNamespace(
                    page_accesses=2,
                    reusable_accesses=2,
                    infinite_prefix_hits=2,
                    capacity_page_hits=(2, 2),
                    capacity_prefix_hits=(2, 2),
                    required_capacity_bytes=requirement,
                ),
                capacities=(("custom", 123), ("large", 2**46)),
                cumulative_page_accesses=2,
                cumulative_page_hits=[2, 2],
                cumulative_prefix_hits=[2, 2],
            )
        self.assertEqual(sample("required_capacity_bytes_count"), 3)
        self.assertEqual(sample("required_capacity_bytes_bucket", le="0.0"), 1)
        self.assertEqual(sample("required_capacity_bytes_bucket", le="123.0"), 2)
        self.assertEqual(sample("required_capacity_bytes_bucket", le="+Inf"), 3)
        self.assertEqual(sample("required_capacity_unreachable_total"), 1)
        self.assertEqual(sample("infinite_prefix_hit_tokens_total"), 260)
        self.assertEqual(
            sample(
                "capacity_prefix_hit_tokens_total",
                capacity="custom",
                capacity_bytes="123",
            ),
            260,
        )
        metrics.record_error(request_seconds=0)
        self.assertEqual(sample("required_capacity_bytes_count"), 3)
        self.assertEqual(sample("requests_total", status="success"), 4)
        self.assertEqual(sample("requests_total", status="error"), 1)

    def test_metrics_follow_real_simulator_without_resetting_history(self):
        from kv_capacity_estimator import ReplaySimulator, simulation_config_from_dict
        from kv_capacity_estimator.models import TokenIdsRequest

        config = simulation_config_from_dict(
            {
                "kv_bytes_per_token": 1,
                "page_size": 2,
                "capacities": [["two-pages", 4]],
                "include_partial_page": True,
            }
        )
        simulator = ReplaySimulator(config)
        metrics = KvCapacityEstimatorMetrics(
            model_name="test",
            capacities=tuple(config.capacities),
            extra_labels=None,
            page_size=config.page_size,
        )
        # A cold request followed by a repeat including a partial final page.
        for _ in range(2):
            analysis = simulator.process(TokenIdsRequest(input_ids=(1, 2, 3)))
            metrics.record_success(
                is_streaming=False,
                prompt_tokens=3,
                tokenize_seconds=0,
                tokenizer_queue_seconds=0,
                simulation_seconds=0,
                request_seconds=0,
                analysis=analysis,
                capacities=tuple(config.capacities),
                cumulative_page_accesses=2,
                cumulative_page_hits=list(analysis.capacity_page_hits),
                cumulative_prefix_hits=list(analysis.capacity_prefix_hits),
            )
        self.assertEqual(
            metrics.registry.get_sample_value(
                "sglang:kv_capacity_estimator_capacity_prefix_hit_tokens_total",
                metrics._capacity_labels("two-pages", 4),
            ),
            3,
        )
        self.assertEqual(
            metrics.registry.get_sample_value(
                "sglang:prompt_tokens_total",
                {**metrics._labels, "is_streaming": "false"},
            ),
            6,
        )
        self.assertEqual(simulator.finish().total_requests, 2)

    def test_prometheus_metrics_reject_reserved_extra_labels(self):
        for label in ("status", "le", "capacity_bytes", "quantile"):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "reserved labels"):
                    KvCapacityEstimatorMetrics(
                        model_name="test-model",
                        capacities=(("1GiB", 2**30),),
                        extra_labels={label: "bad"},
                        page_size=64,
                    )


class TestKvCapacityEstimatorReset(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_dir.cleanup)
        self.output_path = Path(self.output_dir.name) / "result.json"
        cfg = SimpleNamespace(
            tokenizer_worker_num=2,
            kv_capacity_estimator_config={
                "kv_bytes_per_token": 1,
                "page_size": 2,
                "capacities": [["small", 4]],
                "output_path": str(self.output_path),
            },
            served_model_name="test",
            extra_metric_labels={"deployment": "reset-test"},
            enable_metrics=True,
        )
        # Exercise the real simulator and service without loading a model.
        self.globals = KvCapacityEstimatorService.__init__.__globals__
        with patch.dict(
            self.globals,
            resolving_view=lambda args: cfg,
            ProcessPoolExecutor=lambda **kwargs: ThreadPoolExecutor(max_workers=2),
        ):
            self.service = KvCapacityEstimatorService(None)
        self.addAsyncCleanup(self.service.close)
        self.service.metrics.set_worker_count(2)
        self.request = SimpleNamespace(stream=False, model_dump=lambda **kwargs: {})
        self.tokenizer_patch = patch.dict(
            self.globals, _tokenize_chat_request=lambda payload: ((b"page",), 2, 0.01)
        )
        self.tokenizer_patch.start()
        self.addCleanup(self.tokenizer_patch.stop)

    async def test_reset_after_finalize_starts_a_cold_independent_run(self):
        service = self.service
        await service.analyze(self.request)
        repeated, _ = await service.analyze(self.request)
        self.assertEqual(repeated["infinite_prefix_hits"], 1)
        old_result = await service.finalize()
        saved_output = self.output_path.read_text()
        with self.assertRaises(self.globals["HTTPException"]):
            await service.analyze(self.request)
        tokenizer_pool = service._executor
        status = await service.reset()
        self.assertFalse(status["finalized"])
        self.assertEqual(status["requests"], 0)
        self.assertEqual(status["metrics"]["successful_requests"], 0)
        self.assertEqual(status["tokenize_worker_seconds"], 0)
        self.assertIs(service._executor, tokenizer_pool)
        self.assertEqual(self.output_path.read_text(), saved_output)
        metrics = service.metrics
        for suffix in ("requests_total", "required_capacity_bytes_count"):
            labels = dict(metrics._labels)
            if suffix == "requests_total":
                labels["status"] = "success"
            self.assertEqual(
                metrics.registry.get_sample_value(
                    "sglang:kv_capacity_estimator_" + suffix, labels
                ),
                0,
            )
        self.assertEqual(
            metrics.registry.get_sample_value(
                "sglang:kv_capacity_estimator_tokenizer_workers", metrics._labels
            ),
            2,
        )
        cold, _ = await service.analyze(self.request)
        self.assertEqual(cold["request_number"], 1)
        self.assertEqual(cold["infinite_prefix_hits"], 0)
        result = await service.finalize()
        self.assertEqual(result["total_requests"], 1)
        self.assertEqual(old_result["total_requests"], 2)
        self.assertEqual(await service.finalize(), result)

    async def test_reset_drains_requests_and_serializes_finalize(self):
        entered = threading.Event()
        release = threading.Event()

        def tokenize(payload):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test tokenizer was not released")
            return (b"page",), 2, 0.01

        with patch.dict(self.globals, _tokenize_chat_request=tokenize):
            request = asyncio.create_task(self.service.analyze(self.request))
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            reset = asyncio.create_task(self.service.reset())
            try:
                async with asyncio.timeout(5):
                    while self.service._accepting_requests:
                        await asyncio.sleep(0)
                self.assertFalse(reset.done())
                with self.assertRaises(self.globals["HTTPException"]) as error:
                    await self.service.analyze(self.request)
                self.assertEqual(error.exception.status_code, 409)
                finalized = asyncio.create_task(self.service.finalize())
            finally:
                release.set()
            await request
            fresh = await reset
            self.assertEqual(fresh["requests"], 0)
            self.assertEqual((await finalized)["total_requests"], 0)
            self.assertEqual(self.service._pending_requests, 0)

    async def test_reset_endpoint_requires_api_key_and_returns_fresh_status(self):
        import httpx

        await self.service.analyze(self.request)
        await self.service.finalize()
        with patch.dict(
            self.globals,
            publish=lambda *args, **kwargs: None,
            resolving_view=lambda args: SimpleNamespace(api_key="test-key"),
            KvCapacityEstimatorService=lambda args: self.service,
        ):
            app = create_kv_capacity_estimator_app(None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            unauthorized = await client.post("/kv-capacity-estimator/reset")
            self.assertEqual(unauthorized.status_code, 401)
            self.assertTrue((await self.service.status())["finalized"])
            response = await client.post(
                "/kv-capacity-estimator/reset",
                headers={"Authorization": "Bearer test-key"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requests"], 0)
        self.assertFalse(response.json()["finalized"])
        result, _ = await self.service.analyze(self.request)
        self.assertEqual(result["request_number"], 1)

    async def test_reset_without_metrics_and_repeated_reset(self):
        self.service.metrics = None
        await self.service.analyze(self.request)
        for _ in range(2):
            status = await self.service.reset()
            self.assertFalse(status["prometheus_metrics_enabled"])
            self.assertEqual(status["requests"], 0)
        cold, _ = await self.service.analyze(self.request)
        self.assertEqual(cold["request_number"], 1)
        self.assertEqual(cold["infinite_prefix_hits"], 0)

    async def test_disconnected_reset_still_completes(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.service._do_reset

        def blocked_reset():
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test reset was not released")
            original()

        with patch.object(self.service, "_do_reset", blocked_reset):
            reset = asyncio.create_task(self.service.reset())
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            reset.cancel()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await reset
                self.assertFalse(self.service._accepting_requests)
            finally:
                release.set()
            async with asyncio.timeout(5):
                while not self.service._accepting_requests:
                    await asyncio.sleep(0)
        cold, _ = await self.service.analyze(self.request)
        self.assertEqual(cold["request_number"], 1)


if __name__ == "__main__":
    unittest.main()
