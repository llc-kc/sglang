"""Tokenizer-only simulation must resolve without inference hardware or weights."""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKvCapacityEstimatorResolution(unittest.TestCase):
    def test_simulator_skips_hardware_and_model_resolution(self):
        for device in (None, "cuda", "mps"):
            with self.subTest(device=device), patch(
                "sglang.srt.arg_groups.serving_hook.get_device",
                side_effect=AssertionError("accelerator detection"),
            ), patch(
                "sglang.srt.arg_groups.platform_hook.handle_hardware_runtime_validation",
                side_effect=AssertionError("hardware validation"),
            ), patch(
                "sglang.srt.arg_groups.memory_hook.handle_gpu_memory_settings",
                side_effect=AssertionError("GPU memory sizing"),
            ), patch(
                "sglang.srt.arg_groups.overrides.model_config_of",
                side_effect=AssertionError("inference model configuration"),
            ), patch(
                "sglang.srt.server_args.validate_mps_runtime",
                side_effect=AssertionError("MPS validation"),
            ):
                args = ServerArgs(
                    model_path="/local/tokenizer-only-model",
                    kv_capacity_estimator=True,
                    device=device,
                    tokenizer_worker_num=2,
                    tool_call_parser="glm47",
                    enable_metrics=True,
                )
                args.resolve_once()
                args.resolve_once()
                cfg = resolving_view(args)
                self.assertEqual(cfg.device, "cpu")
                self.assertEqual(cfg.tokenizer_path, args.model_path)
                self.assertEqual(cfg.served_model_name, args.model_path)
                self.assertEqual(cfg.tokenizer_worker_num, 2)
                self.assertEqual(cfg.tool_call_parser, "glm47")
                self.assertTrue(cfg.enable_metrics)
                self.assertEqual(args.device, device)

    def test_custom_tokenizer_and_served_name_are_preserved(self):
        args = ServerArgs(
            model_path="/local/model",
            tokenizer_path="/local/tokenizer",
            served_model_name="simulated-model",
            kv_capacity_estimator=True,
        )
        args.resolve_once()
        cfg = resolving_view(args)
        self.assertEqual(cfg.tokenizer_path, "/local/tokenizer")
        self.assertEqual(cfg.served_model_name, "simulated-model")

    def test_simulator_still_validates_http_configuration(self):
        args = ServerArgs(
            model_path="/local/model",
            kv_capacity_estimator=True,
            ssl_keyfile="/missing/key.pem",
        )
        with self.assertRaisesRegex(ValueError, "requires --ssl-certfile"):
            args.resolve_once()


if __name__ == "__main__":
    unittest.main()
