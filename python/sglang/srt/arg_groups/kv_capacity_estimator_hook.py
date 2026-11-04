# SPDX-License-Identifier: Apache-2.0
"""Resolve the tokenizer-only simulator without initializing inference hardware."""

from typing import Any

from sglang.srt.arg_groups.model_path_hook import handle_model_source_paths
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.resolution_hooks import run_hook
from sglang.srt.arg_groups.serving_hook import (
    handle_media_url_security,
    handle_missing_default_values,
    handle_multimodal,
    handle_ssl_validation,
    handle_tokenizer_batching,
)


def resolve_kv_capacity_estimator(server_args: Any) -> None:
    # CPU is metadata for the tokenizer processes, not the CPU inference
    # backend. Even an explicitly supplied accelerator must not initialize
    # hardware or select inference kernels in simulator mode.
    declare_resolution(server_args, "kv_capacity_estimator", device="cpu")
    run_hook(handle_model_source_paths, server_args)
    run_hook(handle_missing_default_values, server_args)
    run_hook(handle_media_url_security, server_args)
    run_hook(handle_multimodal, server_args)
    run_hook(handle_ssl_validation, server_args)
    run_hook(handle_tokenizer_batching, server_args)
