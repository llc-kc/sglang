# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tokenizer-only HTTP server backed by kv-capacity-estimator.

This is deliberately a separate launch path from the SRT engine.  It loads the
model configuration, tokenizer, and chat template, but never constructs a
scheduler, detokenizer, model runner, or weight loader.
"""

from __future__ import annotations

import asyncio
import bisect
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import secrets
import time
import uuid
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import orjson
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import get_model, get_serving, publish
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_TOKENIZER_RUNTIME = None
_DEFAULT_SLIDE_WINDOW_SIZE = 3000
_CAPACITY_PERCENTILES = (
    ("p90", "0.9", 0.9),
    ("p95", "0.95", 0.95),
    ("p99", "0.99", 0.99),
    ("p999", "0.999", 0.999),
)


def _load_simulator_api():
    try:
        from kv_capacity_estimator import (
            ReplaySimulator,
            TokenIdsRequest,
            simulation_config_from_dict,
        )
    except ImportError as error:
        raise RuntimeError(
            "--kv-capacity-estimator requires the kv-capacity-estimator "
            "wheel. Install it in the SGLang environment, for example: "
            "pip install kv_capacity_estimator-0.2.0-py3-none-any.whl"
        ) from error
    return ReplaySimulator, TokenIdsRequest, simulation_config_from_dict


def _build_online_simulation_config(raw_config, config_factory):
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("--kv-capacity-estimator-config must be a JSON object")
    simulator_values = dict(raw_config)
    output_path = simulator_values.pop("output_path", None)
    slide_window_size = simulator_values.pop(
        "slide_window_size", _DEFAULT_SLIDE_WINDOW_SIZE
    )
    if type(slide_window_size) is not int or slide_window_size < 1:
        raise ValueError(
            "kv-capacity-estimator config 'slide_window_size' must be a "
            "positive integer"
        )
    if "warm_up_ratio" in simulator_values:
        raise ValueError(
            "kv-capacity-estimator config 'warm_up_ratio' is only supported "
            "by offline replay"
        )
    # Online traffic has no known end-of-trace boundary from which to derive a
    # warm-up request prefix. Keep live and final metrics aligned by measuring
    # every admitted request in SGLang server mode.
    simulator_values["warm_up_ratio"] = 0.0
    if output_path is not None and not isinstance(output_path, str):
        raise ValueError(
            "kv-capacity-estimator config 'output_path' must be a string"
        )
    try:
        simulation_config = config_factory(simulator_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid --kv-capacity-estimator-config: {error}") from error
    return simulation_config, output_path, slide_window_size


class _RequiredCapacitySlideWindow:
    """Exact percentiles over the latest successfully simulated requests."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._requests: deque[int | None] = deque()
        self._ordered_capacities: list[int] = []

    def add(self, capacity_bytes: int | None) -> None:
        if len(self._requests) == self.size:
            evicted = self._requests.popleft()
            if evicted is not None:
                index = bisect.bisect_left(self._ordered_capacities, evicted)
                self._ordered_capacities.pop(index)

        self._requests.append(capacity_bytes)
        if capacity_bytes is not None:
            bisect.insort(self._ordered_capacities, capacity_bytes)

    def snapshot(self) -> dict[str, Any]:
        sample_count = len(self._ordered_capacities)
        percentiles = {}
        for name, _, percentile in _CAPACITY_PERCENTILES:
            percentiles[f"{name}_bytes"] = (
                self._ordered_capacities[
                    max(0, math.ceil(percentile * sample_count) - 1)
                ]
                if sample_count
                else None
            )
        return {
            "size": self.size,
            "requests": len(self._requests),
            "reachable_samples": sample_count,
            "unreachable_samples": len(self._requests) - sample_count,
            **percentiles,
        }


def _create_tokenizer_only_manager(server_args: ServerArgs) -> TokenizerManager:
    """Initialize only the TokenizerManager pieces used by chat rendering."""
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = server_args
    manager.init_model_config()
    manager.init_tokenizer_and_processor()
    if manager.tokenizer is None:
        raise ValueError(
            "--kv-capacity-estimator cannot be combined with --skip-tokenizer-init"
        )
    return manager


def _resolve_auto_parsers(manager, template_manager) -> None:
    """Match the parser auto-detection performed by the normal engine path."""
    for attr, suggested, label in (
        (
            "reasoning_parser",
            template_manager.suggested_reasoning_parser,
            "reasoning parser",
        ),
        (
            "tool_call_parser",
            template_manager.suggested_tool_call_parser,
            "tool-call parser",
        ),
    ):
        if manager.config_value(attr) != "auto":
            continue
        manager.record_config_updates("template-detection", **{attr: suggested})
        if suggested is None:
            logger.warning(
                "--%s=auto was not detected from the chat template; disabling %s",
                attr.replace("_", "-"),
                label,
            )
        else:
            logger.info(
                "Auto-detected --%s as %r from the chat template",
                attr.replace("_", "-"),
                suggested,
            )


def _create_tokenizer_runtime(server_args: ServerArgs):
    manager = _create_tokenizer_only_manager(server_args)
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=manager,
        model_path=get_model().model_path,
        chat_template=get_serving().chat_template,
        completion_template=get_serving().completion_template,
    )
    _resolve_auto_parsers(manager, template_manager)
    return manager, manager.serving_chat_class(manager, template_manager)


def _init_tokenizer_worker(server_args: ServerArgs) -> None:
    """Initialize one process-local SGLang tokenizer and serving pipeline."""
    global _TOKENIZER_RUNTIME
    server_args.resolve_once()
    publish(server_args, role="tokenizer")
    _TOKENIZER_RUNTIME = _create_tokenizer_runtime(server_args)


def _tokenizer_worker_ping() -> int:
    if _TOKENIZER_RUNTIME is None:
        raise RuntimeError("KV-cache tokenizer worker was not initialized")
    return os.getpid()


def _extract_token_ids(manager, processed) -> list[int]:
    prompt_ids = processed.prompt_ids
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if isinstance(prompt_ids, list) and (prompt_ids or not processed.prompt):
        token_ids = prompt_ids
    elif isinstance(prompt_ids, str):
        token_ids = manager.tokenizer.encode(prompt_ids, add_special_tokens=False)
    elif processed.prompt:
        token_ids = manager.tokenizer.encode(
            processed.prompt, add_special_tokens=False
        )
    else:
        raise ValueError("SGLang chat rendering did not produce token IDs")

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise ValueError("SGLang tokenizer returned invalid token IDs")
    return token_ids


def _tokenize_chat_request(request_payload: dict[str, Any]):
    if _TOKENIZER_RUNTIME is None:
        raise RuntimeError("KV-cache tokenizer worker was not initialized")
    manager, chat_serving = _TOKENIZER_RUNTIME
    request = ChatCompletionRequest.model_validate(request_payload)
    validation_error = chat_serving._validate_request(request)
    if validation_error:
        raise ValueError(validation_error)

    started_at = time.perf_counter()
    processed = chat_serving._process_messages(
        request, manager.model_config.is_multimodal
    )
    if processed.image_data or processed.video_data or processed.audio_data:
        raise ValueError(
            "KV capacity estimator mode currently supports text-only chat requests; "
            "multimodal processor expansion requires the model execution pipeline"
        )
    token_ids = _extract_token_ids(manager, processed)
    return token_ids, time.perf_counter() - started_at


class KvCapacityEstimatorMetrics:
    """Prometheus metrics owned by the tokenizer/simulator HTTP process."""

    _PROMPT_TOKEN_BUCKETS = (
        100,
        300,
        500,
        700,
        1_000,
        2_000,
        4_000,
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_100_000,
    )

    # Power-of-two bounds from 256 MiB (2**28) to 4 TiB (2**42). Histogram
    # buckets must be declared up front, so one bound per doubling is the
    # coarsest resolution that stays accurate across the whole range.
    _REQUIRED_CAPACITY_BUCKETS = tuple(
        float(2**exponent) for exponent in range(28, 43)
    )

    def __init__(
        self,
        *,
        model_name: str,
        capacities: tuple[tuple[str, int], ...],
        extra_labels: dict[str, str] | None,
        slide_window_size: int,
    ) -> None:
        # Keep this import lazy so simulator mode still starts when metrics are
        # disabled and prometheus-client is not installed.
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        reserved_labels = {
            "model_name",
            "engine_type",
            "is_streaming",
            "status",
            "capacity",
            "capacity_bytes",
            "quantile",
        }
        conflicting_labels = reserved_labels.intersection(extra_labels or {})
        if conflicting_labels:
            raise ValueError(
                "--extra-metric-labels uses reserved labels: "
                + ", ".join(sorted(conflicting_labels))
            )
        self.registry = CollectorRegistry()
        self._labels = {
            "model_name": model_name,
            "engine_type": "kv_capacity_estimator",
            **(extra_labels or {}),
        }
        label_names = list(self._labels)

        # These names match regular SGLang tokenizer metrics so existing
        # dashboards continue to show request and prompt-token traffic.
        self.prompt_tokens_total = Counter(
            "sglang:prompt_tokens_total",
            "Number of prompt tokens processed.",
            [*label_names, "is_streaming"],
            registry=self.registry,
        )
        self.prompt_tokens_histogram = Histogram(
            "sglang:prompt_tokens_histogram",
            "Histogram of prompt token length.",
            label_names,
            buckets=self._PROMPT_TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.num_requests_total = Counter(
            "sglang:num_requests_total",
            "Number of requests processed.",
            [*label_names, "is_streaming"],
            registry=self.registry,
        )

        prefix = "sglang:kv_capacity_estimator_"
        self.requests_total = Counter(
            f"{prefix}requests_total",
            "Number of simulator requests by outcome.",
            [*label_names, "status"],
            registry=self.registry,
        )
        self.pending_requests = Gauge(
            f"{prefix}pending_requests",
            "Number of admitted requests awaiting completion.",
            label_names,
            registry=self.registry,
        )
        self.tokenizer_workers = Gauge(
            f"{prefix}tokenizer_workers",
            "Configured tokenizer worker process count.",
            label_names,
            registry=self.registry,
        )
        self.tokenize_seconds = Histogram(
            f"{prefix}tokenize_seconds",
            "Tokenizer worker processing time in seconds.",
            label_names,
            registry=self.registry,
        )
        self.tokenizer_queue_seconds = Histogram(
            f"{prefix}tokenizer_queue_seconds",
            "Tokenizer executor queue and IPC overhead in seconds.",
            label_names,
            registry=self.registry,
        )
        self.simulation_seconds = Histogram(
            f"{prefix}simulation_seconds",
            "Online cache simulation processing time in seconds.",
            label_names,
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            f"{prefix}request_seconds",
            "End-to-end simulator request time in seconds.",
            [*label_names, "status"],
            registry=self.registry,
        )
        self.page_accesses_total = Counter(
            f"{prefix}page_accesses_total",
            "Number of simulated KV-cache page accesses.",
            label_names,
            registry=self.registry,
        )
        self.reusable_page_hits_total = Counter(
            f"{prefix}reusable_page_hits_total",
            "Number of page hits possible with unlimited cache capacity.",
            label_names,
            registry=self.registry,
        )
        self.infinite_prefix_hits_total = Counter(
            f"{prefix}infinite_prefix_hits_total",
            "Number of prefix hits possible with unlimited cache capacity.",
            label_names,
            registry=self.registry,
        )
        capacity_label_names = [*label_names, "capacity", "capacity_bytes"]
        self.capacity_page_hits_total = Counter(
            f"{prefix}capacity_page_hits_total",
            "Number of page hits at each configured cache capacity.",
            capacity_label_names,
            registry=self.registry,
        )
        self.capacity_prefix_hits_total = Counter(
            f"{prefix}capacity_prefix_hits_total",
            "Number of prefix hits at each configured cache capacity.",
            capacity_label_names,
            registry=self.registry,
        )
        self.capacity_page_hit_rate = Gauge(
            f"{prefix}capacity_page_hit_rate",
            "Cumulative page hit rate at each configured cache capacity.",
            capacity_label_names,
            registry=self.registry,
        )
        self.capacity_prefix_hit_rate = Gauge(
            f"{prefix}capacity_prefix_hit_rate",
            "Cumulative prefix hit rate at each configured cache capacity.",
            capacity_label_names,
            registry=self.registry,
        )
        self.required_capacity_bytes = Histogram(
            f"{prefix}required_capacity_bytes",
            "Per-request minimum KV-cache capacity in bytes required to "
            "reach the target hit rate; unreachable requests are excluded.",
            label_names,
            buckets=self._REQUIRED_CAPACITY_BUCKETS,
            registry=self.registry,
        )
        self.required_capacity_unreachable_total = Counter(
            f"{prefix}required_capacity_unreachable_total",
            "Requests for which the target hit rate was not reachable.",
            label_names,
            registry=self.registry,
        )
        self.required_capacity_slide_window_bytes = Gauge(
            f"{prefix}required_capacity_slide_window_bytes",
            "Nearest-rank required-capacity percentile over the latest "
            f"{slide_window_size} successful requests; unreachable requests "
            "are excluded.",
            [*label_names, "quantile"],
            registry=self.registry,
        )
        self.required_capacity_slide_window_requests = Gauge(
            f"{prefix}required_capacity_slide_window_requests",
            "Number of successful requests currently represented in the "
            "required-capacity slide window.",
            label_names,
            registry=self.registry,
        )
        self.required_capacity_slide_window_reachable_samples = Gauge(
            f"{prefix}required_capacity_slide_window_reachable_samples",
            "Number of requests with a reachable capacity in the required-capacity "
            "slide window.",
            label_names,
            registry=self.registry,
        )
        self.required_capacity_slide_window_size = Gauge(
            f"{prefix}required_capacity_slide_window_size",
            "Configured maximum number of requests in the required-capacity "
            "slide window.",
            label_names,
            registry=self.registry,
        )

        self.tokenizer_workers.labels(**self._labels).set(0)
        self.required_capacity_slide_window_size.labels(**self._labels).set(
            slide_window_size
        )
        self.required_capacity_slide_window_requests.labels(**self._labels).set(0)
        self.required_capacity_slide_window_reachable_samples.labels(
            **self._labels
        ).set(0)
        for _, quantile, _ in _CAPACITY_PERCENTILES:
            self.required_capacity_slide_window_bytes.labels(
                **self._labels, quantile=quantile
            ).set(float("nan"))
        for label, capacity_bytes in capacities:
            capacity_labels = self._capacity_labels(label, capacity_bytes)
            self.capacity_page_hit_rate.labels(**capacity_labels).set(0)
            self.capacity_prefix_hit_rate.labels(**capacity_labels).set(0)

    def _capacity_labels(self, label: str, capacity_bytes: int) -> dict[str, str]:
        return {
            **self._labels,
            "capacity": label,
            "capacity_bytes": str(capacity_bytes),
        }

    def set_worker_count(self, worker_count: int) -> None:
        self.tokenizer_workers.labels(**self._labels).set(worker_count)

    def request_admitted(self) -> None:
        self.pending_requests.labels(**self._labels).inc()

    def request_finished(self) -> None:
        self.pending_requests.labels(**self._labels).dec()

    def record_error(self, *, request_seconds: float) -> None:
        labels = {**self._labels, "status": "error"}
        self.requests_total.labels(**labels).inc()
        self.request_seconds.labels(**labels).observe(request_seconds)

    def record_success(
        self,
        *,
        is_streaming: bool,
        prompt_tokens: int,
        tokenize_seconds: float,
        tokenizer_queue_seconds: float,
        simulation_seconds: float,
        request_seconds: float,
        analysis,
        capacities: tuple[tuple[str, int], ...],
        cumulative_page_accesses: int,
        cumulative_page_hits: list[int],
        cumulative_prefix_hits: list[int],
        slide_window: dict[str, Any],
    ) -> None:
        stream_labels = {
            **self._labels,
            "is_streaming": str(is_streaming).lower(),
        }
        self.prompt_tokens_total.labels(**stream_labels).inc(prompt_tokens)
        self.prompt_tokens_histogram.labels(**self._labels).observe(prompt_tokens)
        self.num_requests_total.labels(**stream_labels).inc()
        success_labels = {**self._labels, "status": "success"}
        self.requests_total.labels(**success_labels).inc()
        self.request_seconds.labels(**success_labels).observe(request_seconds)
        self.tokenize_seconds.labels(**self._labels).observe(tokenize_seconds)
        self.tokenizer_queue_seconds.labels(**self._labels).observe(
            tokenizer_queue_seconds
        )
        self.simulation_seconds.labels(**self._labels).observe(simulation_seconds)
        self.page_accesses_total.labels(**self._labels).inc(analysis.page_accesses)
        self.reusable_page_hits_total.labels(**self._labels).inc(
            analysis.reusable_accesses
        )
        self.infinite_prefix_hits_total.labels(**self._labels).inc(
            analysis.infinite_prefix_hits
        )
        for index, (label, capacity_bytes) in enumerate(capacities):
            capacity_labels = self._capacity_labels(label, capacity_bytes)
            self.capacity_page_hits_total.labels(**capacity_labels).inc(
                analysis.capacity_page_hits[index]
            )
            self.capacity_prefix_hits_total.labels(**capacity_labels).inc(
                analysis.capacity_prefix_hits[index]
            )
            denominator = cumulative_page_accesses
            self.capacity_page_hit_rate.labels(**capacity_labels).set(
                cumulative_page_hits[index] / denominator if denominator else 0
            )
            self.capacity_prefix_hit_rate.labels(**capacity_labels).set(
                cumulative_prefix_hits[index] / denominator if denominator else 0
            )
        if analysis.required_capacity_bytes is None:
            self.required_capacity_unreachable_total.labels(**self._labels).inc()
        else:
            self.required_capacity_bytes.labels(**self._labels).observe(
                analysis.required_capacity_bytes
            )
        self.required_capacity_slide_window_requests.labels(**self._labels).set(
            slide_window["requests"]
        )
        self.required_capacity_slide_window_reachable_samples.labels(
            **self._labels
        ).set(slide_window["reachable_samples"])
        for name, quantile, _ in _CAPACITY_PERCENTILES:
            value = slide_window[f"{name}_bytes"]
            self.required_capacity_slide_window_bytes.labels(
                **self._labels, quantile=quantile
            ).set(value if value is not None else float("nan"))

    def render(self) -> bytes:
        from prometheus_client import generate_latest

        return generate_latest(self.registry)


class KvCapacityEstimatorService:
    """Parallelize tokenization and maintain one central online LRU simulator."""

    def __init__(self, server_args: ServerArgs) -> None:
        (
            ReplaySimulator,
            TokenIdsRequest,
            simulation_config_from_dict,
        ) = _load_simulator_api()
        cfg = resolving_view(server_args)

        if cfg.tokenizer_worker_num < 1:
            raise ValueError("--tokenizer-worker-num must be at least 1")
        simulation_config, output_path, slide_window_size = (
            _build_online_simulation_config(
                cfg.kv_capacity_estimator_config,
                simulation_config_from_dict,
            )
        )
        capacities = tuple(simulation_config.capacities)

        self.simulator = ReplaySimulator(simulation_config)
        self.TokenIdsRequest = TokenIdsRequest
        self.capacities = capacities
        self.simulation_config = simulation_config
        self.output_path = Path(output_path) if output_path else None
        self.served_model_name = cfg.served_model_name or cfg.model_path
        self.tokenizer_worker_num = cfg.tokenizer_worker_num
        self._required_capacity_slide_window = _RequiredCapacitySlideWindow(
            slide_window_size
        )
        self.metrics = (
            KvCapacityEstimatorMetrics(
                model_name=self.served_model_name,
                capacities=capacities,
                extra_labels=cfg.extra_metric_labels,
                slide_window_size=slide_window_size,
            )
            if cfg.enable_metrics
            else None
        )
        self._executor = ProcessPoolExecutor(
            max_workers=self.tokenizer_worker_num,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_tokenizer_worker,
            initargs=(server_args,),
        )
        self._condition = asyncio.Condition()
        self._accepting_requests = True
        self._pending_requests = 0
        self._tokenize_worker_seconds = 0.0
        self._successful_requests = 0
        self._failed_requests = 0
        self._prompt_tokens_total = 0
        self._page_accesses_total = 0
        self._reusable_page_hits_total = 0
        self._infinite_prefix_hits_total = 0
        self._capacity_page_hits = [0] * len(capacities)
        self._capacity_prefix_hits = [0] * len(capacities)
        self._required_capacity_samples = 0
        self._required_capacity_sum = 0
        self._required_capacity_unreachable = 0
        self._last_required_capacity_bytes = None
        self._final_result = None
        self._closed = False

    async def start(self) -> None:
        """Start every tokenizer process and surface initialization errors."""
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self._executor, _tokenizer_worker_ping)
            for _ in range(self.tokenizer_worker_num)
        ]
        worker_pids = set(await asyncio.gather(*futures))
        if self.metrics is not None:
            self.metrics.set_worker_count(self.tokenizer_worker_num)
        logger.info(
            "Initialized KV-cache tokenizer pool with %d process(es); startup "
            "checks ran on PIDs %s",
            self.tokenizer_worker_num,
            sorted(worker_pids),
        )

    async def analyze(
        self, request: ChatCompletionRequest
    ) -> tuple[dict[str, Any], int]:
        """Tokenize in a worker and update the central simulator on completion."""
        request_started_at = time.perf_counter()
        async with self._condition:
            if not self._accepting_requests:
                raise HTTPException(
                    status_code=409,
                    detail="KV capacity estimation has already been finalized",
                )
            self._pending_requests += 1
            if self.metrics is not None:
                self.metrics.request_admitted()

        try:
            loop = asyncio.get_running_loop()
            tokenizer_submitted_at = time.perf_counter()
            token_ids, tokenize_seconds = await loop.run_in_executor(
                self._executor,
                _tokenize_chat_request,
                request.model_dump(mode="json"),
            )
            tokenizer_roundtrip_seconds = (
                time.perf_counter() - tokenizer_submitted_at
            )
            async with self._condition:
                # Results intentionally enter the LRU trace in tokenizer
                # completion order. This keeps one global simulator while
                # allowing independent tokenizer processes to run concurrently.
                self._tokenize_worker_seconds += tokenize_seconds
                simulation_started_at = time.perf_counter()
                analysis = self.simulator.process(
                    self.TokenIdsRequest(input_ids=tuple(token_ids))
                )
                simulation_seconds = time.perf_counter() - simulation_started_at
                request_number = self.simulator.request_count
                self._record_analysis(analysis, prompt_tokens=len(token_ids))
                slide_window = self._required_capacity_slide_window.snapshot()
                if self.metrics is not None:
                    self.metrics.record_success(
                        is_streaming=request.stream,
                        prompt_tokens=len(token_ids),
                        tokenize_seconds=tokenize_seconds,
                        tokenizer_queue_seconds=max(
                            0, tokenizer_roundtrip_seconds - tokenize_seconds
                        ),
                        simulation_seconds=simulation_seconds,
                        request_seconds=time.perf_counter() - request_started_at,
                        analysis=analysis,
                        capacities=self.capacities,
                        cumulative_page_accesses=self._page_accesses_total,
                        cumulative_page_hits=self._capacity_page_hits,
                        cumulative_prefix_hits=self._capacity_prefix_hits,
                        slide_window=slide_window,
                    )
                payload = self._analysis_payload(
                    analysis,
                    request_number=request_number,
                    prompt_tokens=len(token_ids),
                    tokenize_seconds=tokenize_seconds,
                )
                return payload, len(token_ids)
        except BaseException:
            async with self._condition:
                self._failed_requests += 1
                if self.metrics is not None:
                    self.metrics.record_error(
                        request_seconds=time.perf_counter() - request_started_at
                    )
            raise
        finally:
            async with self._condition:
                self._pending_requests -= 1
                if self.metrics is not None:
                    self.metrics.request_finished()
                self._condition.notify_all()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(
            self._executor.shutdown, wait=True, cancel_futures=True
        )

    def _analysis_payload(
        self,
        analysis,
        *,
        request_number: int,
        prompt_tokens: int,
        tokenize_seconds: float,
    ) -> dict[str, Any]:
        page_accesses = analysis.page_accesses
        configured = []
        for index, (label, capacity_bytes) in enumerate(self.capacities):
            page_hits = analysis.capacity_page_hits[index]
            prefix_hits = analysis.capacity_prefix_hits[index]
            configured.append(
                {
                    "capacity": label,
                    "capacity_bytes": capacity_bytes,
                    "page_hits": page_hits,
                    "page_hit_rate": (
                        page_hits / page_accesses if page_accesses else 0.0
                    ),
                    "prefix_hits": prefix_hits,
                    "prefix_hit_rate": (
                        prefix_hits / page_accesses if page_accesses else 0.0
                    ),
                }
            )
        return {
            "request_number": request_number,
            "prompt_tokens": prompt_tokens,
            "page_accesses": page_accesses,
            "reusable_accesses": analysis.reusable_accesses,
            "infinite_page_hits": analysis.infinite_page_hits,
            "infinite_prefix_hits": analysis.infinite_prefix_hits,
            "configured_capacities": configured,
            "required_capacity_bytes": analysis.required_capacity_bytes,
            "tokenize_seconds": tokenize_seconds,
        }

    def _record_analysis(self, analysis, *, prompt_tokens: int) -> None:
        self._successful_requests += 1
        self._prompt_tokens_total += prompt_tokens
        self._page_accesses_total += analysis.page_accesses
        self._reusable_page_hits_total += analysis.reusable_accesses
        self._infinite_prefix_hits_total += analysis.infinite_prefix_hits
        for index in range(len(self.capacities)):
            self._capacity_page_hits[index] += analysis.capacity_page_hits[index]
            self._capacity_prefix_hits[index] += analysis.capacity_prefix_hits[index]
        self._last_required_capacity_bytes = analysis.required_capacity_bytes
        self._required_capacity_slide_window.add(analysis.required_capacity_bytes)
        if analysis.required_capacity_bytes is None:
            self._required_capacity_unreachable += 1
        else:
            self._required_capacity_samples += 1
            self._required_capacity_sum += analysis.required_capacity_bytes

    def _online_metrics_payload(self) -> dict[str, Any]:
        denominator = self._page_accesses_total
        configured = []
        for index, (label, capacity_bytes) in enumerate(self.capacities):
            page_hits = self._capacity_page_hits[index]
            prefix_hits = self._capacity_prefix_hits[index]
            configured.append(
                {
                    "capacity": label,
                    "capacity_bytes": capacity_bytes,
                    "page_hits": page_hits,
                    "page_hit_rate": page_hits / denominator if denominator else 0.0,
                    "prefix_hits": prefix_hits,
                    "prefix_hit_rate": (
                        prefix_hits / denominator if denominator else 0.0
                    ),
                }
            )
        return {
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "prompt_tokens": self._prompt_tokens_total,
            "page_accesses": self._page_accesses_total,
            "reusable_page_hits": self._reusable_page_hits_total,
            "infinite_page_hit_rate": (
                self._reusable_page_hits_total / denominator
                if denominator
                else 0.0
            ),
            "infinite_prefix_hits": self._infinite_prefix_hits_total,
            "infinite_prefix_hit_rate": (
                self._infinite_prefix_hits_total / denominator
                if denominator
                else 0.0
            ),
            "configured_capacities": configured,
            "required_capacity": {
                "last_bytes": self._last_required_capacity_bytes,
                "reachable_samples": self._required_capacity_samples,
                "unreachable_samples": self._required_capacity_unreachable,
                "mean_bytes": (
                    self._required_capacity_sum / self._required_capacity_samples
                    if self._required_capacity_samples
                    else None
                ),
                "slide_window": self._required_capacity_slide_window.snapshot(),
            },
        }

    async def status(self) -> dict[str, Any]:
        async with self._condition:
            return {
                "mode": "kv_capacity_estimator",
                "model": self.served_model_name,
                "requests": self.simulator.request_count,
                "pending_requests": self._pending_requests,
                "tokenizer_workers": self.tokenizer_worker_num,
                "tokenize_worker_seconds": self._tokenize_worker_seconds,
                "prometheus_metrics_enabled": self.metrics is not None,
                "finalized": self._final_result is not None,
                "config": dataclasses.asdict(self.simulation_config),
                "metrics": self._online_metrics_payload(),
            }

    async def finalize(self) -> dict[str, Any]:
        async with self._condition:
            self._accepting_requests = False
            while self._pending_requests:
                await self._condition.wait()
            if self._final_result is None:
                result = self.simulator.finish()
                self._final_result = dataclasses.asdict(result)
                self._final_result.update(
                    {
                        "tokenizer_workers": self.tokenizer_worker_num,
                        "tokenize_worker_seconds": self._tokenize_worker_seconds,
                        "online_metrics": self._online_metrics_payload(),
                    }
                )
                self._write_output(self._final_result)
            return self._final_result

    def _write_output(self, result: dict[str, Any]) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote KV capacity estimation result to %s", self.output_path)


def _chat_completion_payload(
    request: ChatCompletionRequest,
    analysis: dict[str, Any],
    prompt_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-kvcache-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        },
        "kv_capacity_estimation": analysis,
    }


async def _stream_chat_completion(
    request: ChatCompletionRequest,
    analysis: dict[str, Any],
    prompt_tokens: int,
) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-kvcache-{uuid.uuid4().hex}"
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "kv_capacity_estimation": analysis,
    }
    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
    if request.stream_options and request.stream_options.include_usage:
        usage = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }
        yield b"data: " + orjson.dumps(usage) + b"\n\n"
    yield b"data: [DONE]\n\n"


def create_kv_capacity_estimator_app(server_args: ServerArgs) -> FastAPI:
    """Build the tokenizer-only app. ``server_args`` must already be resolved."""
    publish(server_args, role="tokenizer")
    service = KvCapacityEstimatorService(server_args)
    cfg = resolving_view(server_args)

    async def validate_api_key(raw_request: Request) -> None:
        if cfg.api_key is None:
            return
        authorization = raw_request.headers.get("authorization", "")
        expected = f"Bearer {cfg.api_key}"
        if not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await service.start()
            logger.info(
                "KV capacity estimator mode is ready with %d tokenizer worker(s); "
                "model weights and SRT scheduler/detokenizer processes were not "
                "loaded",
                service.tokenizer_worker_num,
            )
            yield
        finally:
            try:
                result = await service.finalize()
                logger.info(
                    "KV capacity estimation finalized: requests=%d, page_accesses=%d",
                    result["total_requests"],
                    result["page_accesses"],
                )
            finally:
                await service.close()

    app = FastAPI(lifespan=lifespan)
    app.state.kv_capacity_estimator_service = service

    @app.exception_handler(HTTPException)
    async def openai_http_error(_: Request, error: HTTPException):
        return ORJSONResponse(
            {
                "error": {
                    "message": str(error.detail),
                    "type": "BadRequestError"
                    if error.status_code < 500
                    else "InternalServerError",
                    "param": None,
                    "code": error.status_code,
                }
            },
            status_code=error.status_code,
        )

    @app.get("/health")
    @app.get("/health_generate")
    async def health():
        return ORJSONResponse({"status": "ok", "mode": "kv_capacity_estimator"})

    if service.metrics is not None:

        @app.get("/metrics")
        async def prometheus_metrics():
            return Response(
                content=service.metrics.render(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @app.get("/v1/models", dependencies=[Depends(validate_api_key)])
    async def models():
        return ORJSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": service.served_model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "sglang",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(validate_api_key)])
    async def chat_completions(
        request: ChatCompletionRequest, raw_request: Request
    ):
        del raw_request
        try:
            analysis, prompt_tokens = await service.analyze(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(request, analysis, prompt_tokens),
                media_type="text/event-stream",
            )
        return ORJSONResponse(
            _chat_completion_payload(request, analysis, prompt_tokens)
        )

    @app.get(
        "/kv-capacity-estimator/status", dependencies=[Depends(validate_api_key)]
    )
    async def simulator_status():
        return ORJSONResponse(await service.status())

    @app.post(
        "/kv-capacity-estimator/finalize", dependencies=[Depends(validate_api_key)]
    )
    async def simulator_finalize():
        return ORJSONResponse(await service.finalize())

    return app


def launch_kv_capacity_estimator_server(server_args: ServerArgs) -> None:
    """Launch the lightweight tokenizer/simulator HTTP server."""
    app = create_kv_capacity_estimator_app(server_args)
    cfg = resolving_view(server_args)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        root_path=cfg.fastapi_root_path,
        log_level=cfg.log_level_http or cfg.log_level,
        ssl_keyfile=cfg.ssl_keyfile,
        ssl_certfile=cfg.ssl_certfile,
        ssl_ca_certs=cfg.ssl_ca_certs,
        ssl_keyfile_password=cfg.ssl_keyfile_password,
    )
