# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tokenizer-only HTTP server backed by kv-capacity-estimator.

This is deliberately a separate launch path from the SRT engine.  It loads the
model configuration, tokenizer, and chat template, but never constructs a
scheduler, detokenizer, model runner, or weight loader.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import multiprocessing
import os
import functools
import secrets
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
_PAGE_SIZE = None
_INCLUDE_PARTIAL_PAGE = False


def _load_simulator_api():
    try:
        from kv_capacity_estimator import (
            ReplaySimulator,
            simulation_config_from_dict,
        )
    except ImportError as error:
        raise RuntimeError(
            "--kv-capacity-estimator requires the kv-capacity-estimator "
            "wheel. Install it in the SGLang environment, for example: "
            "pip install kv_capacity_estimator-0.2.0-py3-none-any.whl"
        ) from error
    return ReplaySimulator, simulation_config_from_dict


def _build_online_simulation_config(raw_config, config_factory):
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("--kv-capacity-estimator-config must be a JSON object")
    simulator_values = dict(raw_config)
    output_path = simulator_values.pop("output_path", None)
    if "slide_window_size" in simulator_values:
        raise ValueError(
            "slide_window_size has been removed; select a time range with "
            "Prometheus increase()/rate() and histogram_quantile() instead"
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
        raise ValueError("kv-capacity-estimator config 'output_path' must be a string")
    try:
        simulation_config = config_factory(simulator_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid --kv-capacity-estimator-config: {error}") from error
    return simulation_config, output_path


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
    global _TOKENIZER_RUNTIME, _PAGE_SIZE, _INCLUDE_PARTIAL_PAGE
    server_args.resolve_once()
    publish(server_args, role="tokenizer")
    # These processes only run template rendering and CPU tokenization; they
    # never do torch compute. Limit the OMP/MKL pool to one thread so dozens
    # of spawned workers don't oversubscribe the CPUs.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    cfg = resolving_view(server_args)
    from kv_capacity_estimator.defaults import DEFAULT_PAGE_SIZE

    raw_sim_config = cfg.kv_capacity_estimator_config
    if isinstance(raw_sim_config, str):
        sim_config = json.loads(raw_sim_config) if raw_sim_config else {}
    elif isinstance(raw_sim_config, dict):
        sim_config = raw_sim_config
    else:
        sim_config = {}
    _PAGE_SIZE = int(sim_config.get("page_size", DEFAULT_PAGE_SIZE))
    _INCLUDE_PARTIAL_PAGE = bool(sim_config.get("include_partial_page", False))
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
        token_ids = manager.tokenizer.encode(processed.prompt, add_special_tokens=False)
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

    # Compute the chained page hashes inside the tokenizer process. This is
    # pure-function CPU work over the same token list, and doing it here
    # parallelizes it across the tokenizer pool instead of leaving it as a
    # single-threaded step in the main process (which also competes for the
    # GIL with the event loop). Returning ~N/64 page keys also shrinks the
    # IPC payload by an order of magnitude.
    from kv_capacity_estimator.pages import chained_page_hashes

    page_keys = chained_page_hashes(
        token_ids,
        page_size=_PAGE_SIZE,
        include_partial_page=_INCLUDE_PARTIAL_PAGE,
    )
    return page_keys, len(token_ids), time.perf_counter() - started_at


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

    # Include zero (cold requests need no cache to meet their reusable-prefix
    # target), and quarter-octave bounds up to 1 PiB. Configured capacities are
    # also exact bucket boundaries, including capacities beyond this range.
    _REQUIRED_CAPACITY_BUCKETS = (0.0,) + tuple(
        2 ** (exponent / 4) for exponent in range(80, 201)
    )

    def __init__(
        self,
        *,
        model_name: str,
        capacities: tuple[tuple[str, int], ...],
        extra_labels: dict[str, str] | None,
        page_size: int,
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
            "le",
        }
        conflicting_labels = reserved_labels.intersection(extra_labels or {})
        if conflicting_labels:
            raise ValueError(
                "--extra-metric-labels uses reserved labels: "
                + ", ".join(sorted(conflicting_labels))
            )
        self.page_size = page_size
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
        self.capacity_prefix_hit_tokens_total = Counter(
            f"{prefix}capacity_prefix_hit_tokens_total",
            "Number of reusable prefix tokens at each configured capacity.",
            capacity_label_names,
            registry=self.registry,
        )
        self.infinite_prefix_hit_tokens_total = Counter(
            f"{prefix}infinite_prefix_hit_tokens_total",
            "Number of reusable prefix tokens with unlimited capacity.",
            label_names,
            registry=self.registry,
        )
        self.required_capacity_bytes = Histogram(
            f"{prefix}required_capacity_bytes",
            "Per-request minimum KV-cache capacity in bytes required to "
            "reach target_hit_rate_ratio of the unlimited-cache reusable prefix; "
            "unreachable requests are excluded and zero-capacity requests included.",
            label_names,
            buckets=sorted(
                set(self._REQUIRED_CAPACITY_BUCKETS).union(
                    capacity_bytes for _, capacity_bytes in capacities
                )
            ),
            registry=self.registry,
        )
        self.required_capacity_unreachable_total = Counter(
            f"{prefix}required_capacity_unreachable_total",
            "Requests for which the target hit rate was not reachable.",
            label_names,
            registry=self.registry,
        )
        # Instantiate cumulative children before the first scrape so the first
        # observed requests, zero hits, and unreachable counts have a baseline.
        self.tokenizer_workers.labels(**self._labels).set(0)
        self.required_capacity_bytes.labels(**self._labels)
        self.required_capacity_unreachable_total.labels(**self._labels)
        self.infinite_prefix_hit_tokens_total.labels(**self._labels)
        for is_streaming in ("true", "false"):
            self.prompt_tokens_total.labels(**self._labels, is_streaming=is_streaming)
            self.num_requests_total.labels(**self._labels, is_streaming=is_streaming)
        for status in ("success", "error"):
            self.requests_total.labels(**self._labels, status=status)
        for counter in (
            self.page_accesses_total,
            self.reusable_page_hits_total,
            self.infinite_prefix_hits_total,
        ):
            counter.labels(**self._labels)
        for label, capacity_bytes in capacities:
            capacity_labels = self._capacity_labels(label, capacity_bytes)
            self.capacity_page_hits_total.labels(**capacity_labels)
            self.capacity_prefix_hits_total.labels(**capacity_labels)
            self.capacity_prefix_hit_tokens_total.labels(**capacity_labels)
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
        self.infinite_prefix_hit_tokens_total.labels(**self._labels).inc(
            min(analysis.infinite_prefix_hits * self.page_size, prompt_tokens)
        )
        for index, (label, capacity_bytes) in enumerate(capacities):
            capacity_labels = self._capacity_labels(label, capacity_bytes)
            self.capacity_page_hits_total.labels(**capacity_labels).inc(
                analysis.capacity_page_hits[index]
            )
            self.capacity_prefix_hits_total.labels(**capacity_labels).inc(
                analysis.capacity_prefix_hits[index]
            )
            self.capacity_prefix_hit_tokens_total.labels(**capacity_labels).inc(
                min(
                    analysis.capacity_prefix_hits[index] * self.page_size, prompt_tokens
                )
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

    def render(self) -> bytes:
        from prometheus_client import generate_latest

        return generate_latest(self.registry)


class KvCapacityEstimatorService:
    """Parallelize tokenization and maintain one central online LRU simulator."""

    def __init__(self, server_args: ServerArgs) -> None:
        ReplaySimulator, simulation_config_from_dict = _load_simulator_api()
        cfg = resolving_view(server_args)

        if cfg.tokenizer_worker_num < 1:
            raise ValueError("--tokenizer-worker-num must be at least 1")
        simulation_config, output_path = _build_online_simulation_config(
            cfg.kv_capacity_estimator_config,
            simulation_config_from_dict,
        )
        capacities = tuple(simulation_config.capacities)

        self._simulator_factory = ReplaySimulator
        self.simulator = ReplaySimulator(simulation_config)
        self.capacities = capacities
        self.simulation_config = simulation_config
        self.output_path = Path(output_path) if output_path else None
        self.served_model_name = cfg.served_model_name or cfg.model_path
        self.tokenizer_worker_num = cfg.tokenizer_worker_num
        self._extra_metric_labels = cfg.extra_metric_labels
        self.metrics = (
            KvCapacityEstimatorMetrics(
                model_name=self.served_model_name,
                capacities=capacities,
                extra_labels=cfg.extra_metric_labels,
                page_size=simulation_config.page_size,
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
        # The simulator is stateful (one global LRU history), so all its
        # updates must be serialized. Running them on a dedicated single
        # worker thread keeps the ~10-20 ms of CPU work per request off the
        # asyncio event loop, where it previously blocked request
        # acceptance, IPC draining, and response streaming for every
        # in-flight request.
        self._simulator_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kv-capacity-simulator",
        )
        self._simulator_lock = threading.Lock()
        self._condition = asyncio.Condition()
        self._lifecycle_lock = asyncio.Lock()
        self._accepting_requests = True
        self._pending_requests = 0
        self._clear_statistics()
        self._closed = False

    def _clear_statistics(self) -> None:
        self._tokenize_worker_seconds = 0.0
        self._successful_requests = 0
        self._failed_requests = 0
        self._prompt_tokens_total = 0
        self._page_accesses_total = 0
        self._reusable_page_hits_total = 0
        self._infinite_prefix_hits_total = 0
        self._capacity_page_hits = [0] * len(self.capacities)
        self._capacity_prefix_hits = [0] * len(self.capacities)
        self._required_capacity_samples = 0
        self._required_capacity_sum = 0
        self._required_capacity_unreachable = 0
        self._last_required_capacity_bytes = None
        self._final_result = None

    async def start(self) -> None:
        """Start every tokenizer process and surface initialization errors."""
        loop = asyncio.get_running_loop()
        # Spawned processes each re-initialize the tokenizer and chat
        # template, which takes tens of seconds. A single burst of pings can
        # all be consumed by the first ready process, so keep pinging until
        # every worker PID has reported in; otherwise the server advertises
        # readiness while most workers are still initializing.
        started_at = time.perf_counter()
        deadline = started_at + 900
        worker_pids: set[int] = set()
        while len(worker_pids) < self.tokenizer_worker_num:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"Only {len(worker_pids)}/{self.tokenizer_worker_num} "
                    "tokenizer workers became ready within 900s"
                )
            futures = [
                loop.run_in_executor(self._executor, _tokenizer_worker_ping)
                for _ in range(self.tokenizer_worker_num)
            ]
            worker_pids.update(await asyncio.gather(*futures))
            if len(worker_pids) < self.tokenizer_worker_num:
                logger.info(
                    "Waiting for tokenizer workers: %d/%d ready after %.0fs",
                    len(worker_pids),
                    self.tokenizer_worker_num,
                    time.perf_counter() - started_at,
                )
                await asyncio.sleep(1.0)
        if self.metrics is not None:
            self.metrics.set_worker_count(self.tokenizer_worker_num)
        logger.info(
            "Initialized KV-cache tokenizer pool with %d process(es) in %.0fs; "
            "startup checks ran on PIDs %s",
            self.tokenizer_worker_num,
            time.perf_counter() - started_at,
            sorted(worker_pids),
        )

    def _run_simulation_and_record(
        self,
        page_keys: tuple[bytes, ...],
        token_count: int,
        *,
        is_streaming: bool,
        tokenize_seconds: float,
        tokenizer_roundtrip_seconds: float,
        request_started_at: float,
    ) -> tuple[dict[str, Any], int]:
        """Update the stateful simulator and record metrics.

        The tokenizer worker has already rendered, encoded, and built the
        chained page hashes. Only the shared LRU-history update must be
        serialized, so this runs on a dedicated single-worker thread and
        calls the simulator's public process_page_keys entry point with the
        prebuilt page keys, keeping the CPU-bound work off the asyncio event
        loop.
        """
        with self._simulator_lock:
            # Results intentionally enter the LRU trace in tokenizer
            # completion order. The single simulator thread preserves one
            # global, deterministic trace.
            self._tokenize_worker_seconds += tokenize_seconds
            simulation_started_at = time.perf_counter()
            # Update the shared LRU history from the prebuilt page keys.
            # Page hashing was already done in the parallel tokenizer
            # process, so only the serialized history update runs here.
            analysis = self.simulator.process_page_keys(page_keys)
            simulation_seconds = time.perf_counter() - simulation_started_at
            request_number = self.simulator.request_count
            self._record_analysis(analysis, prompt_tokens=token_count)
            if self.metrics is not None:
                self.metrics.record_success(
                    is_streaming=is_streaming,
                    prompt_tokens=token_count,
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
                )
            payload = self._analysis_payload(
                analysis,
                request_number=request_number,
                prompt_tokens=token_count,
                tokenize_seconds=tokenize_seconds,
            )
            return payload, token_count

    async def analyze(
        self, request: ChatCompletionRequest
    ) -> tuple[dict[str, Any], int]:
        """Tokenize in a worker and update the central simulator on completion."""
        request_started_at = time.perf_counter()
        async with self._condition:
            if not self._accepting_requests:
                raise HTTPException(
                    status_code=409,
                    detail="KV capacity estimation is finalizing, finalized, or resetting",
                )
            self._pending_requests += 1
            if self.metrics is not None:
                self.metrics.request_admitted()

        try:
            loop = asyncio.get_running_loop()
            tokenizer_submitted_at = time.perf_counter()
            page_keys, token_count, tokenize_seconds = await loop.run_in_executor(
                self._executor,
                _tokenize_chat_request,
                request.model_dump(mode="json"),
            )
            tokenizer_roundtrip_seconds = time.perf_counter() - tokenizer_submitted_at
            return await loop.run_in_executor(
                self._simulator_executor,
                functools.partial(
                    self._run_simulation_and_record,
                    page_keys,
                    token_count,
                    is_streaming=request.stream,
                    tokenize_seconds=tokenize_seconds,
                    tokenizer_roundtrip_seconds=tokenizer_roundtrip_seconds,
                    request_started_at=request_started_at,
                ),
            )
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
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        # Drain after the tokenizer pool so no simulation work can be
        # submitted concurrently.
        await asyncio.to_thread(self._simulator_executor.shutdown, wait=True)

    def _do_finalize(self) -> dict[str, Any]:
        """Finalize the simulator state. Runs on the dedicated simulator thread
        so ``finish()`` happens strictly after every in-flight ``process()``."""
        with self._simulator_lock:
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
                self._reusable_page_hits_total / denominator if denominator else 0.0
            ),
            "infinite_prefix_hits": self._infinite_prefix_hits_total,
            "infinite_prefix_hit_rate": (
                self._infinite_prefix_hits_total / denominator if denominator else 0.0
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
        # Keep lifecycle transitions running even if the HTTP client disconnects.
        return await asyncio.shield(self._finalize())

    async def _finalize(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            async with self._condition:
                self._accepting_requests = False
                while self._pending_requests:
                    await self._condition.wait()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._simulator_executor, self._do_finalize
            )

    def _do_reset(self) -> None:
        """Replace all trace state after previously submitted simulation work."""
        with self._simulator_lock:
            simulator = self._simulator_factory(self.simulation_config)
            metrics = None
            if self.metrics is not None:
                metrics = KvCapacityEstimatorMetrics(
                    model_name=self.served_model_name,
                    capacities=self.capacities,
                    extra_labels=self._extra_metric_labels,
                    page_size=self.simulation_config.page_size,
                )
                metrics.set_worker_count(self.tokenizer_worker_num)
            self.simulator = simulator
            self.metrics = metrics
            self._clear_statistics()

    async def reset(self) -> dict[str, Any]:
        return await asyncio.shield(self._reset())

    async def _reset(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            async with self._condition:
                was_accepting = self._accepting_requests
                self._accepting_requests = False
                while self._pending_requests:
                    await self._condition.wait()
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._simulator_executor, self._do_reset)
            except BaseException:
                async with self._condition:
                    self._accepting_requests = was_accepting
                raise
            async with self._condition:
                self._accepting_requests = True
            return await self.status()

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
                    "type": (
                        "BadRequestError"
                        if error.status_code < 500
                        else "InternalServerError"
                    ),
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
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
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

    @app.get("/kv-capacity-estimator/status", dependencies=[Depends(validate_api_key)])
    async def simulator_status():
        return ORJSONResponse(await service.status())

    @app.post(
        "/kv-capacity-estimator/finalize", dependencies=[Depends(validate_api_key)]
    )
    async def simulator_finalize():
        return ORJSONResponse(await service.finalize())

    @app.post("/kv-capacity-estimator/reset", dependencies=[Depends(validate_api_key)])
    async def simulator_reset():
        return ORJSONResponse(await service.reset())

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
