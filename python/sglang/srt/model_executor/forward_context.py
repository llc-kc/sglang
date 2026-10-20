"""Per-forward-call control context.

Owns ForwardContext — a frozen dataclass holding control configs the model
layer reads at depth via get_forward_context(). The only mandatory field
today is attn_backend; pool refs are derived from attn_backend.*
(every backend caches them at __init__), so a published ForwardContext
is enough to resolve the active pools without a separate global.

ModelRunner._forward_raw publishes a fresh ForwardContext for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
dataclasses.replace and wrap the override scope with forward_context().

Distinct from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.TcPiecewiseForwardContext,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: _current is a plain module-level global, not thread-local.
This matches the global_server_args precedent and is safe because each
forward runs synchronously on a single Python thread per worker process. If
worker threads ever share a process, migrate to contextvars.ContextVar.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.cache_controller import ForwardLayerLoad
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via get_forward_context();
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use dataclasses.replace for per-call overrides."""

    attn_backend: AttentionBackend
    mla_forward_load: Optional[ForwardLayerLoad] = None


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the forward_context() context manager."""
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    return _current is not None


def get_forward_context() -> ForwardContext:
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


def bind_mla_forward_load(
    kv_pool, *, can_run_tbo: bool, split_prefill: bool, model
) -> Optional[ForwardLayerLoad]:
    counter = getattr(kv_pool, "layer_transfer_counter", None)
    bind = getattr(counter, "bind_forward_load", None)
    handle = bind() if bind is not None else None
    if handle is not None and (
        can_run_tbo
        or split_prefill
        or not getattr(model, "supports_mla_forward_load", False)
    ):
        raise RuntimeError(
            "Forward MLA load requires a supported model without TBO or split prefill"
        )
    return handle


def has_pending_mla_forward_load(kv_pool, consumer_index: int) -> bool:
    counter = getattr(kv_pool, "layer_transfer_counter", None)
    check = getattr(counter, "has_pending_forward_load", None)
    return check is not None and check(consumer_index)


def before_mla_forward_layer(layer_index: int) -> None:
    # Runtime handles belong to the invocation, not to ForwardBatch: eager and
    # speculative paths replace that dataclass when preparing their buffers.
    handle = _current.mla_forward_load if _current is not None else None
    if handle is not None:
        handle.before_layer(layer_index)


def wait_for_mla_state_copies(*, clear_indices, cow_src_indices) -> None:
    handle = _current.mla_forward_load if _current is not None else None
    if handle is not None and any(
        indices is not None and len(indices) > 0
        for indices in (clear_indices, cow_src_indices)
    ):
        # Mamba clear/COW touches state across layers before the model loop.
        # It must follow local H2D, but need not submit dedup collectives early.
        handle.wait_for_local_copies()


@contextmanager
def forward_context(ctx: ForwardContext):
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
