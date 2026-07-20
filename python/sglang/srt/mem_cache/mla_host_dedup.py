"""Host-memory dedup for MLA/DSA HiCache across attention-TP ranks.

MLA KV is identical on every attn-TP rank, so only the src rank (attn-TP
rank 0) keeps a real host pool; the other ranks run allocator-only "dummy"
pools and receive loaded pages via an NCCL broadcast on the load stream.

Single source of truth for the dedup gating and the broadcast machinery —
every dedup decision elsewhere must derive from these helpers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch

from sglang.srt.distributed import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# Backends that tolerate a logical target anchor with no host KV buffer.
# Mooncake keeps real per-rank draft sidecars on non-owner ranks. Other RDMA
# backends pin or register the target buffer and therefore remain excluded.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file", "mooncake"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def mla_dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    if is_dp_attention_enabled():
        return (
            get_attn_tensor_model_parallel_rank(),
            get_attn_tensor_model_parallel_world_size(),
        )
    return (
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )


def mla_host_dedup_eligible(
    kv_cache, storage_backend: Optional[str], enabled: bool = True
) -> bool:
    """Rank-independent gate. CUDA only; FP4 excluded (its per-rank scale
    buffer is not covered by the broadcast)."""
    return (
        enabled
        and isinstance(kv_cache, MLATokenToKVPool)
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


def is_mla_dedup_dummy_rank(
    kv_cache, storage_backend: Optional[str], enabled: bool = True
) -> bool:
    """Whether this rank must construct an allocator-only (dummy) host pool."""
    if not mla_host_dedup_eligible(kv_cache, storage_backend, enabled):
        return False
    rank, size = mla_dedup_rank_and_size()
    return size > 1 and rank != 0


class MLAHostDedupBroadcaster:
    """Replicates host-loaded MLA KV (and DSA indexer) device pages from the
    src rank to its attention-TP peers.

    Same-node groups use CUDA IPC direct-push; unsupported configurations fall
    back to a dedicated NCCL group with a reused staging buffer.
    """

    # Tokens (or DSA indexer pages) staged per broadcast chunk.
    CHUNK_TOKENS = 512

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        group: torch.distributed.ProcessGroup,
        cpu_group: torch.distributed.ProcessGroup,
        group_ranks: Sequence[int],
        src_global_rank: int,
    ):
        self.device_pool = device_pool
        self.group = group
        self.cpu_group = cpu_group
        self.group_ranks = list(group_ranks)
        self.src_global_rank = src_global_rank
        self.is_src = mla_dedup_rank_and_size()[0] == 0
        self.rank = torch.distributed.get_rank(group=cpu_group)
        self.world_size = torch.distributed.get_world_size(group=cpu_group)
        self.layer_num = device_pool.layer_num
        self.device = device_pool.device
        self.kv_staging = torch.empty(
            self.layer_num * self.CHUNK_TOKENS * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype,
            device=self.device,
        )
        # DSA keeps a per-page indexer buffer that must be broadcast too.
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.CHUNK_TOKENS * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )

        # Direct-push sends only target indices and a completion token through
        # NCCL. The KV/indexer payload is written from group rank 0 straight to
        # peer cache buffers through CUDA IPC mappings.
        self.direct_push_enabled = False
        self.kv_direct_push = None
        self.idx_direct_push = None
        self.direct_indices = torch.empty(
            self.world_size * self.CHUNK_TOKENS,
            dtype=torch.int64,
            device=self.device,
        )
        self.direct_done = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self._try_init_direct_push()

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> MLAHostDedupBroadcaster:
        """Build the NCCL group (a world collective — all dedup participants
        must call in lockstep) and the staging buffers."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        return cls(
            device_pool,
            group,
            cpu_group=base_group,
            group_ranks=group_ranks,
            src_global_rank=group_ranks[0],
        )

    @staticmethod
    def _buffer_signature(buffers: Sequence[torch.Tensor]) -> tuple:
        return tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
            for tensor in buffers
        )

    def _gather_objects(self, value: Any) -> List[Any]:
        """Gloo-compatible object all-gather that also works in inference mode."""
        gathered = []
        for group_rank, global_rank in enumerate(self.group_ranks):
            holder = [value if self.rank == group_rank else None]
            torch.distributed.broadcast_object_list(
                holder, src=global_rank, group=self.cpu_group
            )
            gathered.append(holder[0])
        return gathered

    def _broadcast_src_object(self, value: Any) -> Any:
        holder = [value if self.is_src else None]
        torch.distributed.broadcast_object_list(
            holder, src=self.src_global_rank, group=self.cpu_group
        )
        return holder[0]

    def _close_direct_push(self) -> None:
        for state_name in ("kv_direct_push", "idx_direct_push"):
            state = getattr(self, state_name, None)
            if state is not None:
                try:
                    state.close()
                except Exception:
                    pass
                setattr(self, state_name, None)
        self.direct_push_enabled = False

    def _try_init_direct_push(self) -> None:
        """Collectively initialize same-node CUDA IPC mappings.

        Any rank-local setup failure is shared over the CPU group so every rank
        selects the NCCL fallback consistently.
        """
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(self.cpu_group, source_rank=0)):
            if self.is_src:
                logger.info("MLA CUDA IPC direct-push disabled: TP group spans nodes")
            return

        local_states = []
        local_payload = None
        try:
            from sglang.jit_kernel.cuda_ipc_direct_push import CudaIpcDirectPush

            kv_state = CudaIpcDirectPush(
                self.rank, self.world_size, self.device_pool.kv_buffer
            )
            local_states.append(kv_state)
            idx_state = None
            if self.idx_bufs is not None:
                idx_state = CudaIpcDirectPush(self.rank, self.world_size, self.idx_bufs)
                local_states.append(idx_state)
            local_payload = {
                "ok": True,
                "kv": kv_state.share_buffers(),
                "kv_signature": self._buffer_signature(self.device_pool.kv_buffer),
                "idx": None if idx_state is None else idx_state.share_buffers(),
                "idx_signature": (
                    None
                    if self.idx_bufs is None
                    else self._buffer_signature(self.idx_bufs)
                ),
            }
        except Exception as exc:
            local_payload = {"ok": False, "error": repr(exc)}

        peer_payloads = self._gather_objects(local_payload)
        signatures_match = all(
            payload.get("ok")
            and payload.get("kv_signature") == peer_payloads[0].get("kv_signature")
            and payload.get("idx_signature") == peer_payloads[0].get("idx_signature")
            for payload in peer_payloads
        )
        if not signatures_match:
            errors = [
                payload.get("error", "buffer signature mismatch")
                for payload in peer_payloads
                if not payload.get("ok")
                or payload.get("kv_signature") != peer_payloads[0].get("kv_signature")
                or payload.get("idx_signature") != peer_payloads[0].get("idx_signature")
            ]
            for state in local_states:
                state.close()
            if self.is_src:
                logger.warning(
                    "MLA CUDA IPC direct-push unavailable; using NCCL fallback: %s",
                    "; ".join(errors),
                )
            return

        open_status = None
        if self.is_src:
            try:
                kv_state.open_peer_buffers([payload["kv"] for payload in peer_payloads])
                if idx_state is not None:
                    idx_state.open_peer_buffers(
                        [payload["idx"] for payload in peer_payloads]
                    )
                open_status = {"ok": True}
            except Exception as exc:
                open_status = {"ok": False, "error": repr(exc)}
        open_status = self._broadcast_src_object(open_status)
        if not open_status["ok"]:
            for state in local_states:
                state.close()
            if self.is_src:
                logger.warning(
                    "MLA CUDA IPC peer mapping failed; using NCCL fallback: %s",
                    open_status["error"],
                )
            return

        if self.is_src:
            self.kv_direct_push = kv_state
            self.idx_direct_push = idx_state
        else:
            # Exported handles remain valid for the lifetime of their cache
            # tensors; non-source ranks do not need to retain a control object.
            for state in local_states:
                state.close()
        self.direct_push_enabled = True
        if self.is_src:
            logger.info(
                "MLA CUDA IPC direct-push enabled for %d local TP ranks",
                self.world_size,
            )

    def broadcast_loaded(self, device_indices: torch.Tensor, load_stream) -> None:
        """Broadcast loaded pages from the src rank. Must run with
        ``load_stream`` as the current stream."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)
        self._bcast_rows(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
            self.kv_direct_push,
        )
        # to do: support indexer sharing across layers
        if self.idx_bufs is not None:
            page_size = self.device_pool.page_size
            page_idx = (
                torch.unique(torch.div(indices, page_size, rounding_mode="floor"))
                if page_size > 1
                else indices
            )
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
            self._bcast_rows(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                self.idx_direct_push,
            )

    def _bcast_rows(self, buf_list, staging, target, elem, direct_push) -> None:
        """Chunked broadcast of one per-layer buffer set; ``target`` indexes
        dim 0 (token indices for KV, page indices for the DSA indexer)."""
        if self.direct_push_enabled:
            self._direct_push_rows(target, direct_push)
            return

        n = target.shape[0]
        for start in range(0, n, self.CHUNK_TOKENS):
            cur = min(self.CHUNK_TOKENS, n - start)
            idx = target[start : start + cur]
            chunk = staging[: self.layer_num * cur * elem]
            if self.is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    chunk[o : o + cur * elem].copy_(buf_list[layer_id][idx].reshape(-1))
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    dst = buf_list[layer_id]
                    dst[idx] = chunk[o : o + cur * elem].view(cur, *dst.shape[1:])

    def _direct_push_rows(self, target, direct_push) -> None:
        n = target.shape[0]
        if target.dtype != torch.int64:
            target = target.to(dtype=torch.int64)
        for start in range(0, n, self.CHUNK_TOKENS):
            cur = min(self.CHUNK_TOKENS, n - start)
            local_indices = target[start : start + cur].contiguous()
            indices_per_rank = self.direct_indices[: self.world_size * cur].view(
                self.world_size, cur
            )
            torch.distributed.all_gather_into_tensor(
                indices_per_rank.reshape(-1),
                local_indices,
                group=self.group,
            )
            if self.is_src:
                assert direct_push is not None
                direct_push.push(indices_per_rank)
            # A one-byte collective orders the source's remote writes before
            # peer load streams publish their per-layer completion events.
            torch.distributed.broadcast(
                self.direct_done,
                src=self.src_global_rank,
                group=self.group,
            )

    def destroy(self) -> None:
        self._close_direct_push()
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


def maybe_build_mla_broadcaster(
    device_pool,
    tp_group: torch.distributed.ProcessGroup,
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = True,
) -> Optional[MLAHostDedupBroadcaster]:
    """None when dedup does not engage (gate fails or single attn-TP rank)."""
    if not mla_host_dedup_eligible(device_pool, storage_backend, enabled):
        return None
    if mla_dedup_rank_and_size()[1] <= 1:
        return None
    return MLAHostDedupBroadcaster.build(device_pool, tp_group, attn_tp_group)


@dataclass
class MLAHostDedupPrebuild:
    """Groups/buffers rendezvoused ahead of the slow host KV allocation."""

    broadcaster: MLAHostDedupBroadcaster
    # None without a storage backend, so a later runtime attach still builds
    # its gloo groups inline.
    prefetch_sync_groups: Optional[List[torch.distributed.ProcessGroup]]


def maybe_prebuild_mla_host_dedup(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = True,
) -> Optional[MLAHostDedupPrebuild]:
    """Issue the controller's init-time world collectives BEFORE the host KV
    pool is allocated.

    The src rank can spend many minutes pinning host KV while the dummy
    ranks race ahead into create_custom_parallel_group (NCCL bcast group +
    gloo prefetch groups) and trip the 600s NCCL watchdog; prebuilding
    completes the rendezvouses in lockstep first. Returns None when dedup
    does not engage — same gating as the controller, so groups are never
    built on ranks that would ignore them.
    """
    broadcaster = maybe_build_mla_broadcaster(
        kv_cache, tp_group, attn_tp_group, storage_backend, enabled
    )
    if broadcaster is None:
        return None
    prefetch_sync_groups = None
    if storage_backend is not None:
        prefetch_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return MLAHostDedupPrebuild(broadcaster, prefetch_sync_groups)


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Same construction as HiCacheController._create_prefetch_sync_groups."""
    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
        )
    return groups
