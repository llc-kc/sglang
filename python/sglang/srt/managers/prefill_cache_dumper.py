"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Persist per-request cache snapshots at the prefill/decode boundary.

This is an opt-in debugging/export facility.  It intentionally performs a
synchronous device-to-host copy and file write so that a consumer never sees a
decode-mutated cache in a snapshot advertised as post-prefill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_SNAPSHOT_SCHEMA_VERSION = 1


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Write a torch artifact atomically; an existing snapshot is replaced."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_json_save(value: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fout:
            json.dump(value, fout, ensure_ascii=False, indent=2, default=str)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


class PrefillCacheDumper:
    """Save target/draft cache data for requests that have finished prefill."""

    def __init__(
        self,
        output_dir: Optional[str],
        *,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        gpu_id: int,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        draft_worker: Any = None,
    ) -> None:
        self.enabled = output_dir is not None
        self.output_dir = (
            Path(output_dir).expanduser().resolve() if output_dir is not None else None
        )
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker = draft_worker

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Post-prefill cache snapshots enabled: path=%s, tp_rank=%d, "
                "pp_rank=%d, dp_rank=%s, gpu_id=%d",
                self.output_dir,
                self.tp_rank,
                self.pp_rank,
                self.dp_rank,
                self.gpu_id,
            )

    def _file_prefix(self, rid: Any) -> str:
        rid_text = str(rid)
        safe_rid = "".join(
            char if char.isalnum() or char in "._-" else "_" for char in rid_text
        )
        safe_rid = safe_rid.strip("._")[:96] or "request"
        # Keep a digest even for apparently safe ids: truncation and sanitization
        # must never make two request ids overwrite each other.
        rid_digest = hashlib.sha256(rid_text.encode("utf-8")).hexdigest()[:12]
        rank_suffix = (
            f"gpu_{self.gpu_id}__tp_rank_{self.tp_rank}__pp_rank_{self.pp_rank}"
        )
        if self.dp_rank is not None:
            rank_suffix += f"__dp_rank_{self.dp_rank}"
        return f"req_id_{safe_rid}_{rid_digest}__{rank_suffix}"

    @staticmethod
    def _request_cache_len(req: Any) -> int:
        # Prefill samples the first output token but has not computed its KV yet.
        return max(0, int(req.seqlen) - 1)

    def _target_cache_copy(
        self, req: Any, token_indices: torch.Tensor
    ) -> tuple[Any, Any]:
        mamba_index = getattr(req, "mamba_pool_idx", None)
        cache_copy = self.token_to_kv_pool_allocator.get_cpu_copy(
            token_indices, mamba_indices=mamba_index
        )

        target_pool = self.token_to_kv_pool_allocator.get_kvcache()
        if hasattr(target_pool, "mamba_pool"):
            # HybridLinearKVPool.get_cpu_copy returns (dense_kv, mamba_state).
            dense_kv, mamba_state = cache_copy
            return dense_kv, mamba_state
        return cache_copy, None

    def _draft_cache_copy(self, req: Any, cache_len: int) -> list[dict[str, Any]]:
        if self.draft_worker is None:
            return []

        get_runners = getattr(self.draft_worker, "_draft_model_runners", None)
        if get_runners is None:
            return []

        snapshots = []
        for runner_index, runner in enumerate(get_runners()):
            model_path = getattr(
                getattr(runner, "model_config", None), "model_path", None
            )
            try:
                draft_pool = getattr(runner, "token_to_kv_pool", None)
                draft_req_pool = getattr(runner, "req_to_token_pool", None)
                if draft_pool is None or draft_req_pool is None:
                    continue

                row = draft_req_pool.req_to_token[req.req_pool_idx, :cache_len]
                # Some compact draft layouts populate only a subset of the target
                # positions. Slot zero is the reserved/dummy slot in request pools.
                valid_positions = torch.nonzero(row > 0, as_tuple=False).flatten()
                draft_indices = row.index_select(0, valid_positions)
                draft_cache = draft_pool.get_cpu_copy(draft_indices)
                snapshots.append(
                    {
                        "runner_index": runner_index,
                        "model_path": model_path,
                        "token_positions": valid_positions.cpu(),
                        "token_pool_indices": draft_indices.cpu(),
                        "cache": draft_cache,
                    }
                )
            except Exception as exc:
                # Preserve target/Mamba snapshots even when an uncommon draft
                # pool does not implement CPU export.
                logger.exception(
                    "Failed to copy draft cache runner %d for request %s",
                    runner_index,
                    req.rid,
                )
                snapshots.append(
                    {
                        "runner_index": runner_index,
                        "model_path": model_path,
                        "error": str(exc),
                    }
                )
        return snapshots

    def should_dump(self, req: Any, decoding_reqs: Optional[list[Any]] = None) -> bool:
        if not self.enabled or getattr(req, "_prefill_cache_dumped", False):
            return False
        if req.req_pool_idx is None or getattr(req, "is_retracted", False):
            return False
        if getattr(req, "inflight_middle_chunks", 0) > 0 or req.finished():
            return False
        if decoding_reqs and any(req is decode_req for decode_req in decoding_reqs):
            return False
        return True

    def dump_request(self, req: Any, decoding_reqs: Optional[list[Any]] = None) -> bool:
        """Synchronously save one request. Returns whether a snapshot was written."""
        if not self.should_dump(req, decoding_reqs):
            return False
        assert self.output_dir is not None

        cache_len = self._request_cache_len(req)
        token_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :cache_len
        ]
        prefix = self._file_prefix(req.rid)
        dense_name = f"{prefix}__dense_kv.pt"
        mamba_name = f"{prefix}__mamba_state.pt"
        draft_name = f"{prefix}__draft_kv.pt"
        metadata_name = f"{prefix}__metadata.json"
        metadata_path = self.output_dir / metadata_name

        try:
            # Invalidate an older snapshot with the same request/rank key before
            # replacing its component files. Metadata is the completion marker.
            metadata_path.unlink(missing_ok=True)
            dense_kv, mamba_state = self._target_cache_copy(req, token_indices)
            draft_kv = self._draft_cache_copy(req, cache_len)

            common = {
                "schema_version": _SNAPSHOT_SCHEMA_VERSION,
                "request_id": req.rid,
                "gpu_id": self.gpu_id,
                "tp_rank": self.tp_rank,
                "pp_rank": self.pp_rank,
                "dp_rank": self.dp_rank,
                "cache_len": cache_len,
            }
            _atomic_torch_save(
                {
                    **common,
                    "token_pool_indices": token_indices.cpu(),
                    "cache": dense_kv,
                },
                self.output_dir / dense_name,
            )
            _atomic_torch_save(
                {
                    **common,
                    "available": mamba_state is not None,
                    "mamba_pool_index": (
                        getattr(req, "mamba_pool_idx", None).cpu()
                        if isinstance(
                            getattr(req, "mamba_pool_idx", None), torch.Tensor
                        )
                        else getattr(req, "mamba_pool_idx", None)
                    ),
                    "state": mamba_state,
                },
                self.output_dir / mamba_name,
            )
            _atomic_torch_save(
                {
                    **common,
                    "available": any("cache" in runner for runner in draft_kv),
                    "runners": draft_kv,
                },
                self.output_dir / draft_name,
            )
            # Metadata is committed last and therefore acts as the completion
            # marker for readers scanning a directory while serving is active.
            _atomic_json_save(
                {
                    **common,
                    "files": {
                        "dense_kv": dense_name,
                        "mamba_state": mamba_name,
                        "draft_kv": draft_name,
                    },
                },
                metadata_path,
            )
            req._prefill_cache_dumped = True
            logger.info(
                "Saved post-prefill cache snapshot for request %s at TP rank %d "
                "(%d tokens): %s",
                req.rid,
                self.tp_rank,
                cache_len,
                self.output_dir / metadata_name,
            )
            return True
        except Exception:
            # Snapshotting is an observability/export feature; a storage failure
            # must not terminate an otherwise valid generation request.
            logger.exception(
                "Failed to save post-prefill cache snapshot for request %s at "
                "TP rank %d",
                req.rid,
                self.tp_rank,
            )
            return False

    def dump_batch(self, batch: Any, reqs: Optional[list[Any]] = None) -> int:
        decoding_reqs = getattr(batch, "decoding_reqs", None)
        return sum(
            self.dump_request(req, decoding_reqs=decoding_reqs)
            for req in (batch.reqs if reqs is None else reqs)
        )
