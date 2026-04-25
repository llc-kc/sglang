from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class DeepSeekV4CompressedComponent(TreeComponent):
    """DeepSeek V4 compressed/cache-state sidecar transfer component.

    The tree stores only FULL and SWA indices. This component has no node-local
    value; it emits HiCache transfers for V4 pools whose indices are derived
    from FULL pages or from the SWA suffix window.
    """

    component_type = ComponentType.DSV4_COMPRESSED

    _FULL_DERIVED_POOLS = (
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
    )
    _SWA_DERIVED_STATE_POOLS = (
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_INDEXER_STATE,
        PoolName.DEEPSEEK_V4_C128_STATE,
    )

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        self.sliding_window_size = getattr(params, "sliding_window_size", None) or float(
            "inf"
        )

    def _has_pool(self, name: PoolName) -> bool:
        controller = self.cache.cache_controller
        if controller is None:
            return False
        return name in controller.mem_pool_host.entry_map

    def _available_full_derived_transfers(self) -> list[PoolTransfer]:
        return [
            PoolTransfer(name=name)
            for name in self._FULL_DERIVED_POOLS
            if self._has_pool(name)
        ]

    def _available_state_transfers(self) -> list[PoolTransfer]:
        return [
            PoolTransfer(name=name, device_indices_source=PoolName.SWA)
            for name in self._SWA_DERIVED_STATE_POOLS
            if self._has_pool(name)
        ]

    def _collect_swa_host_suffix(
        self, node: UnifiedTreeNode
    ) -> Optional[torch.Tensor]:
        collected_leaf_first: list[torch.Tensor] = []
        n_swa = 0
        cur = node
        while cur is not None and cur.evicted:
            cd = cur.component_data[ComponentType.SWA]
            if cd.host_value is None:
                break
            collected_leaf_first.append(cd.host_value)
            n_swa += len(cd.host_value)
            if n_swa >= self.sliding_window_size:
                break
            cur = cur.parent
        if not collected_leaf_first:
            return None
        collected_leaf_first.reverse()
        return torch.cat(collected_leaf_first)

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        return None

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        return 0, 0

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        return None

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        return None

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        transfers = self._available_full_derived_transfers()

        if phase == CacheTransferPhase.BACKUP_HOST:
            if node.component_data[ComponentType.SWA].value is not None:
                transfers.extend(self._available_state_transfers())
            return transfers or None

        if phase == CacheTransferPhase.LOAD_BACK:
            swa_host_indices = self._collect_swa_host_suffix(node)
            if swa_host_indices is not None:
                for transfer in self._available_state_transfers():
                    transfer.host_indices = swa_host_indices
                    transfers.append(transfer)
            return transfers or None

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        return None
