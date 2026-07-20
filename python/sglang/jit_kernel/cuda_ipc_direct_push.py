from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple, cast

import torch
import tvm_ffi

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

IpcInputPair = Tuple[int, List[int]]

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    class CudaIpcDirectPushObj:
        def __init__(self, rank: int, world_size: int) -> None: ...
        def share_tensor(self, tensor: torch.Tensor): ...
        def open_peer_tensors(
            self,
            peer_tensors: List[List[IpcInputPair]],
            local_ptrs: List[int],
        ) -> None: ...
        def free(self) -> None: ...


DEFAULT_BLOCK_QUOTA = 64
DEFAULT_BLOCK_SIZE = 256


@cache_once
def _jit_direct_push_base_module() -> Module:
    return load_jit(
        "cuda_ipc_direct_push_base",
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/cuda_ipc_direct_push_base.cuh"],
        cuda_wrappers=[("register_once", "register_cuda_ipc_direct_push")],
    )


@cache_once
def get_cuda_ipc_direct_push_cls() -> type[CudaIpcDirectPushObj]:
    module = _jit_direct_push_base_module()
    module.register_once()

    @tvm_ffi.register_object("sgl.CudaIpcDirectPush")
    class CudaIpcDirectPushObjReal(tvm_ffi.Object):
        __slots__ = ("__dict__",)

        def __init__(self, rank: int, world_size: int) -> None:
            self.__ffi_init__(rank, world_size)

    return cast(type["CudaIpcDirectPushObj"], CudaIpcDirectPushObjReal)


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4
    if element_size <= 1024:
        return 2
    return 1


@cache_once
def _jit_direct_push_module(
    *, element_size: int, unroll: int, block_quota: int
) -> Module:
    args = make_cpp_args(element_size, unroll, block_quota, DEFAULT_BLOCK_SIZE)
    return load_jit(
        "cuda_ipc_direct_push",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/cuda_ipc_direct_push.cuh"],
        cuda_wrappers=[("push", f"cuda_ipc_direct_push<{args}>")],
    )


class CudaIpcDirectPush:
    """Owns CUDA IPC mappings for one homogeneous per-layer buffer set."""

    def __init__(self, rank: int, world_size: int, buffers: Sequence[torch.Tensor]):
        if not buffers:
            raise ValueError("CUDA IPC direct-push requires at least one buffer")
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in buffers):
            raise ValueError(
                "CUDA IPC direct-push buffers must be contiguous CUDA tensors"
            )
        self.rank = rank
        self.world_size = world_size
        self.buffers = list(buffers)
        self.element_size = self.buffers[0][0].numel() * self.buffers[0].element_size()
        if any(
            tensor[0].numel() * tensor.element_size() != self.element_size
            for tensor in self.buffers
        ):
            raise ValueError(
                "CUDA IPC direct-push requires an identical row size on every layer"
            )
        if self.element_size % 128 != 0:
            raise ValueError(
                f"CUDA IPC direct-push row size must be a multiple of 128 bytes, got {self.element_size}"
            )
        self.obj = get_cuda_ipc_direct_push_cls()(rank, world_size)
        self._module = _jit_direct_push_module(
            element_size=self.element_size,
            unroll=_default_unroll(self.element_size),
            block_quota=DEFAULT_BLOCK_QUOTA,
        )
        self._closed = False

    def share_buffers(self) -> List[IpcInputPair]:
        result = []
        for tensor in self.buffers:
            offset, handle = self.obj.share_tensor(tensor)
            result.append((int(offset), [int(value) for value in handle]))
        return result

    def open_peer_buffers(self, peer_buffers: List[List[IpcInputPair]]) -> None:
        self.obj.open_peer_tensors(
            peer_buffers, [int(tensor.data_ptr()) for tensor in self.buffers]
        )

    @debug_kernel_api
    def push(self, indices_per_rank: torch.Tensor) -> None:
        self._module.push(
            self.obj,
            indices_per_rank,
            self.element_size,
            self.element_size,
        )

    def close(self) -> None:
        if not self._closed:
            self.obj.free()
            self._closed = True
