"""Multi-GPU correctness test for MLA CUDA IPC direct-push."""

from __future__ import annotations

import atexit
import os

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.cuda_ipc_direct_push import CudaIpcDirectPush
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-unit",
    runner_config="8-gpu-h200",
)


def _init_groups() -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    assert coord.cpu_group is not None
    assert coord.device_group is not None
    return coord.cpu_group, coord.device_group


def test_cuda_ipc_direct_push() -> None:
    cpu_group, nccl_group = _init_groups()
    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    num_layers = 3
    num_rows = 64
    row_elements = 64  # 128 bytes in bf16
    num_indices = 7
    buffers = [
        torch.full(
            (num_rows, row_elements),
            -1,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(num_layers)
    ]
    state = CudaIpcDirectPush(rank, world_size, buffers)

    local_metadata = state.share_buffers()
    peer_metadata = [None for _ in range(world_size)]
    dist.all_gather_object(peer_metadata, local_metadata, group=cpu_group)
    if rank == 0:
        state.open_peer_buffers(peer_metadata)

    local_indices = (
        torch.arange(num_indices, dtype=torch.int64, device=device) * 3
        + rank * num_indices
        + 1
    ) % num_rows
    indices_per_rank = torch.empty(
        (world_size, num_indices), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(
        indices_per_rank.reshape(-1), local_indices, group=nccl_group
    )

    expected = []
    if rank == 0:
        for layer, buffer in enumerate(buffers):
            value = (
                torch.arange(
                    num_indices * row_elements,
                    dtype=torch.float32,
                    device=device,
                ).view(num_indices, row_elements)
                + layer * 2048
            ).to(torch.bfloat16)
            buffer[local_indices] = value
            expected.append(value)

    # Make the expected values available locally without sharing the payload
    # through the path under test.
    if rank != 0:
        expected = [
            (
                torch.arange(
                    num_indices * row_elements,
                    dtype=torch.float32,
                    device=device,
                ).view(num_indices, row_elements)
                + layer * 2048
            ).to(torch.bfloat16)
            for layer in range(num_layers)
        ]

    if rank == 0:
        state.push(indices_per_rank)
    done = torch.zeros(1, dtype=torch.uint8, device=device)
    dist.broadcast(done, src=0, group=nccl_group)
    torch.cuda.synchronize()

    for layer, buffer in enumerate(buffers):
        torch.testing.assert_close(
            buffer[local_indices], expected[layer], atol=0, rtol=0
        )

    state.close()
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
