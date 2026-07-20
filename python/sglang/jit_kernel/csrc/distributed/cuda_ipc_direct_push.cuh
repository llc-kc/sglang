#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/cuda_ipc_direct_push.cuh>

#include <algorithm>
#include <cstdint>

namespace {

using host::distributed::CudaIpcDirectPushRef;

template <uint32_t kNumThreads>
struct DirectPushPackage;

template <>
struct DirectPushPackage<8> {
  using type = uint4;
};

template <>
struct DirectPushPackage<16> {
  using type = uint2;
};

template <>
struct DirectPushPackage<32> {
  using type = uint1;
};

template <typename T, uint32_t N>
struct DirectPushStorage {
  T data[N];
};

template <int64_t kBytes, uint32_t kNumThreads>
__device__ auto direct_push_load(const void* src) {
  static_assert(kBytes % 128 == 0);
  using Package = typename DirectPushPackage<kNumThreads>::type;
  using Storage = DirectPushStorage<Package, kBytes / 128>;
  const auto packed = static_cast<const Package*>(src);
  const auto lane = threadIdx.x % kNumThreads;
  Storage value;
#pragma unroll
  for (uint32_t i = 0; i < kBytes / 128; ++i)
    value.data[i] = packed[i * kNumThreads + lane];
  return value;
}

template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
__device__ void direct_push_store(void* dst, const Storage& value) {
  using Package = typename DirectPushPackage<kNumThreads>::type;
  auto packed = static_cast<Package*>(dst);
  const auto lane = threadIdx.x % kNumThreads;
#pragma unroll
  for (uint32_t i = 0; i < kBytes / 128; ++i)
    packed[i * kNumThreads + lane] = value.data[i];
}

struct DirectPushParams {
  void** peer_ptrs;
  const void* indices;
  int64_t src_stride_bytes;
  int64_t dst_stride_bytes;
  uint32_t length;
  uint32_t num_layers;
  uint32_t world_size;
};

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize, typename IndexT>
__global__ __launch_bounds__(kBlockSize, 1) void cuda_ipc_direct_push_kernel(DirectPushParams params) {
  using namespace device;
  using src_ptr_t = const void*;
  using dst_ptr_t = void*;

  static_assert(kElementSize % 128 == 0);
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  const auto indices = static_cast<const IndexT*>(params.indices);

  for (uint32_t token = work_id; token < params.length; token += kNumWorkers) {
    const auto src_pos = indices[token];  // group rank 0 occupies row 0
    for (uint32_t layer = 0; layer < params.num_layers; ++layer) {
      const auto src_base = static_cast<src_ptr_t>(params.peer_ptrs[layer]);
      const auto src = pointer::offset(src_base, src_pos * params.src_stride_bytes);
      const auto value = direct_push_load<kElementSize, kNumThreads>(src);

      for (uint32_t peer = 1; peer < params.world_size; ++peer) {
        const auto dst_pos = indices[peer * params.length + token];
        const auto dst_base = static_cast<dst_ptr_t>(params.peer_ptrs[peer * params.num_layers + layer]);
        const auto dst = pointer::offset(dst_base, dst_pos * params.dst_stride_bytes);
        direct_push_store<kElementSize, kNumThreads>(dst, value);
      }
    }
  }
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
void cuda_ipc_direct_push(
    CudaIpcDirectPushRef obj,
    const tvm::ffi::TensorView indices,
    const int64_t src_stride_bytes,
    const int64_t dst_stride_bytes) {
  using namespace host;

  auto world = SymbolicSize{"world size"};
  auto length = SymbolicSize{"indices length"};
  auto index_dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  TensorMatcher({world, length}).with_dtype<int32_t, int64_t>(index_dtype).with_device<kDLCUDA>(device).verify(indices);

  auto* state = obj.get();
  RuntimeCheck(state->rank() == 0, "CUDA IPC direct-push kernel must run on TP-group rank 0");
  RuntimeCheck(state->device_ptrs() != nullptr, "CUDA IPC direct-push peer pointers are not initialized");
  RuntimeCheck(world.unwrap() == state->world_size(), "CUDA IPC direct-push world size mismatch");
  RuntimeCheck(src_stride_bytes >= kElementSize, "CUDA IPC direct-push source stride is too small");
  RuntimeCheck(dst_stride_bytes >= kElementSize, "CUDA IPC direct-push destination stride is too small");

  constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
  const auto num_blocks = std::min(div_ceil(static_cast<uint32_t>(length.unwrap()), kWorkersPerBlock), kBlockQuota);
  if (num_blocks == 0) return;

  const DirectPushParams params{
      .peer_ptrs = state->device_ptrs(),
      .indices = indices.data_ptr(),
      .src_stride_bytes = src_stride_bytes,
      .dst_stride_bytes = dst_stride_bytes,
      .length = static_cast<uint32_t>(length.unwrap()),
      .num_layers = state->num_layers(),
      .world_size = state->world_size(),
  };
  const auto kernel = index_dtype.unwrap().bits == 32
                          ? cuda_ipc_direct_push_kernel<kElementSize, kUnroll, kBlockQuota, kBlockSize, int32_t>
                          : cuda_ipc_direct_push_kernel<kElementSize, kUnroll, kBlockQuota, kBlockSize, int64_t>;
  LaunchKernel(num_blocks, kBlockSize, device.unwrap())(kernel, params);
}

}  // namespace
