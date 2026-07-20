#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace host::distributed {

using DirectPushHandle = tvm::ffi::Array<char>;

inline DirectPushHandle direct_push_to_handle(void* ptr) {
  DirectPushHandle result;
  cudaIpcMemHandle_t handle;
  RuntimeDeviceCheck(cudaIpcGetMemHandle(&handle, ptr));
  result.reserve(sizeof(handle));
  for (size_t i = 0; i < sizeof(handle); ++i)
    result.push_back(handle.reserved[i]);
  return result;
}

inline cudaIpcMemHandle_t direct_push_from_handle(const DirectPushHandle& array) {
  RuntimeCheck(array.size() == sizeof(cudaIpcMemHandle_t), "Invalid CUDA IPC handle size: ", array.size());
  cudaIpcMemHandle_t handle;
  for (size_t i = 0; i < sizeof(handle); ++i)
    handle.reserved[i] = array[i];
  return handle;
}

struct DirectPushHandleHash {
  std::size_t operator()(const cudaIpcMemHandle_t& handle) const {
    return std::hash<std::string_view>{}({handle.reserved, sizeof(handle.reserved)});
  }
};

struct DirectPushHandleEqual {
  bool operator()(const cudaIpcMemHandle_t& a, const cudaIpcMemHandle_t& b) const {
    return std::memcmp(a.reserved, b.reserved, sizeof(a.reserved)) == 0;
  }
};

class CudaIpcDirectPushBase : public tvm::ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.CudaIpcDirectPush", CudaIpcDirectPushBase, tvm::ffi::Object);

  using InputPair = tvm::ffi::Tuple<int64_t, DirectPushHandle>;  // (byte offset, IPC handle)

  CudaIpcDirectPushBase(uint32_t rank, uint32_t world_size) : m_rank(rank), m_world_size(world_size) {
    RuntimeCheck(world_size > 1, "CUDA IPC direct-push requires world_size > 1");
    RuntimeCheck(rank < world_size, "Invalid direct-push rank: ", rank);
  }

  InputPair share_tensor(const tvm::ffi::TensorView tensor) {
    using namespace host;
    RuntimeCheck(tensor.device().device_type == kDLCUDA, "CUDA IPC direct-push only supports CUDA tensors");
    RuntimeCheck(tensor.IsContiguous(), "CUDA IPC direct-push requires contiguous tensors");

    const auto ptr = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
    CUdeviceptr base = 0;
    size_t allocation_bytes = 0;
    const auto result = cuMemGetAddressRange(&base, &allocation_bytes, ptr);
    RuntimeCheck(result == CUDA_SUCCESS, "cuMemGetAddressRange failed: ", result);

    const auto offset = static_cast<int64_t>(ptr - base);
    const auto tensor_bytes = tensor.numel() * dtype_bytes(tensor.dtype());
    RuntimeCheck(
        offset >= 0 && static_cast<uint64_t>(offset + tensor_bytes) <= allocation_bytes,
        "CUDA IPC direct-push tensor crosses allocation boundaries; use NCCL fallback");

    return InputPair{offset, direct_push_to_handle(reinterpret_cast<void*>(base))};
  }

  void
  open_peer_tensors(tvm::ffi::Array<tvm::ffi::Array<InputPair>> peer_tensors, tvm::ffi::Array<int64_t> local_ptrs) {
    using namespace host;
    RuntimeCheck(m_rank == 0, "Only TP-group rank 0 may open direct-push destination tensors");
    RuntimeCheck(m_device_ptrs == nullptr, "CUDA IPC direct-push tensors are already open");
    RuntimeCheck(peer_tensors.size() == m_world_size, "Direct-push peer count mismatch");
    RuntimeCheck(peer_tensors.size() > 0, "Direct-push peer tensor table is empty");

    m_num_layers = static_cast<uint32_t>(peer_tensors[0].size());
    RuntimeCheck(m_num_layers > 0, "Direct-push layer count must be positive");
    RuntimeCheck(local_ptrs.size() == m_num_layers, "Direct-push local pointer count mismatch");

    std::vector<void*> host_ptrs(m_world_size * m_num_layers);
    for (uint32_t peer = 0; peer < m_world_size; ++peer) {
      const auto& tensors = peer_tensors[peer];
      RuntimeCheck(tensors.size() == m_num_layers, "Direct-push layer count differs across ranks");
      for (uint32_t layer = 0; layer < m_num_layers; ++layer) {
        void* ptr = nullptr;
        if (peer == m_rank) {
          ptr = reinterpret_cast<void*>(local_ptrs[layer]);
        } else {
          const auto pair = tensors[layer];
          const auto offset = pair.get<0>();
          const auto handle = direct_push_from_handle(pair.get<1>());
          auto it = m_ipc_cache.find(handle);
          if (it == m_ipc_cache.end()) {
            void* base = nullptr;
            RuntimeDeviceCheck(cudaIpcOpenMemHandle(&base, handle, cudaIpcMemLazyEnablePeerAccess));
            it = m_ipc_cache.emplace(handle, base).first;
          }
          ptr = static_cast<char*>(it->second) + offset;
        }
        host_ptrs[peer * m_num_layers + layer] = ptr;
      }
    }

    RuntimeDeviceCheck(cudaMalloc(&m_device_ptrs, host_ptrs.size() * sizeof(void*)));
    RuntimeDeviceCheck(
        cudaMemcpy(m_device_ptrs, host_ptrs.data(), host_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice));
  }

  void free() {
    if (m_device_ptrs != nullptr) {
      RuntimeDeviceCheck(cudaFree(m_device_ptrs));
      m_device_ptrs = nullptr;
    }
    for (const auto& [_, ptr] : m_ipc_cache)
      RuntimeDeviceCheck(cudaIpcCloseMemHandle(ptr));
    m_ipc_cache.clear();
    m_num_layers = 0;
  }

  void** device_ptrs() const {
    return m_device_ptrs;
  }
  uint32_t rank() const {
    return m_rank;
  }
  uint32_t world_size() const {
    return m_world_size;
  }
  uint32_t num_layers() const {
    return m_num_layers;
  }

 private:
  const uint32_t m_rank;
  const uint32_t m_world_size;
  uint32_t m_num_layers = 0;
  void** m_device_ptrs = nullptr;
  std::unordered_map<cudaIpcMemHandle_t, void*, DirectPushHandleHash, DirectPushHandleEqual> m_ipc_cache;
};

struct CudaIpcDirectPushRef : public tvm::ffi::ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CudaIpcDirectPushRef, tvm::ffi::ObjectRef, CudaIpcDirectPushBase);
};

}  // namespace host::distributed
