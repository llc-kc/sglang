#include <sgl_kernel/distributed/cuda_ipc_direct_push.cuh>

inline void register_cuda_ipc_direct_push() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::CudaIpcDirectPushBase;
  refl::ObjectDef<Class>()
      .def(refl::init<uint32_t, uint32_t>(), "__init__")
      .def("share_tensor", &Class::share_tensor)
      .def("open_peer_tensors", &Class::open_peer_tensors)
      .def("free", &Class::free);
}
