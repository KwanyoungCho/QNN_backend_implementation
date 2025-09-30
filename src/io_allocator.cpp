#include "io_allocator.h"

#include <cstdlib>
#include <cstring>

namespace llm_test {

uint64_t IoAllocator::build_plan(const GraphIOMeta& meta) {
  group_id_to_size_.clear();
  name_to_ptr_.clear();
  group_buffers_.clear();
  total_planned_bytes_ = 0;

  auto consider = [&](const TensorDesc& t) {
    total_planned_bytes_ += t.nbytes;
    if (t.mutable_buffer_id >= 0) {
      auto& s = group_id_to_size_[t.mutable_buffer_id];
      if (t.nbytes > s) s = t.nbytes;
    }
  };
  for (const auto& t : meta.inputs) consider(t);
  for (const auto& t : meta.outputs) consider(t);
  return total_planned_bytes_;
}

uint64_t IoAllocator::allocate(size_t alignment) {
  // Release previous allocations first
  release();
  total_allocated_bytes_ = 0;

  for (const auto& kv : group_id_to_size_) {
    const int gid = kv.first;
    const uint64_t sz = kv.second;
    void* p = nullptr;
    if (alignment && is_pow2(alignment)) {
      if (posix_memalign(&p, alignment, static_cast<size_t>(sz)) != 0) p = nullptr;
    } else {
      p = std::malloc(static_cast<size_t>(sz));
    }
    if (p == nullptr) continue;
    total_allocated_bytes_ += sz;
    group_buffers_[gid] = BufferInfo{p, sz, gid};
  }
  return total_allocated_bytes_;
}

void IoAllocator::build_bindings(const GraphIOMeta& meta) {
  name_to_ptr_.clear();
  auto map_tensor = [&](const TensorDesc& t) {
    if (t.mutable_buffer_id >= 0) {
      auto it = group_buffers_.find(t.mutable_buffer_id);
      if (it != group_buffers_.end()) {
        name_to_ptr_[t.name] = it->second.ptr;
        return;
      }
    }
    // Per-tensor fallback is deferred to caller (optional behavior)
  };
  for (const auto& t : meta.inputs) map_tensor(t);
  for (const auto& t : meta.outputs) map_tensor(t);
}

void IoAllocator::release() {
  for (auto& kv : group_buffers_) {
    if (kv.second.ptr) std::free(kv.second.ptr);
    kv.second.ptr = nullptr;
    kv.second.size = 0;
  }
  group_buffers_.clear();
  name_to_ptr_.clear();
  total_allocated_bytes_ = 0;
}

} // namespace llm_test
