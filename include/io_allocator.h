#pragma once

#include "io_meta.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llm_test {

// IoAllocator: Build an allocation plan from GraphIOMeta and allocate buffers.
// - Group buffers by mutable_buffer_id (>=0) and allocate one buffer per group
//   with the maximum size across member tensors.
// - Tensors without a group id (-1) fallback to per-tensor allocation.
// - Designed to mirror ExecuTorch's mutable buffer sharing semantics while
//   remaining backend-agnostic. Registration to QNN (ION/custom) can be layered
//   on top using QnnLoader APIs.
class IoAllocator {
public:
  struct BufferInfo {
    void* ptr {nullptr};
    uint64_t size {0};
    int group_id {-1};
  };

  // Build plan only (no allocation). Returns total planned bytes.
  uint64_t build_plan(const GraphIOMeta& meta);

  // Allocate according to the last built plan. Alignment must be a power of two.
  // Returns total allocated bytes. Replaces previously allocated buffers.
  uint64_t allocate(size_t alignment);

  // Map tensor name to buffer address (grouped or per-tensor) for binding.
  // Requires prior build_plan and allocate.
  void build_bindings(const GraphIOMeta& meta);

  // Accessors
  const std::map<int, BufferInfo>& group_buffers() const { return group_buffers_; }
  const std::map<std::string, void*>& tensor_bindings() const { return name_to_ptr_; }
  uint64_t total_planned_bytes() const { return total_planned_bytes_; }
  uint64_t total_allocated_bytes() const { return total_allocated_bytes_; }

  // Release all allocated buffers (idempotent).
  void release();

private:
  static bool is_pow2(size_t v) { return v && ((v & (v - 1)) == 0); }

  uint64_t total_planned_bytes_ {0};
  uint64_t total_allocated_bytes_ {0};
  std::map<int, uint64_t> group_id_to_size_;
  std::map<int, BufferInfo> group_buffers_;
  std::map<std::string, void*> name_to_ptr_;
};

} // namespace llm_test
