#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm_test {

struct TensorDesc {
  std::string name;
  uint64_t nbytes {0};
  int mutable_buffer_id {-1};
};

struct GraphIOMeta {
  std::string graph_name;
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
};

// Parse a minimal subset of the runtime-dumped graph_io.json
// Expected keys per tensor object: name (string), nbytes (number), mutableBufferId (number)
// Returns true on success.
bool parse_graph_io_json(const std::string& json_path, GraphIOMeta& out);

}


