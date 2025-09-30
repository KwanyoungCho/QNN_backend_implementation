#include "io_meta.h"
#include "qnn_loader.h"
#include "binary_provider.h"
#include "io_allocator.h"
#include <QnnInterface.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <cstdio>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --ctx_dir DIR --prefill_io JSON --kv_io JSON"
            << " [--backend_so PATH --system_so PATH] [--log_level N]"
            << " [--alloc] [--align N] [--bind_plan_out FILE]\n";
}

int main(int argc, char** argv) {
  std::string ctx_dir;
  std::string backend_so;
  std::string system_so;
  int log_level = 5;
  std::string prefill_io_json;
  std::string kv_io_json;
  bool do_alloc = false;
  size_t align_bytes = 64;
  std::string bind_plan_out;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
    else if (a == "--prefill_io" && i + 1 < argc) prefill_io_json = argv[++i];
    else if (a == "--kv_io" && i + 1 < argc) kv_io_json = argv[++i];
    else if (a == "--alloc") do_alloc = true;
    else if (a == "--align" && i + 1 < argc) align_bytes = static_cast<size_t>(std::stoul(argv[++i]));
    else if (a == "--bind_plan_out" && i + 1 < argc) bind_plan_out = argv[++i];
  }

  if (backend_so.empty()) backend_so = "libQnnHtp.so";
  if (system_so.empty()) system_so = "libQnnSystem.so";
  if (ctx_dir.empty() || prefill_io_json.empty() || kv_io_json.empty()) {
    usage(argv[0]);
    return 1;
  }

  // Load IO metadata
  GraphIOMeta prefill_meta, kv_meta;
  if (!parse_graph_io_json(prefill_io_json, prefill_meta)) {
    std::cerr << "Failed to parse prefill_io: " << prefill_io_json << "\n";
    return 1;
  }
  if (!parse_graph_io_json(kv_io_json, kv_meta)) {
    std::cerr << "Failed to parse kv_io: " << kv_io_json << "\n";
    return 1;
  }

  // Initialize QNN loader
  QnnLoader loader;
  loader.set_log_level(log_level);
  if (!loader.load(backend_so, system_so)) {
    std::cerr << "Failed to load QNN libraries\n";
    return 1;
  }
  if (!loader.get_interface_provider()) {
    std::cerr << "Failed to get QNN interface provider\n";
    return 1;
  }
  if (!loader.create_backend_and_device()) {
    std::cerr << "Failed to create backend/device\n";
    return 1;
  }

  // Restore contexts from DIR (forward_ preferred)
  FileShardProvider provider(ctx_dir);
  if (!provider.init_from_dir({"forward_", "kv_forward_prefill_forward_", "kv_forward_", "prefill_forward_"})) {
    std::cerr << "No shards found under " << ctx_dir << "\n";
    return 1;
  }
  std::vector<void*> params;
  std::vector<std::unique_ptr<MappingOwner>> owners;
  if (!provider.build_params(params, owners) || params.empty()) {
    std::cerr << "Failed to build shard params\n";
    return 1;
  }
  size_t created = 0;
  for (void* p : params) {
    if (!p) break;
    const QnnContext_Params_t* cp = reinterpret_cast<const QnnContext_Params_t*>(p);
    bool ok = loader.create_context_from_binary(cp->v1.binaryBuffer, cp->v1.binaryBufferSize);
    if (!ok) {
      std::cerr << "contextCreateFromBinary failed on shard index " << created << "\n";
      for (void* q : params) { if (!q) break; std::free(q); }
      owners.clear();
      return 1;
    }
    ++created;
  }
  for (void* p : params) { if (!p) break; std::free(p); }
  owners.clear();

  // Retrieve graphs by name for the first context (prefill, kv)
  bool ok_prefill = loader.retrieve_graph(0, "prefill_forward");
  bool ok_kv = loader.retrieve_graph(0, "kv_forward");
  if (!ok_prefill && !ok_kv) {
    std::cerr << "Failed to retrieve graphs from context 0\n";
  }

  auto dump_plan = [&](const GraphIOMeta& meta) {
    IoAllocator alloc;
    const uint64_t planned = alloc.build_plan(meta);
    std::cout << "graph: " << meta.graph_name << "\n";
    std::cout << "num_inputs: " << meta.inputs.size() << ", num_outputs: " << meta.outputs.size() << "\n";
    std::cout << "total_io_bytes: " << planned << "\n";
    std::cout << "mutable_buffer_groups: " << alloc.group_buffers().size() << "\n";
    for (const auto& [id, info] : alloc.group_buffers()) {
      std::cout << "  mutbuf[" << id << "] size_bytes=" << info.size << "\n";
    }
    if (do_alloc) {
      const uint64_t allocated = alloc.allocate(align_bytes);
      alloc.build_bindings(meta);
      std::cout << "allocated_bytes: " << allocated << "\n";
      if (!bind_plan_out.empty()) {
        FILE* f = std::fopen(bind_plan_out.c_str(), "a");
        if (f) {
          std::fprintf(f, "{\n\"graph\":\"%s\",\n", meta.graph_name.c_str());
          std::fprintf(f, "\"alloc_bytes\":%llu,\n", static_cast<unsigned long long>(allocated));
          std::fprintf(f, "\"bindings\":[\n");
          bool first = true;
          for (const auto& kv : alloc.tensor_bindings()) {
            if (!first) std::fprintf(f, ",\n");
            first = false;
            std::fprintf(f, "{\"name\":\"%s\",\"addr\":\"%p\"}", kv.first.c_str(), kv.second);
          }
          std::fprintf(f, "\n]\n}\n");
          std::fclose(f);
        }
      }
      alloc.release();
    }
  };

  dump_plan(prefill_meta);
  dump_plan(kv_meta);
  return 0;
}


