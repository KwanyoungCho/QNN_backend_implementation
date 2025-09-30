#include "qnn_loader.h"
#include "binary_provider.h"

#include <iostream>
#include <string>
#include <QnnContext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " --backend_so PATH --system_so PATH --ctx_file FILE --graph_name NAME [--log_level N]\n";
}

int main(int argc, char** argv) {
  std::string backend_so;
  std::string system_so;
  std::string ctx_file;
  std::string ctx_dir;
  std::string graph_name;
  int log_level = 5;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--ctx_file" && i + 1 < argc) ctx_file = argv[++i];
    else if (a == "--graph_name" && i + 1 < argc) graph_name = argv[++i];
    else if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
  }
  if (backend_so.empty()) backend_so = "libQnnHtp.so";
  if (system_so.empty()) system_so = "libQnnSystem.so";
  if (ctx_file.empty() && ctx_dir.empty()) { usage(argv[0]); return 1; }

  QnnLoader loader;
  loader.set_log_level(log_level);
  if (!loader.load(backend_so, system_so)) { std::cerr << "load libs failed\n"; return 1; }
  if (!loader.get_interface_provider(nullptr)) { std::cerr << "get provider failed\n"; return 1; }
  if (!loader.create_backend_and_device()) { std::cerr << "backend/device failed\n"; return 1; }

  if (!ctx_file.empty()) {
    int fd = open(ctx_file.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "open ctx_file failed\n"; return 1; }
    struct stat st{}; if (fstat(fd, &st) != 0) { close(fd); std::cerr << "stat failed\n"; return 1; }
    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { close(fd); std::cerr << "mmap failed\n"; return 1; }
    bool ok = loader.create_context_from_binary(addr, static_cast<size_t>(st.st_size));
    munmap(addr, st.st_size); close(fd);
    if (!ok) { std::cerr << "create context failed\n"; return 1; }
  } else {
    llm_test::FileShardProvider provider(ctx_dir);
    if (!provider.init_from_dir({"forward_", "kv_forward_prefill_forward_", "kv_forward_", "prefill_forward_"})) {
      std::cerr << "No shards found under " << ctx_dir << "\n";
      return 1;
    }
    std::vector<void*> params;
    std::vector<std::unique_ptr<llm_test::MappingOwner>> owners;
    if (!provider.build_params(params, owners) || params.size() < 2) {
      std::cerr << "Failed to build shard params\n";
      return 1;
    }
    size_t created = 0;
    for (void* p : params) {
      if (!p) break;
      const QnnContext_Params_t* cp = reinterpret_cast<const QnnContext_Params_t*>(p);
      if (!loader.create_context_from_binary(cp->v1.binaryBuffer, cp->v1.binaryBufferSize)) {
        std::cerr << "create context failed on shard index " << created << "\n";
        for (void* q : params) { if (!q) break; std::free(q); }
        owners.clear();
        return 1;
      }
      ++created;
    }
    for (void* p : params) { if (!p) break; std::free(p); }
    owners.clear();
    std::cout << "Created " << created << " contexts from shards in " << ctx_dir << std::endl;
  }

  if (!graph_name.empty()) {
    bool ok_all = true;
    for (size_t i = 0; i < loader.num_contexts(); ++i) {
      if (!loader.retrieve_graph(i, graph_name)) { std::cerr << "retrieve_graph failed (ctx=" << i << ")\n"; ok_all = false; }
      else { std::cout << "Graph retrieved (ctx=" << i << "): " << graph_name << std::endl; }
    }
    return ok_all ? 0 : 1;
  }
  // No graph name provided: try both names per context
  bool any = false;
  for (size_t i = 0; i < loader.num_contexts(); ++i) {
    for (const std::string& name : {std::string("prefill_forward"), std::string("kv_forward")}) {
      if (loader.retrieve_graph(i, name)) {
        std::cout << "Graph retrieved (ctx=" << i << "): " << name << std::endl;
        any = true;
      }
    }
  }
  if (!any) { std::cerr << "No known graphs found (prefill_forward/kv_forward) in any context" << std::endl; return 1; }
  return 0;
}


