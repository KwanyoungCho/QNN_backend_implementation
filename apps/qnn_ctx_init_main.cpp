#include "qnn_loader.h"
#include "binary_provider.h"

#include <QnnInterface.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " --ctx_file FILE | --ctx_dir DIR --backend_so PATH --system_so PATH [--provider NAME] [--log_level N]\n";
}

int main(int argc, char** argv) {
  std::string ctx_dir = "/home/chokwans99/tmp/executorch/ctx_out";
  std::string backend_so;
  std::string system_so;
  std::string ctx_file;
  int log_level = 5;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--ctx_file" && i + 1 < argc) ctx_file = argv[++i];
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
  }

  if (backend_so.empty()) backend_so = "libQnnHtp.so";
  if (system_so.empty()) system_so = "libQnnSystem.so";

  QnnLoader loader;
  loader.set_log_level(log_level);
  if (!loader.load(backend_so, system_so)) {
    std::cerr << "Failed to load QNN libraries\n";
    return 1;
  }
  const void* qnn = loader.get_interface_provider();
  if (!qnn) {
    std::cerr << "Failed to get QNN interface provider\n";
    return 1;
  }
  if (!loader.create_backend_and_device()) {
    std::cerr << "Failed to create backend/device\n";
    return 1;
  }

  if (!ctx_file.empty()) {
    int fd = open(ctx_file.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "Failed to open ctx_file\n"; return 1; }
    struct stat st{}; if (fstat(fd, &st) != 0) { close(fd); std::cerr << "stat failed\n"; return 1; }
    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { close(fd); std::cerr << "mmap failed\n"; return 1; }
    bool ok = loader.create_context_from_binary(addr, static_cast<size_t>(st.st_size));
    munmap(addr, st.st_size); close(fd);
    if (!ok) { std::cerr << "contextCreateFromBinary failed" << std::endl; return 1; }
    std::cout << "Context created via single-binary API from --ctx_file" << std::endl;
  } else {
    FileShardProvider provider(ctx_dir);
    if (!provider.init_from_dir({"forward_", "kv_forward_prefill_forward_", "kv_forward_", "prefill_forward_"})) {
      std::cerr << "No shards found under " << ctx_dir << "\n";
      return 1;
    }
    std::vector<void*> params;
    std::vector<std::unique_ptr<MappingOwner>> owners;
    if (!provider.build_params(params, owners) || params.size() < 2) {
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
    std::cout << "Created " << created << " QNN contexts from shards in " << ctx_dir << std::endl;
  }

  return 0;
}


