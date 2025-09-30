#include "qnn_loader.h"
#include "qnn_qnnjson.h"
#include "io_alloc.h"
#include "qnn_tensor_util.h"

#include <QnnInterface.h>
#include <cstdio>
#include <algorithm>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --ctx_dir DIR"
            << " [--backend_so PATH --system_so PATH] [--log_level N]"
            << " [--align N] [--bind_plan_out FILE]\n";
}

int main(int argc, char** argv) {
  std::string ctx_dir;
  std::string backend_so;
  std::string system_so;
  int log_level = 5;
  std::size_t align_bytes = 64;
  std::string bind_plan_out;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
    else if (a == "--align" && i + 1 < argc) align_bytes = static_cast<std::size_t>(std::stoul(argv[++i]));
    else if (a == "--bind_plan_out" && i + 1 < argc) bind_plan_out = argv[++i];
  }

  if (backend_so.empty()) backend_so = "libQnnHtp.so";
  if (system_so.empty()) system_so = "libQnnSystem.so";
  if (ctx_dir.empty()) { usage(argv[0]); return 1; }

  // 1) QNN 초기화(Executorch 흐름과 동일)
  QnnLoader loader; loader.set_log_level(log_level);
  if (!loader.load(backend_so, system_so)) { std::cerr << "load so failed\n"; return 1; }
  if (!loader.get_interface_provider()) { std::cerr << "get provider failed\n"; return 1; }
  if (!loader.create_backend_and_device()) { std::cerr << "backend/device failed\n"; return 1; }

  // 2) ctx_dir 스캔: forward_{i}.bin ↔ forward_{i}_json.json 짝 구성
  struct Shard { int idx; std::string bin_path; std::string json_path; };
  std::vector<Shard> shards;
  if (DIR* d = opendir(ctx_dir.c_str())) {
    std::map<int, std::string> bins, jsons;
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '\0') continue;
      std::string name = e->d_name;
      if (name.rfind("forward_", 0) != 0) continue;
      std::string path = ctx_dir + "/" + name;
      // forward_<n>.bin
      if (name.size() > 12 && name.find(".bin") == name.size() - 4) {
        size_t us = name.find('_');
        size_t dot = name.rfind(".bin");
        if (us != std::string::npos && dot != std::string::npos && dot > us + 1) {
          std::string num = name.substr(us + 1, dot - (us + 1));
          try { bins[std::stoi(num)] = path; } catch (...) {}
        }
      }
      // forward_<n>_json.json
      if (name.size() > 17 && name.find("_json.json") == name.size() - 10) {
        size_t us = name.find('_');
        size_t suf = name.rfind("_json.json");
        if (us != std::string::npos && suf != std::string::npos && suf > us + 1) {
          std::string num = name.substr(us + 1, suf - (us + 1));
          try { jsons[std::stoi(num)] = path; } catch (...) {}
        }
      }
    }
    closedir(d);
    for (const auto& kv : bins) {
      int i = kv.first;
      auto it = jsons.find(i);
      if (it != jsons.end()) shards.push_back({i, kv.second, it->second});
    }
  }
  if (shards.empty()) { std::cerr << "no shard pairs found in ctx_dir\n"; return 1; }
  std::sort(shards.begin(), shards.end(), [](const Shard& a, const Shard& b){ return a.idx < b.idx; });

  auto alloc_and_run = [&](size_t ctx_idx, const QnnJsonGraphDesc& g, std::size_t align, const std::string& out) {
    // [중요] ExecuTorch는 이 단계에서 mutable buffer 공유가 있으면 1회만 할당해 공유한다.
    // 현재 입력 JSON에는 공유 정보가 없으므로 텐서별 할당만 수행한다.
    QNNIOAllocator alloc;
    alloc.build_from_qnnjson(g);
    auto bytes = alloc.allocate(align);
    std::cout << "graph: " << g.graph_name << "\n";
    std::cout << "num_inputs: " << g.inputs.size() << ", num_outputs: " << g.outputs.size() << "\n";
    std::cout << "allocated_bytes: " << bytes << "\n";
    if (!out.empty()) {
      FILE* f = std::fopen(out.c_str(), "a");
      if (f) {
        std::fprintf(f, "{\n\"graph\":\"%s\",\n", g.graph_name.c_str());
        std::fprintf(f, "\"alloc_bytes\":%llu,\n", static_cast<unsigned long long>(bytes));
        std::fprintf(f, "\"bindings\":[\n");
        bool first = true;
        for (const auto& kv : alloc.bindings()) {
          if (!first) std::fprintf(f, ",\n");
          first = false;
          std::fprintf(f, "{\"name\":\"%s\",\"addr\":\"%p\"}", kv.first.c_str(), kv.second);
        }
        std::fprintf(f, "\n]\n}\n");
        std::fclose(f);
      }
    }
    // QNN 텐서 배열 구성 후 그래프 실행 1회 시도
    std::vector<Qnn_Tensor_t> in_tensors;
    std::vector<Qnn_Tensor_t> out_tensors;
    std::vector<std::unique_ptr<QnnTensorHolder>> holders;
    holders.reserve(g.inputs.size() + g.outputs.size());
    // JSON에 id가 있을 경우 해당 id/name으로 텐서를 구성(가급적 ID 유지)
    for (const auto& t : g.inputs) {
      auto it = alloc.bindings().find(t.name);
      if (it == alloc.bindings().end()) continue;
      auto h = std::make_unique<QnnTensorHolder>();
      if (h->init_from_json(t, it->second, t.nbytes, /*is_input*/true)) {
        in_tensors.push_back(h->tensor());
        holders.push_back(std::move(h));
      }
    }
    for (const auto& t : g.outputs) {
      auto it = alloc.bindings().find(t.name);
      if (it == alloc.bindings().end()) continue;
      auto h = std::make_unique<QnnTensorHolder>();
      if (h->init_from_json(t, it->second, t.nbytes, /*is_input*/false)) {
        out_tensors.push_back(h->tensor());
        holders.push_back(std::move(h));
      }
    }
    // ExecuTorch와 동일: 등록된 텐서 ID를 가진 배열을 넘기고 clientBuf만 채워 실행
    bool ran = loader.execute_graph(ctx_idx, g.graph_name, in_tensors, out_tensors);
    std::cout << "graphExecute(" << g.graph_name << ") => " << (ran ? "OK" : "FAIL") << "\n";
    alloc.release();
  };

  // 3) 샤드별로 컨텍스트 복원 → 그래프 retrieve → JSON으로 I/O 할당
  size_t ctx_index = 0;
  for (const auto& sh : shards) {
    // mmap bin
    int fd = open(sh.bin_path.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "open failed: " << sh.bin_path << "\n"; return 1; }
    struct stat st{}; if (fstat(fd, &st) != 0) { std::cerr << "fstat failed\n"; close(fd); return 1; }
    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { std::cerr << "mmap failed\n"; close(fd); return 1; }
    // create context
    if (!loader.create_context_from_binary(addr, static_cast<size_t>(st.st_size))) {
      std::cerr << "context create failed for shard " << sh.idx << "\n";
      munmap(addr, st.st_size); close(fd); return 1;
    }
    // parse json for this shard
    std::map<std::string, QnnJsonGraphDesc> graphs;
    if (!parse_qnn_json(sh.json_path, graphs)) {
      std::cerr << "parse json failed: " << sh.json_path << "\n";
      munmap(addr, st.st_size); close(fd); return 1;
    }
    // retrieve graphs existing in JSON
    if (graphs.count("prefill_forward")) (void)loader.retrieve_graph(ctx_index, "prefill_forward");
    if (graphs.count("kv_forward")) (void)loader.retrieve_graph(ctx_index, "kv_forward");
    // allocate for each present graph
    if (graphs.count("prefill_forward")) alloc_and_run(ctx_index, graphs["prefill_forward"], align_bytes, bind_plan_out);
    if (graphs.count("kv_forward")) alloc_and_run(ctx_index, graphs["kv_forward"], align_bytes, bind_plan_out);
    // cleanup mapping (context stays alive)
    munmap(addr, st.st_size); close(fd);
    ctx_index++;
  }
  return 0;
}


