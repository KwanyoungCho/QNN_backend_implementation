#include "qnn_loader.h"
#include "qnn_qnnjson.h"
#include "io_alloc.h"
#include "qnn_tensor_util.h"
#include "tokenizer_llama.h"
#include "llm_input_preparer.h"
#include "llm_output_processor.h"

#include <fcntl.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --ctx_dir DIR [--graph_name prefill_forward]"
            << " [--backend_so PATH --system_so PATH]"
            << " [--gguf PATH --prompt STR | --tokens FILE]"
            << " [--token_input NAME] [--ids_output NAME] [--logits_output NAME]"
            << " [--dump_ids FILE] [--decode] [--log_level N] [--align N]\n";
}

static bool read_tokens_i32(const std::string& path, std::vector<int32_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t n = buf.size() / sizeof(int32_t);
  out.resize(n);
  std::memcpy(out.data(), buf.data(), n * sizeof(int32_t));
  return true;
}

int main(int argc, char** argv) {
  std::string ctx_dir;
  std::string graph_name = "prefill_forward";
  std::string backend_so = "libQnnHtp.so";
  std::string system_so = "libQnnSystem.so";
  std::string gguf;
  std::string prompt;
  std::string tokens_file;
  std::string token_input_name;
  std::string ids_output_name;
  std::string logits_output_name;
  std::string dump_ids_file;
  bool decode_out = false;
  int log_level = 5;
  std::size_t align_bytes = 64;
  int num_iters = 1; // run prefill repeatedly by appending next token

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--graph_name" && i + 1 < argc) graph_name = argv[++i];
    else if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
    else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
    else if (a == "--tokens" && i + 1 < argc) tokens_file = argv[++i];
    else if (a == "--token_input" && i + 1 < argc) token_input_name = argv[++i];
    else if (a == "--ids_output" && i + 1 < argc) ids_output_name = argv[++i];
    else if (a == "--logits_output" && i + 1 < argc) logits_output_name = argv[++i];
    else if (a == "--dump_ids" && i + 1 < argc) dump_ids_file = argv[++i];
    else if (a == "--decode") decode_out = true;
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
    else if (a == "--align" && i + 1 < argc) align_bytes = static_cast<std::size_t>(std::stoul(argv[++i]));
    else if (a == "--ids_output" && i + 1 < argc) ids_output_name = argv[++i];
    else if (a == "--logits_output" && i + 1 < argc) logits_output_name = argv[++i];
    else if (a == "--iters" && i + 1 < argc) num_iters = std::max(1, std::stoi(argv[++i]));
  }
  if (ctx_dir.empty()) { usage(argv[0]); return 1; }
  if (tokens_file.empty() && (gguf.empty() || prompt.empty())) {
    std::cerr << "Provide either --tokens FILE or (--gguf PATH and --prompt STR)\n";
    return 1;
  }

  std::vector<int32_t> tokens;
  LlamaTokenizer tok;
  bool tok_ready = false;
  if (!gguf.empty() || decode_out) {
    tok_ready = tok.init(gguf.c_str());
    if (!tok_ready && (!gguf.empty())) { std::cerr << "failed to load gguf\n"; return 1; }
  }
  if (!tokens_file.empty()) {
    if (!read_tokens_i32(tokens_file, tokens)) { std::cerr << "failed to read tokens file\n"; return 1; }
    std::cout << "[info] loaded tokens from file: count=" << tokens.size() << "\n";
  } else {
    std::string formatted = format_llama32_prompt(prompt, /*system*/"");
    if (!tok_ready) { std::cerr << "tokenizer not ready for prompt encode\n"; return 1; }
    tokens = tok.encode(formatted, /*add_special*/true, /*parse_special*/true);
    std::cout << "[info] encoded prompt tokens: count=" << tokens.size() << "\n";
  }

  QnnLoader loader; loader.set_log_level(log_level);
  if (!loader.load(backend_so, system_so)) { std::cerr << "load so failed\n"; return 1; }
  if (!loader.get_interface_provider()) { std::cerr << "get provider failed\n"; return 1; }
  if (!loader.create_backend_and_device()) { std::cerr << "backend/device failed\n"; return 1; }

  // locate first shard pair
  std::string bin_path, json_path;
  {
    DIR* d = opendir(ctx_dir.c_str());
    if (!d) { std::cerr << "open ctx_dir failed\n"; return 1; }
    int best_idx = -1;
    std::map<int,std::string> bins, jsons;
    while (dirent* e = readdir(d)) {
      if (!e || e->d_name[0] == '\0') continue;
      std::string name = e->d_name;
      if (name.rfind("forward_", 0) != 0) continue;
      std::string path = ctx_dir + "/" + name;
      if (name.size() > 12 && name.find(".bin") == name.size() - 4) {
        size_t us = name.find('_'); size_t dot = name.rfind(".bin");
        if (us != std::string::npos && dot != std::string::npos && dot > us + 1) {
          try { bins[std::stoi(name.substr(us+1, dot-(us+1)))] = path; } catch (...) {}
        }
      }
      if (name.size() > 17 && name.find("_json.json") == name.size() - 10) {
        size_t us = name.find('_'); size_t suf = name.rfind("_json.json");
        if (us != std::string::npos && suf != std::string::npos && suf > us + 1) {
          try { jsons[std::stoi(name.substr(us+1, suf-(us+1)))] = path; } catch (...) {}
        }
      }
    }
    closedir(d);
    for (const auto& kv : bins) {
      int idx = kv.first;
      auto it = jsons.find(idx);
      if (it != jsons.end()) { best_idx = idx; bin_path = kv.second; json_path = it->second; break; }
    }
    if (best_idx < 0) { std::cerr << "no shard pair in ctx_dir\n"; return 1; }
  }

  // mmap bin and create context
  int fd = open(bin_path.c_str(), O_RDONLY);
  if (fd < 0) { std::cerr << "open failed: " << bin_path << "\n"; return 1; }
  struct stat st{}; if (fstat(fd, &st) != 0) { std::cerr << "fstat failed\n"; close(fd); return 1; }
  void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) { std::cerr << "mmap failed\n"; close(fd); return 1; }
  if (!loader.create_context_from_binary(addr, static_cast<size_t>(st.st_size))) {
    std::cerr << "context create failed\n"; munmap(addr, st.st_size); close(fd); return 1;
  }

  // parse json, get graph desc
  std::map<std::string, QnnJsonGraphDesc> graphs;
  if (!parse_qnn_json(json_path, graphs)) {
    std::cerr << "parse json failed\n"; munmap(addr, st.st_size); close(fd); return 1;
  }
  auto git = graphs.find(graph_name);
  if (git == graphs.end()) {
    std::cerr << "graph not found: " << graph_name << "\n"; munmap(addr, st.st_size); close(fd); return 1;
  }
  (void)loader.retrieve_graph(0, graph_name);

  // allocate IO
  const QnnJsonGraphDesc& g = git->second;
  QNNIOAllocator alloc; alloc.build_from_qnnjson(g); alloc.allocate(align_bytes);

  // Debug: concise IO summary (only meaningful tensors)
  auto is_interesting_input = [&](const QnnJsonTensorDesc& t) -> bool {
    std::string n = t.name; for (auto& c : n) c = (char)tolower(c);
    if (n.find("token") != std::string::npos || n.find("input_ids") != std::string::npos || n.find("ids") != std::string::npos) return true;
    if (n.find("position") != std::string::npos || n.find("pos_id") != std::string::npos || n.find("pos") != std::string::npos) return true;
    if (n.find("seq_len") != std::string::npos || n.find("input_length") != std::string::npos || n.find("token_count") != std::string::npos || n.find("past_len") != std::string::npos || n.find("start_pos") != std::string::npos) return true;
    if (n.find("atten_mask") != std::string::npos || n.find("attn_mask") != std::string::npos || (n.find("mask") != std::string::npos && n.find("attention") != std::string::npos)) return true;
    return false;
  };
  auto is_interesting_output = [&](const QnnJsonTensorDesc& t) -> bool {
    std::string n = t.name; for (auto& c : n) c = (char)tolower(c);
    bool token_like = (n.find("token") != std::string::npos) || (n.find("ids") != std::string::npos) || (n.find("output_ids") != std::string::npos);
    bool logits_like = (t.dims.size() >= 2) && (t.dims.back() >= 10000) && (t.data_type.find("FLOAT_") != std::string::npos || t.data_type.find("UFIXED_POINT_16") != std::string::npos);
    return token_like || logits_like;
  };
  std::cout << "[info] graph: " << g.graph_name << " inputs=" << g.inputs.size() << " outputs=" << g.outputs.size() << "\n";
  for (const auto& t : g.inputs) {
    if (!is_interesting_input(t)) continue;
    std::cout << "  [in ] name=" << t.name << " type=" << t.data_type << " dims=";
    for (size_t i = 0; i < t.dims.size(); ++i) std::cout << (i?"x":"") << t.dims[i];
    std::cout << " nbytes=" << t.nbytes << "\n";
  }
  for (const auto& t : g.outputs) {
    if (!is_interesting_output(t)) continue;
    std::cout << "  [out] name=" << t.name << " type=" << t.data_type << " dims=";
    for (size_t i = 0; i < t.dims.size(); ++i) std::cout << (i?"x":"") << t.dims[i];
    std::cout << " nbytes=" << t.nbytes << "\n";
  }

  // heuristics to find token input tensor
  auto pick_token_input = [&](const QnnJsonGraphDesc& gd) -> const QnnJsonTensorDesc* {
    if (!token_input_name.empty()) {
      for (const auto& t : gd.inputs) if (t.name == token_input_name) return &t;
    }
    const QnnJsonTensorDesc* best = nullptr;
    for (const auto& t : gd.inputs) {
      std::string n = t.name; for (auto& c : n) c = (char)tolower(c);
      bool name_ok = (n.find("token") != std::string::npos) || (n.find("input_ids") != std::string::npos) || (n.find("ids") != std::string::npos);
      bool type_ok = (t.data_type.find("INT_32") != std::string::npos) || (t.data_type.find("UINT_32") != std::string::npos);
      bool rank_ok = (t.dims.size() == 1 || t.dims.size() == 2);
      if (name_ok && type_ok && rank_ok) { best = &t; break; }
    }
    return best;
  };
  const QnnJsonTensorDesc* tok_in = nullptr;
  if (!token_input_name.empty()) {
    for (const auto& t : g.inputs) if (t.name == token_input_name) { tok_in = &t; break; }
  }
  if (!tok_in) tok_in = pick_token_input(g);
  if (!tok_in) { std::cerr << "failed to locate token input tensor; use --token_input NAME\n"; return 1; }
  std::cout << "[info] token input: name=" << tok_in->name << " nbytes=" << tok_in->nbytes << "\n";

  // fill inputs
  std::vector<Qnn_Tensor_t> in_tensors; in_tensors.reserve(g.inputs.size());
  std::vector<Qnn_Tensor_t> out_tensors; out_tensors.reserve(g.outputs.size());
  std::vector<std::unique_ptr<QnnTensorHolder>> holders; holders.reserve(g.inputs.size()+g.outputs.size());

  auto bind_tensor = [&](const QnnJsonTensorDesc& td, bool is_input) {
    auto it = alloc.bindings().find(td.name);
    if (it == alloc.bindings().end()) return;
    auto h = std::make_unique<QnnTensorHolder>();
    if (h->init_from_json(td, it->second, td.nbytes, is_input)) {
      if (is_input) { in_tensors.push_back(h->tensor()); }
      else { out_tensors.push_back(h->tensor()); }
      holders.push_back(std::move(h));
    }
  };

  // pre-fill all tensors: bind ALL inputs to satisfy QNN's expected tensor count
  for (const auto& t : g.inputs) bind_tensor(t, true);
  for (const auto& t : g.outputs) bind_tensor(t, false);

  std::cout << "[info] bound inputs: " << in_tensors.size() << " (expected=" << g.inputs.size() << ")\n";

  // Use InputPreparer to fill all input tensors
  std::cout << "[info] preparing input tensors with " << tokens.size() << " tokens\n";
  auto get_buffer = [&](const std::string& name) -> void* {
    auto it = alloc.bindings().find(name);
    return (it != alloc.bindings().end()) ? it->second : nullptr;
  };
  
  if (!InputPreparer::auto_fill_inputs(g, get_buffer, tokens, /*verbose=*/true)) {
    std::cerr << "failed to prepare input tensors\n";
    return 1;
  }

  bool ran = loader.execute_graph(0, graph_name, in_tensors, out_tensors);
  std::cout << "prefill graphExecute(" << graph_name << ") => " << (ran ? "OK" : "FAIL") << "\n";

  // Optional: inspect outputs (token ids or logits)
  if (ran) {
    auto find_token_output = [&](const QnnJsonGraphDesc& gd) -> const QnnJsonTensorDesc* {
      if (!ids_output_name.empty()) {
        for (const auto& t : gd.outputs) if (t.name == ids_output_name) return &t;
      }
      for (const auto& t : gd.outputs) {
        std::string n = t.name; for (auto& c : n) c = (char)tolower(c);
        bool name_ok = (n.find("token") != std::string::npos) || (n.find("ids") != std::string::npos) || (n.find("output_ids") != std::string::npos);
        bool type_ok = (t.data_type.find("INT_32") != std::string::npos) || (t.data_type.find("UINT_32") != std::string::npos);
        if (name_ok && type_ok) return &t;
      }
      return nullptr;
    };
    auto find_logits_output = [&](const QnnJsonGraphDesc& gd) -> const QnnJsonTensorDesc* {
      if (!logits_output_name.empty()) {
        for (const auto& t : gd.outputs) if (t.name == logits_output_name) return &t;
      }
      const QnnJsonTensorDesc* best = nullptr;
      size_t best_vocab = 0;
      for (const auto& t : gd.outputs) {
        bool f32 = t.data_type.find("FLOAT_32") != std::string::npos;
        bool f16 = t.data_type.find("FLOAT_16") != std::string::npos;
        bool ufix16 = t.data_type.find("UFIXED_POINT_16") != std::string::npos;
        if (!(f32 || f16 || ufix16) || t.dims.size() < 2) continue;
        uint64_t last = t.dims.back();
        if (last > best_vocab && last >= 10000) { best_vocab = (size_t)last; best = &t; }
      }
      return best;
    };

    auto ids_out = find_token_output(g);
    auto decode_ids = [&](const std::vector<int32_t>& ids){
      if (decode_out && tok_ready) {
        std::string text = tok.decode(ids, /*special*/true);
        std::cout << "decoded: " << text << "\n";
      }
    };

    auto do_argmax_from_logits = [&](const QnnJsonTensorDesc* logits_desc, int& out_id){
      out_id = -1;
      auto it = alloc.bindings().find(logits_desc->name);
      if (it == alloc.bindings().end()) return;
      uint64_t vocab = logits_desc->dims.back();
      uint64_t seq = logits_desc->dims.size() >= 2 ? logits_desc->dims[logits_desc->dims.size()-2] : 1;
      bool f32 = logits_desc->data_type.find("FLOAT_32") != std::string::npos;
      bool f16 = logits_desc->data_type.find("FLOAT_16") != std::string::npos;
      bool ufix16 = logits_desc->data_type.find("UFIXED_POINT_16") != std::string::npos;
      if (vocab == 0) return;
      uint64_t valid_len = tokens.empty() ? 1 : (uint64_t)tokens.size();
      if (valid_len > seq) valid_len = seq;
      uint64_t pos = (valid_len > 0 ? (valid_len - 1) : 0);
      uint64_t offset = pos * vocab;
      std::cout << "[info] logits argmax at pos=" << pos << "/" << seq << "\n";
      if (f32) {
        const float* base = reinterpret_cast<const float*>(it->second);
        const float* row = base + offset;
        float maxv = row[0]; int maxi = 0;
        for (uint64_t i = 1; i < vocab; ++i) if (row[i] > maxv) { maxv = row[i]; maxi = (int)i; }
        out_id = maxi;
      } else if (f16) {
        auto to_f32 = [](uint16_t h){
          uint16_t s = (h >> 15) & 0x1; uint16_t e = (h >> 10) & 0x1F; uint16_t f = h & 0x3FF; float val;
          if (e == 0) { val = (f ? std::ldexp((float)f, -24) : 0.0f); }
          else if (e == 31) { val = f ? NAN : INFINITY; }
          else { val = std::ldexp((float)(f | 0x400), (int)e - 25); }
          return s ? -val : val;
        };
        const uint16_t* base = reinterpret_cast<const uint16_t*>(it->second);
        const uint16_t* row = base + offset;
        float maxv = to_f32(row[0]); int maxi = 0;
        for (uint64_t i = 1; i < vocab; ++i) { float v = to_f32(row[i]); if (v > maxv) { maxv = v; maxi = (int)i; } }
        out_id = maxi;
      } else if (ufix16) {
        const uint16_t* base = reinterpret_cast<const uint16_t*>(it->second);
        const uint16_t* row = base + offset;
        uint16_t maxv = row[0]; int maxi = 0;
        for (uint64_t i = 1; i < vocab; ++i) { uint16_t v = row[i]; if (v > maxv) { maxv = v; maxi = (int)i; } }
        out_id = maxi;
      }
    };

    auto print_topk_from_logits = [&](const QnnJsonTensorDesc* logits_desc, int k){
      auto it = alloc.bindings().find(logits_desc->name);
      if (it == alloc.bindings().end()) return;
      uint64_t vocab = logits_desc->dims.back();
      uint64_t seq = logits_desc->dims.size() >= 2 ? logits_desc->dims[logits_desc->dims.size()-2] : 1;
      bool f32 = logits_desc->data_type.find("FLOAT_32") != std::string::npos;
      bool f16 = logits_desc->data_type.find("FLOAT_16") != std::string::npos;
      bool ufix16 = logits_desc->data_type.find("UFIXED_POINT_16") != std::string::npos;
      if (vocab == 0) return;
      uint64_t valid_len = tokens.empty() ? 1 : (uint64_t)tokens.size();
      if (valid_len > seq) valid_len = seq;
      uint64_t pos = (valid_len > 0 ? (valid_len - 1) : 0);
      uint64_t offset = pos * vocab;
      struct Pair { int id; float val; };
      std::vector<Pair> top; top.reserve(k);
      auto consider = [&](int id, float val){
        if ((int)top.size() < k) { top.push_back({id, val});
          if (top.size()==(size_t)k) std::sort(top.begin(), top.end(), [](const Pair&a,const Pair&b){return a.val>b.val;});
          return; }
        if (val > top.back().val) { top.back() = {id,val}; std::sort(top.begin(), top.end(), [](const Pair&a,const Pair&b){return a.val>b.val;}); }
      };
      if (f32) {
        const float* row = reinterpret_cast<const float*>(it->second) + offset;
        for (uint64_t i=0;i<vocab;++i) consider((int)i, row[i]);
      } else if (f16) {
        auto to_f32 = [](uint16_t h){ uint16_t s=(h>>15)&1,e=(h>>10)&31,f=h&1023; float v; if(e==0){v=(f?std::ldexp((float)f,-24):0.0f);} else if(e==31){v=f?NAN:INFINITY;} else {v=std::ldexp((float)(f|0x400),(int)e-25);} return s?-v:v; };
        const uint16_t* row = reinterpret_cast<const uint16_t*>(it->second) + offset;
        for (uint64_t i=0;i<vocab;++i) consider((int)i, to_f32(row[i]));
      } else if (ufix16) {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(it->second) + offset;
        for (uint64_t i=0;i<vocab;++i) consider((int)i, (float)row[i]); // raw UFIX16; ranking 확인용
      }
      std::cout << "top-" << k << " logits (id:val):";
      for (const auto& p : top) std::cout << " " << p.id << ":" << p.val;
      std::cout << "\n";
    };

    if (ids_out) {
      auto it = alloc.bindings().find(ids_out->name);
      if (it != alloc.bindings().end()) {
        size_t count = (size_t)(ids_out->nbytes / 4);
        const int32_t* p = reinterpret_cast<const int32_t*>(it->second);
        std::vector<int32_t> v(p, p + count);
        size_t show = std::min<size_t>(v.size(), 32);
        std::cout << "output ids (first " << show << "):";
        for (size_t i = 0; i < show; ++i) std::cout << ' ' << v[i];
        std::cout << "\n";
        if (!dump_ids_file.empty()) {
          std::ofstream f(dump_ids_file, std::ios::binary);
          if (f) f.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(int32_t));
        }
        if (decode_out) decode_ids(v);
      }
    } else {
      auto logits = find_logits_output(g);
      if (logits) {
        std::cout << "[DEBUG] logits tensor: " << logits->name 
                  << ", quant_scale=" << logits->quant_scale
                  << ", quant_offset=" << logits->quant_offset
                  << ", quant_encoding=" << logits->quant_encoding << "\n";
        int next_id = -1; do_argmax_from_logits(logits, next_id);
        if (next_id >= 0) {
          // print top-5 for inspection
          print_topk_from_logits(logits, 20);
          std::cout << "argmax next id: " << next_id << "\n";
          if (decode_out) { std::vector<int32_t> one{next_id}; decode_ids(one); }
          // Iterate more prefill runs by appending next_id
          for (int itn = 1; itn < num_iters; ++itn) {
            tokens.push_back(next_id);
            
            // Use InputPreparer to refill all inputs
            if (!InputPreparer::auto_fill_inputs(g, get_buffer, tokens, /*verbose=*/false)) {
              std::cerr << "[info] failed to refill inputs in iteration " << itn << "\n";
              break;
            }
            // run again
            ran = loader.execute_graph(0, graph_name, in_tensors, out_tensors);
            std::cout << "prefill graphExecute(" << graph_name << ") => " << (ran ? "OK" : "FAIL") << "\n";
            if (!ran) break;
            // argmax again
            next_id = -1; do_argmax_from_logits(logits, next_id);
            if (next_id < 0) break;
            print_topk_from_logits(logits, 20);
            std::cout << "argmax next id: " << next_id << "\n";
            if (decode_out) { std::vector<int32_t> one{next_id}; decode_ids(one); }
          }
        }
      }
    }
  }

  alloc.release();
  munmap(addr, st.st_size); close(fd);
  return ran ? 0 : 2;
}


