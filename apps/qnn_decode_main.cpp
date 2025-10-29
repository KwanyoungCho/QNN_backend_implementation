#include "qnn_loader.h"
#include "qnn_qnnjson.h"
#include "io_alloc.h"
#include "qnn_tensor_util.h"
#include "tokenizer_llama.h"
#include "llm_input_preparer.h"
#include "llm_kv_cache_manager.h"

#include <fcntl.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --ctx_dir DIR"
            << " --gguf PATH --prompt STR"
            << " [--backend_so PATH --system_so PATH]"
            << " [--max_gen N] [--log_level N]\n";
}

int main(int argc, char** argv) {
  std::string ctx_dir;
  std::string backend_so = "libQnnHtp.so";
  std::string system_so = "libQnnSystem.so";
  std::string gguf;
  std::string prompt;
  int max_gen = 10;
  int log_level = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ctx_dir" && i + 1 < argc) ctx_dir = argv[++i];
    else if (a == "--backend_so" && i + 1 < argc) backend_so = argv[++i];
    else if (a == "--system_so" && i + 1 < argc) system_so = argv[++i];
    else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
    else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
    else if (a == "--max_gen" && i + 1 < argc) max_gen = std::stoi(argv[++i]);
    else if (a == "--log_level" && i + 1 < argc) log_level = std::stoi(argv[++i]);
  }

  if (ctx_dir.empty() || gguf.empty() || prompt.empty()) {
    usage(argv[0]);
    return 1;
  }

  // ========== 1. Initialize Tokenizer ==========
  LlamaTokenizer tokenizer;
  if (!tokenizer.init(gguf.c_str())) {
    std::cerr << "Failed to load tokenizer\n";
    return 1;
  }

  // Use raw prompt without special tokens and system prompt
  // std::string formatted_prompt = format_llama32_prompt(prompt, "");
  // std::vector<int32_t> tokens = tokenizer.encode(formatted_prompt, true, true);
  std::vector<int32_t> tokens = tokenizer.encode(prompt, false, false);  // No special tokens
  if (tokens.empty()) {
    std::cerr << "Failed to encode prompt\n";
    return 1;
  }

  std::cout << "[Tokenizer] Encoded " << tokens.size() << " tokens\n";
  std::cout << "[Prompt] " << prompt << "\n";

  // ========== 2. Initialize QNN ==========
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
    std::cerr << "Failed to create QNN backend/device\n";
    return 1;
  }

  // ========== 3. Load Context Binary ==========
  std::string bin_path = ctx_dir + "/forward_0.bin";
  std::string json_path = ctx_dir + "/forward_0_json.json";

  int fd = open(bin_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "Failed to open context binary: " << bin_path << "\n";
    return 1;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    std::cerr << "Failed to stat context binary\n";
    close(fd);
    return 1;
  }

  void* ctx_addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (ctx_addr == MAP_FAILED) {
    std::cerr << "Failed to mmap context binary\n";
    close(fd);
    return 1;
  }

  if (!loader.create_context_from_binary(ctx_addr, st.st_size)) {
    std::cerr << "Failed to create QNN context from binary\n";
    munmap(ctx_addr, st.st_size);
    close(fd);
    return 1;
  }

  std::cout << "[QNN] Context loaded successfully\n";

  // ========== 4. Parse JSON and Retrieve Graphs ==========
  std::map<std::string, QnnJsonGraphDesc> graphs;
  if (!parse_qnn_json(json_path, graphs)) {
    std::cerr << "Failed to parse JSON: " << json_path << "\n";
    return 1;
  }

  if (graphs.find("prefill_forward") == graphs.end() ||
      graphs.find("kv_forward") == graphs.end()) {
    std::cerr << "Required graphs not found in JSON\n";
    return 1;
  }

  if (!loader.retrieve_graph(0, "prefill_forward")) {
    std::cerr << "Failed to retrieve prefill_forward graph\n";
    return 1;
  }

  if (!loader.retrieve_graph(0, "kv_forward")) {
    std::cerr << "Failed to retrieve kv_forward graph\n";
    return 1;
  }

  const auto& prefill_graph = graphs["prefill_forward"];
  const auto& kv_graph = graphs["kv_forward"];

  std::cout << "[Graphs] prefill_forward: " << prefill_graph.inputs.size() 
            << " inputs, " << prefill_graph.outputs.size() << " outputs\n";
  std::cout << "[Graphs] kv_forward: " << kv_graph.inputs.size() 
            << " inputs, " << kv_graph.outputs.size() << " outputs\n";

  // ========== 5. Extract Metadata from Graphs ==========
  // Find attention mask to get context_len and ar_len
  int32_t prefill_ar_len = 0;
  int32_t kv_ar_len = 0;
  int32_t context_len = 0;
  
  for (const auto& t : prefill_graph.inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      prefill_ar_len = t.dims[t.dims.size() - 2];
      context_len = t.dims[t.dims.size() - 1];
      break;
    }
  }
  
  for (const auto& t : kv_graph.inputs) {
    std::string name_lower = t.name;
    for (auto& c : name_lower) c = (char)tolower(c);
    if (name_lower.find("atten_mask") != std::string::npos && t.dims.size() >= 2) {
      kv_ar_len = t.dims[t.dims.size() - 2];
      if (context_len == 0) context_len = t.dims[t.dims.size() - 1];
      break;
    }
  }

  // Extract num_layers, num_heads, head_dim from KV cache tensors
  // KV cache input/output tensor naming: {input/output}_X_args_Y_Z
  // Input dims: [batch=1, max_cache_len, head_dim]
  // Output dims: [batch=1, ar_len, head_dim]
  int32_t num_layers = 0;
  int32_t num_heads = 0;
  int32_t head_dim = 0;
  
  // Find head_dim from any KV cache input tensor
  for (const auto& t : prefill_graph.inputs) {
    if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
      // dims format: [batch=1, max_cache_len, head_dim]
      head_dim = t.dims[2];
      if (head_dim > 0) break;
    }
  }
  
  // Count unique KV cache tensors
  // Naming pattern: input_2_args_128_0 (input index 2, internal ID 128, batch index 0)
  std::set<std::string> kv_cache_names;
  
  for (const auto& t : prefill_graph.inputs) {
    if (t.name.find("_args_") != std::string::npos && t.dims.size() == 3) {
      kv_cache_names.insert(t.name);
    }
  }
  
  // Total KV cache tensors = 2 (K+V) * num_layers * num_heads
  // For llama 1B: 2 * 16 * 8 = 256 tensors
  int total_kv_tensors = kv_cache_names.size();
  
  // Deduce num_layers and num_heads
  // Assuming llama 1B: 16 layers, 8 KV heads
  num_layers = 16;
  num_heads = total_kv_tensors / (2 * num_layers);
  
  // ===== Cache Length 계산 (그래프별로 다름!) =====
  // ExecutorchReader는 AR length에 따라 KV cache size를 동적으로 변경 (rearrange_cache)
  // 
  // Prefill (AR=32): cache_len = context_len - 32 = 512 - 32 = 480
  // Decode (AR=1):   cache_len = context_len - 1  = 512 - 1  = 511
  //
  // 우리는 rearrange를 구현하지 않았으므로, Decode 기준 (더 큰 값) 사용
  // → 문제: Prefill 그래프는 480을 기대하는데, 우리는 511 제공
  //
  // 해결: 각 그래프의 실제 KV cache input dimensions 확인 필요!
  
  // ===== Cache Length 계산: ExecutorchReader 방식 =====
  // 
  // ExecutorchReader는 Prefill 기준으로 할당 후 Decode로 rearrange!
  // - Prefill (AR=32): cache_len = context_len - 32 = 512 - 32 = 480
  // - Decode (AR=1):   cache_len = context_len - 1  = 512 - 1  = 511
  // 
  // 초기 할당: Prefill 기준 (480)
  // Prefill→Decode 전환: rearrange_cache(32, 1) 호출하여 480→511 확장
  
  int32_t prefill_cache_len = context_len - prefill_ar_len;  // 512 - 32 = 480
  int32_t kv_cache_len = context_len - kv_ar_len;            // 512 - 1 = 511
  int32_t max_cache_len = kv_cache_len;  // ✅ 511 (최대, rearrange 후 사용)
  int32_t max_ar_len = prefill_ar_len;  // ✅ Prefill AR length (32)

  std::cout << "[Metadata] context_len=" << context_len 
            << ", prefill_ar=" << prefill_ar_len
            << ", kv_ar=" << kv_ar_len << "\n";
  std::cout << "[Metadata] num_layers=" << num_layers 
            << ", num_heads=" << num_heads
            << ", head_dim=" << head_dim << "\n";
  std::cout << "[Metadata] prefill_cache_len=" << prefill_cache_len
            << ", kv_cache_len=" << kv_cache_len 
            << ", max_cache_len=" << max_cache_len << "\n";
  std::cout << "[Metadata] max_ar_len (initial)=" << max_ar_len << "\n";

  // ========== 6. Initialize KV Cache Manager (최대 cache_len=511로 할당!) ==========
  // 
  // ✅ CRITICAL FIX: ExecutorchReader는 최대 cache_len (511)로 메모리 할당!
  // - 메모리는 511로 할당, Prefill은 480만 사용
  // - rearrange_cache()는 메모리 재할당이 아닌 stride 변경만!
  // - 480→511 rearrange는 같은 버퍼 내에서 memmove로 처리
  //
  // 이전 버그: initial_cache_len=480으로 할당 → rearrange 시 511 접근 → out-of-bounds!
  
  LLMKVCacheManager::Metadata kv_metadata{
      context_len,        // 512
      head_dim,           // 64
      max_ar_len,         // 32 (Prefill AR length)
      max_cache_len,      // ✅ 511 (최대 cache_len, rearrange 후 접근)
      num_heads,          // 8
      num_layers          // 16
  };

  LLMKVCacheManager kv_manager(kv_metadata);
  if (!kv_manager.allocate()) {
    std::cerr << "Failed to allocate KV cache memory\n";
    return 1;
  }

  std::cout << "[KV Cache] Allocated " << (kv_manager.total_cache_size() / 1024.0 / 1024.0) 
            << " MiB (max_cache_len=" << max_cache_len << ")\n";

  // ========== 7. Allocate I/O for Both Graphs ==========
  QNNIOAllocator prefill_alloc;
  prefill_alloc.build_from_qnnjson(prefill_graph);
  auto prefill_bytes = prefill_alloc.allocate(64);
  std::cout << "[Prefill] Allocated " << (prefill_bytes / 1024.0) << " KiB for I/O\n";

  QNNIOAllocator kv_alloc;
  kv_alloc.build_from_qnnjson(kv_graph);
  auto kv_bytes = kv_alloc.allocate(64);
  std::cout << "[KV Decode] Allocated " << (kv_bytes / 1024.0) << " KiB for I/O\n";

  // ========== 8. Map KV Cache Tensors to Shared Memory ==========
  // Build mapping: tensor_name -> (layer, head, is_key)
  // Pattern: input_X_args_Y_Z where Y encodes layer/head info
  std::map<std::string, void*> prefill_kv_input_override;
  std::map<std::string, void*> prefill_kv_output_override;
  std::map<std::string, void*> kv_kv_input_override;
  std::map<std::string, void*> kv_kv_output_override;

  std::cout << "[KV Binding] Mapping KV cache tensors to shared memory...\n";
  
  // For now, use simple index-based mapping
  // Total KV cache: 256 tensors (16 layers * 8 heads * 2 (K+V))
  int kv_idx = 0;
  for (int32_t layer = 0; layer < num_layers; ++layer) {
    for (int32_t head = 0; head < num_heads; ++head) {
      // K cache
      const auto& k_buf = kv_manager.get_k_cache(layer, head);
      // V cache
      const auto& v_buf = kv_manager.get_v_cache(layer, head);
      
      // We need to match tensor names - for now skip detailed mapping
      // TODO: Implement proper tensor name -> layer/head mapping
      kv_idx += 2;
    }
  }
  
  std::cout << "[KV Binding] Total KV cache buffers: " << (num_layers * num_heads * 2) << "\n";

  // ========== 9. Prepare Prefill Inputs ==========
  auto get_prefill_buffer = [&](const std::string& name) -> void* {
    // Check if it's a KV cache input (should use shared memory)
    auto it = prefill_kv_input_override.find(name);
    if (it != prefill_kv_input_override.end()) return it->second;
    
    // Otherwise use allocated buffer
    auto& bindings = prefill_alloc.bindings();
    auto bit = bindings.find(name);
    return (bit != bindings.end()) ? bit->second : nullptr;
  };

  if (!InputPreparer::auto_fill_inputs(prefill_graph, get_prefill_buffer, tokens, true)) {
    std::cerr << "Failed to prepare prefill inputs\n";
    return 1;
  }

  // ========== 10. Build QNN Tensors for Prefill ==========
  std::vector<Qnn_Tensor_t> prefill_inputs, prefill_outputs;
  std::vector<std::unique_ptr<QnnTensorHolder>> prefill_holders;
  
  for (const auto& t : prefill_graph.inputs) {
    void* buf = get_prefill_buffer(t.name);
    if (!buf) continue;
    
    auto h = std::make_unique<QnnTensorHolder>();
    if (h->init_from_json(t, buf, t.nbytes, true)) {
      prefill_inputs.push_back(h->tensor());
      prefill_holders.push_back(std::move(h));
    }
  }
  
  for (const auto& t : prefill_graph.outputs) {
    auto& bindings = prefill_alloc.bindings();
    auto it = bindings.find(t.name);
    if (it == bindings.end()) continue;
    
    auto h = std::make_unique<QnnTensorHolder>();
    if (h->init_from_json(t, it->second, t.nbytes, false)) {
      prefill_outputs.push_back(h->tensor());
      prefill_holders.push_back(std::move(h));
    }
  }

  std::cout << "[Prefill] Prepared " << prefill_inputs.size() << " inputs, " 
            << prefill_outputs.size() << " outputs\n";

  // ========== 11. Execute Prefill ==========
  std::cout << "\n========== PREFILL PHASE ==========\n";
  if (!loader.execute_graph(0, "prefill_forward", prefill_inputs, prefill_outputs)) {
    std::cerr << "Prefill execution failed\n";
    return 1;
  }
  std::cout << "[Prefill] Execution successful\n";

  // ========== 12. Extract Logits and Decode First Token ==========
  const QnnJsonTensorDesc* logits_desc = nullptr;
  for (const auto& t : prefill_graph.outputs) {
    if (t.name.find("squeeze") != std::string::npos || 
        t.name.find("logit") != std::string::npos) {
      logits_desc = &t;
      break;
    }
  }

  if (!logits_desc) {
    std::cerr << "Logits output not found\n";
    return 1;
  }

  auto& prefill_bindings = prefill_alloc.bindings();
  auto logits_it = prefill_bindings.find(logits_desc->name);
  if (logits_it == prefill_bindings.end()) {
    std::cerr << "Logits buffer not found\n";
    return 1;
  }
  
  void* logits_buf = logits_it->second;
  const uint16_t* logits_u16 = reinterpret_cast<const uint16_t*>(logits_buf);
  
  // Find argmax at last valid token position
  int32_t vocab_size = 128256;
  int32_t last_token_offset = (tokens.size() - 1) * vocab_size;
  
  uint16_t max_val = logits_u16[last_token_offset];
  int32_t next_token = 0;
  for (int32_t i = 1; i < vocab_size; ++i) {
    if (logits_u16[last_token_offset + i] > max_val) {
      max_val = logits_u16[last_token_offset + i];
      next_token = i;
    }
  }

  std::string decoded = tokenizer.decode({next_token});
  std::cout << "[Prefill] Next token: " << next_token << " -> \"" << decoded << "\"\n";
  
  tokens.push_back(next_token);
  std::cout << "[Output] " << decoded;
  std::cout.flush();

  // ========== 13. Identify KV Cache Output Tensors ==========
  std::cout << "\n[KV Cache] Identifying output tensors...\n";
  
  // KV cache outputs:
  // V cache: output_aten_view_copy_default_* with dims [1, ar_len, head_dim]
  // K cache: output_aten_permute_copy_default_* with dims [1, head_dim, ar_len]
  struct KVCacheOutput {
    std::string name;
    void* buffer;
    bool is_v_cache; // true for V, false for K
    int layer;
    int head;
  };
  
  std::vector<KVCacheOutput> kv_outputs;
  
  int v_idx = 0, k_idx = 0;
  for (const auto& t : prefill_graph.outputs) {
    std::string n = t.name;
    
    // V cache: view_copy with [1, ar_len, head_dim]
    if (n.find("view_copy") != std::string::npos && 
        t.dims.size() == 3 && t.dims[1] == prefill_ar_len && t.dims[2] == head_dim) {
      auto it = prefill_alloc.bindings().find(t.name);
      if (it != prefill_alloc.bindings().end()) {
        int layer = v_idx / num_heads;
        int head = v_idx % num_heads;
        kv_outputs.push_back({t.name, it->second, true, layer, head});
        v_idx++;
      }
    }
    // K cache: permute_copy with [1, head_dim, ar_len]
    else if (n.find("permute_copy") != std::string::npos && 
             t.dims.size() == 3 && t.dims[1] == head_dim && t.dims[2] == prefill_ar_len) {
      auto it = prefill_alloc.bindings().find(t.name);
      if (it != prefill_alloc.bindings().end()) {
        int layer = k_idx / num_heads;
        int head = k_idx % num_heads;
        kv_outputs.push_back({t.name, it->second, false, layer, head});
        k_idx++;
      }
    }
  }
  
  std::cout << "[KV Cache] Found " << v_idx << " V cache + " << k_idx << " K cache outputs\n";
  std::cout << "[KV Cache] Total KV outputs: " << kv_outputs.size() << "\n";

  // ========== 14. Prefill 후 KV Cache Update (SMART_MASK 방식) ==========
  std::cout << "\n[KV Update] Copying prefill outputs to KVManager...\n";
  
  // Prefill 실행 후 상태:
  // - Input tokens: 예를 들어 12개 (BOS, format, "Hello", EOT 등)
  // - Prefill output: 32개 position의 logits (ar_len=32)
  // - KV cache output: 각 layer/head당 [1, 32, 64] (V) 또는 [1, 64, 32] (K)
  // 
  // Update 작업:
  // - KV cache output에서 실제로 의미있는 처음 n_update개만 KVManager에 복사
  // - n_update = tokens.size() - 1 = 12 - 1 = 11 (next_token은 아직 KV cache에 없음)
  // - 복사 위치: n_past = 0 (첫 prefill이므로)
  
  // ===== Prefill KV cache update 파라미터 =====
  // ExecutorchReader 분석:
  //   n_update = 1 + ((num_prompt_tokens - 1) % ar_len)
  // 
  // 예: num_prompt_tokens=1, ar_len=32
  //   n_update = 1 + ((1 - 1) % 32) = 1
  // 
  // 의미: Prefill 실행 후, output에서 의미있는 n_update개 토큰의 KV를 복사
  // - AR=32로 실행했지만, 실제 입력 토큰은 1개뿐
  // - Output의 [0..n_update-1] 위치만 유효한 데이터
  
  int32_t n_past = 0; // 첫 prefill이므로 KV cache는 비어있음
  int32_t num_prompt_tokens = tokens.size();  // "Hello" → 1
  int32_t n_update = 1 + ((num_prompt_tokens - 1) % prefill_ar_len);  // ✅ ExecutorchReader 방식
  
  // DEBUG: Prefill 첫 번째 KV output 검사 (Layer 0, Head 0)
  if (log_level >= 1) {
    bool logged_v = false, logged_k = false;
    for (const auto& kv_out : kv_outputs) {
      if (!logged_v && kv_out.is_v_cache && kv_out.layer == 0 && kv_out.head == 0) {
        uint8_t* data = reinterpret_cast<uint8_t*>(kv_out.buffer);
        int non_zero = 0;
        int total_bytes = prefill_ar_len * head_dim;  // [1, ar_len, head_dim]
        for (int i = 0; i < std::min(1000, total_bytes); ++i) {
          if (data[i] != 0) non_zero++;
        }
        std::cout << "\n[Debug] Prefill V cache output (L0H0):\n";
        std::cout << "  Shape: [1, " << prefill_ar_len << ", " << head_dim << "]\n";
        std::cout << "  First 10 bytes: [";
        for (int i = 0; i < 10; ++i) std::cout << (int)data[i] << (i<9 ? ", " : "]\n");
        std::cout << "  Non-zero in first 1000 bytes: " << non_zero << "/1000\n";
        logged_v = true;
      }
      if (!logged_k && !kv_out.is_v_cache && kv_out.layer == 0 && kv_out.head == 0) {
        uint8_t* data = reinterpret_cast<uint8_t*>(kv_out.buffer);
        int non_zero = 0;
        int total_bytes = head_dim * prefill_ar_len;  // [1, head_dim, ar_len]
        for (int i = 0; i < std::min(1000, total_bytes); ++i) {
          if (data[i] != 0) non_zero++;
        }
        std::cout << "[Debug] Prefill K cache output (L0H0):\n";
        std::cout << "  Shape: [1, " << head_dim << ", " << prefill_ar_len << "]\n";
        std::cout << "  First 10 bytes: [";
        for (int i = 0; i < 10; ++i) std::cout << (int)data[i] << (i<9 ? ", " : "]\n");
        std::cout << "  Non-zero in first 1000 bytes: " << non_zero << "/1000\n";
        logged_k = true;
      }
      if (logged_v && logged_k) break;
    }
  }
  
  // 모든 layer와 head에 대해 KV cache 업데이트
  for (const auto& kv_out : kv_outputs) {
    if (kv_out.is_v_cache) {
      // ===== V Cache Update =====
      // Output: [1, ar_len=32, head_dim=64] (sequential layout)
      // Input:  [1, max_cache_len=511, head_dim=64] (sequential layout)
      // 
      // SMART_MASK: 유효 데이터는 output의 끝에 있음!
      // Prefill output [1, 32, 64]: 유효 데이터는 position [32-n_update..31]
      //
      // 예: n_update=1, ar_len=32
      //   Output: [padding...][DATA][head_dim=64] (position 31에 유효 데이터)
      //   → src를 (ar_len - n_update) * head_dim offset해야 함!
      
      const auto& v_buf = kv_manager.get_v_cache(kv_out.layer, kv_out.head);
      
      // ✅ ExecutorchReader 방식: Output의 시작부터 n_update개 복사
      // SMART_MASK에서 output은 이미 올바른 위치에 데이터 배치됨
      // Output [1, 32, 64]: position [0..n_update-1]이 유효 데이터
      uint8_t* src = reinterpret_cast<uint8_t*>(kv_out.buffer);  // ✅ 시작부터!
      uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past * head_dim;
      
      // 연속 복사: n_update개 토큰 × head_dim bytes
      std::memcpy(dst, src, n_update * head_dim);
      
    } else {
      // ===== K Cache Update =====
      // Output: [1, head_dim=64, ar_len=32] (strided layout)
      // Input:  [1, head_dim=64, max_cache_len=511] (strided layout)
      // 
      // Strided이므로 dimension별로 복사:
      // output[dim][0..n_update-1] → input[dim][n_past..n_past+n_update-1]
      // 
      // Memory layout:
      // Output: [dim0: pos0,pos1,...,pos31][dim1: pos0,pos1,...,pos31]...[dim63]
      // Input:  [dim0: pos0,pos1,...,pos510][dim1: pos0,pos1,...,pos510]...[dim63]
      
      const auto& k_buf = kv_manager.get_k_cache(kv_out.layer, kv_out.head);
      
      // ===== SMART_MASK K cache update =====
      // ExecutorchReader 실제 동작 (로그 확인):
      //   write_ptr = k_cache.buffer + past_size (n_past offset)
      //   read_ptr = k_cache.output_buffer  ← 시작부터!
      //   copy_size = n_update * sizeof(T)
      //   for each dimension:
      //     memcpy(write_ptr, read_ptr, copy_size)
      //     write_ptr += iter_size (cache_len stride)
      //     read_ptr += out_size (ar_len stride)
      //
      // ✅ SMART_MASK에서 output은 이미 올바른 위치에 데이터 배치
      // Prefill output [1, 64, 32]: position [0..n_update-1]이 유효
      //
      // 예: n_update=18, ar_len=32
      //   Output: [dim0][DATA(18개), padding(14개)]
      //   read_ptr[0:3] = [130, 121, 38]  ← 시작부터 유효!
      
      uint8_t* src = reinterpret_cast<uint8_t*>(kv_out.buffer);  // ✅ 시작부터!
      uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past;  // n_past offset
      
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        // 각 dimension에서 n_update bytes 복사
        std::memcpy(dst, src, n_update);
        src += prefill_ar_len;       // Output: 다음 dimension (stride=32)
        dst += prefill_cache_len;    // Input: 다음 dimension (stride=480, rearrange 전!)
      }
    }
  }
  
  std::cout << "[KV Update] Copied " << n_update << " tokens to KV cache at position " << n_past << "\n";
  std::cout << "[KV Update] KV cache now contains positions [0.." << (n_past + n_update - 1) << "]\n";
  
  // DEBUG: Prefill KV update 후 메모리 상태 확인 (Layer 0, Head 0)
  if (log_level >= 1) {
    const auto& k_buf = kv_manager.get_k_cache(0, 0);
    uint8_t* k_data = reinterpret_cast<uint8_t*>(k_buf.input_buffer);
    int non_zero = 0;
    for (int i = 0; i < std::min((int)k_buf.input_bytes, 10000); ++i) {
      if (k_data[i] != 0) non_zero++;
    }
    std::cout << "[Debug] After Prefill KV update (L0H0):\n";
    std::cout << "  K cache buffer[0]=" << (int)k_data[0]
              << ", buffer[" << prefill_cache_len << "]=" << (int)k_data[prefill_cache_len]
              << ", buffer[" << (prefill_cache_len*2) << "]=" << (int)k_data[prefill_cache_len*2] << "\n";
    std::cout << "  Non-zero in first 10000 bytes: " << non_zero << "/10000\n";
    std::cout << "  Expected: n_update=" << n_update << " tokens copied to position " << n_past << "\n";
  }

  // ========== 14.5. Rearrange Cache for Decode ==========
  //
  // ExecutorchReader 방식:
  // - Prefill (AR=32, cache_len=480) → Decode (AR=1, cache_len=511)로 전환
  // - rearrange_cache(32, 1)를 호출하여 480→511로 메모리 재배치
  //
  // K cache:
  //   Before: [head_dim, 480] - [dim0][pos0..479][dim1][pos0..479]...[dim63]
  //   After:  [head_dim, 511] - [dim0][pos0..510][dim1][pos0..510]...[dim63]
  //   → 각 dimension을 뒤로 31 positions 이동 (backward memmove)
  //
  // V cache:
  //   Before: [480, head_dim] - sequential [pos0..479][head_dim]
  //   After:  [511, head_dim] - sequential [pos0..479][head_dim] + 31 empty
  //   → 이미 올바른 위치에 있으므로 no-op
  
  std::cout << "\n[Rearrange] Expanding KV cache: Prefill (AR=32, cache_len=480) → Decode (AR=1, cache_len=511)\n";
  kv_manager.rearrange_cache(prefill_ar_len, kv_ar_len);
  
  // ========== 14.6. Rearrange 후 메모리 검증 ==========
  if (log_level >= 1) {
    std::cout << "\n[Debug] Verifying K cache after rearrange (Layer 0, Head 0):\n";
    const auto& k_buf_after = kv_manager.get_k_cache(0, 0);
    uint8_t* k_data = reinterpret_cast<uint8_t*>(k_buf_after.input_buffer);
    
    // Check first dimension at positions [0..1] (should have data)
    int non_zero_pos0 = 0, non_zero_pos1 = 0;
    for (int i = 0; i < 5; ++i) {
      if (k_data[0 * kv_cache_len + i] != 0) non_zero_pos0++;
      if (k_data[1 * kv_cache_len + i] != 0) non_zero_pos1++;
    }
    std::cout << "  Dim 0: positions [0..4] have " << non_zero_pos0 << "/5 non-zero\n";
    std::cout << "  Dim 1: positions [0..4] have " << non_zero_pos1 << "/5 non-zero\n";
    
    // Total non-zero bytes in first 1000 bytes
    int total_non_zero = 0;
    for (int i = 0; i < std::min((int)k_buf_after.input_bytes, 1000); ++i) {
      if (k_data[i] != 0) total_non_zero++;
    }
    std::cout << "  Total: " << total_non_zero << "/1000 bytes non-zero\n";
  }

  // ========== 15. Decoding Loop ==========
  std::cout << "\n========== DECODING PHASE ==========\n";
  
  // ===== Decode 시작 position =====
  // Prefill 후 KV cache에 저장된 토큰 수
  // = n_update (prefill에서 복사한 유효 토큰 수)
  int32_t initial_tokens = n_update;  // ✅ Prefill에서 복사한 토큰 수
  
  for (int gen_idx = 0; gen_idx < max_gen - 1; ++gen_idx) {
    std::cout << "\n--- Decode Step " << (gen_idx + 1) << " ---\n";
    
    // Prepare single-token input for kv_forward
    std::vector<int32_t> decode_tokens = {next_token};
    int32_t n_past = initial_tokens + gen_idx; // Total tokens in KV cache
    
    auto get_kv_buffer = [&](const std::string& name) -> void* {
      auto& bindings = kv_alloc.bindings();
      auto it = bindings.find(name);
      return (it != bindings.end()) ? it->second : nullptr;
    };
    
    // Fill kv_forward inputs
    // Token input
    for (const auto& t : kv_graph.inputs) {
      std::string n = t.name;
      for (auto& c : n) c = (char)tolower(c);
      
      void* buf = get_kv_buffer(t.name);
      if (!buf) continue;
      
      // Token
      if (n.find("token") != std::string::npos && n.find("input") != std::string::npos) {
        std::memcpy(buf, &next_token, sizeof(int32_t));
        if (log_level >= 2 && gen_idx == 0) {
          std::cout << "[Debug] Token input: " << next_token << "\n";
        }
      }
      // Position
      else if (n.find("pos") != std::string::npos && t.data_type.find("INT_32") != std::string::npos) {
        std::memcpy(buf, &n_past, sizeof(int32_t));
        if (log_level >= 2 && gen_idx == 0) {
          std::cout << "[Debug] Position input: " << n_past << "\n";
        }
      }
      // ===== Attention mask (decode: [1, ar_len=1, context_len]) =====
      else if (n.find("atten_mask") != std::string::npos || n.find("attn_mask") != std::string::npos) {
        uint16_t* mask = reinterpret_cast<uint16_t*>(buf);
        std::memset(mask, 0, t.nbytes);  // Clear all (masked)
        
        // SMART_MASK 방식: Decode attention mask (ar_len=1)
        //
        // Decode 상태 (예: Step 1):
        // - KV cache: positions [0..n_past-1]에 이미 저장된 토큰들 (prefill + 이전 decode)
        // - 현재 토큰: position n_past에 생성 예정
        //
        // Attention 구조:
        // Row 0 (현재 토큰):
        //   [65535, 65535, ..., 65535, 0, 0, ..., 0, 65535]
        //    └─ [0..n_past-1] ─┘                    └─ context_len-1
        //    Past tokens (attend)                   Current token (attend)
        //
        // ExecutorchReader 참고:
        // - init_attention_mask: n_past가 0이면 past_ptr에 아무것도 채우지 않음
        // - new_ptr[i] = 65535: 항상 자기 자신에게 attend
        // - update_attention_mask: cur_ptr += n_past로 past 영역 채움
        
        // 1. Past tokens에 attend (KV cache에 저장된 모든 토큰)
        for (int32_t i = 0; i < n_past; ++i) {
          mask[i] = 65535;  // attend (positions [0..n_past-1])
        }
        
        // 2. 현재 토큰에 attend (context window 끝)
        // ar_len=1이므로 마지막 position (context_len - 1)
        mask[context_len - 1] = 65535;
        
        if (log_level >= 2 && gen_idx == 0) {
          std::cout << "[Debug] Attention mask: n_past=" << n_past << ", attend to [0.." << (n_past-1) << "] and [" << (context_len-1) << "]\n";
          int attend_count = 0;
          for (int i = 0; i < context_len; ++i) {
            if (mask[i] == 65535) attend_count++;
          }
          std::cout << "  Total positions attending: " << attend_count << "/" << context_len << "\n";
        }
      }
      // KV cache inputs: use KVManager buffers directly
      else if (n.find("_args_") != std::string::npos && t.dims.size() == 3) {
        // Map tensor name index to layer/head
        // Pattern: input_2_args_128_0 → index 2
        // First 256 inputs: 2~257
        // V cache (first 128): input_2~input_129 with dims [1, max_cache_len, head_dim]
        // K cache (next 128): input_130~input_257 with dims [1, head_dim, max_cache_len]
        
        // Extract index from tensor name
        size_t idx_start = t.name.find("input_") + 6;
        size_t idx_end = t.name.find('_', idx_start);
        int tensor_idx = std::stoi(t.name.substr(idx_start, idx_end - idx_start));
        
        if (t.dims[1] == kv_cache_len && t.dims[2] == head_dim) {
          // V cache input (Decode 그래프, cache_len=511)
          int v_offset = tensor_idx - 2; // input_2 is first V cache
          int layer = v_offset / num_heads;
          int head = v_offset % num_heads;
          
          if (layer >= 0 && layer < num_layers && head >= 0 && head < num_heads) {
            const auto& v_buf = kv_manager.get_v_cache(layer, head);
            std::memcpy(buf, v_buf.input_buffer, v_buf.input_bytes);
            
            // Debug: verify KV cache data integrity
            if (gen_idx == 0 && layer == 0 && head == 0) {
              // Check source (KVManager)
              uint8_t* src_check = reinterpret_cast<uint8_t*>(v_buf.input_buffer);
              int src_non_zero = 0;
              for (size_t i = 0; i < std::min(v_buf.input_bytes, (size_t)1000); ++i) {
                if (src_check[i] != 0) src_non_zero++;
              }
              
              // Check destination (kv_alloc)
              uint8_t* dst_check = reinterpret_cast<uint8_t*>(buf);
              int dst_non_zero = 0;
              for (size_t i = 0; i < std::min(v_buf.input_bytes, (size_t)1000); ++i) {
                if (dst_check[i] != 0) dst_non_zero++;
              }
              
              std::cout << "[Debug] V[0][0]: KVManager has " << src_non_zero 
                        << "/1000 non-zero, kv_alloc has " << dst_non_zero << "/1000 non-zero\n";
            }
          } else {
            std::memset(buf, 0, t.nbytes);
          }
        } else if (t.dims[1] == head_dim && t.dims[2] == kv_cache_len) {
          // K cache input (Decode 그래프, cache_len=511)
          int k_offset = tensor_idx - (2 + 128); // input_130 is first K cache
          int layer = k_offset / num_heads;
          int head = k_offset % num_heads;
          
          if (layer >= 0 && layer < num_layers && head >= 0 && head < num_heads) {
            const auto& k_buf = kv_manager.get_k_cache(layer, head);
            std::memcpy(buf, k_buf.input_buffer, k_buf.input_bytes);
            
            // Debug: verify K cache data integrity
            if (gen_idx == 0 && layer == 0 && head == 0 && log_level >= 2) {
              uint8_t* src_check = reinterpret_cast<uint8_t*>(k_buf.input_buffer);
              int src_non_zero = 0;
              for (size_t i = 0; i < std::min(k_buf.input_bytes, (size_t)1000); ++i) {
                if (src_check[i] != 0) src_non_zero++;
              }
              
              uint8_t* dst_check = reinterpret_cast<uint8_t*>(buf);
              int dst_non_zero = 0;
              for (size_t i = 0; i < std::min(k_buf.input_bytes, (size_t)1000); ++i) {
                if (dst_check[i] != 0) dst_non_zero++;
              }
              
              std::cout << "[Debug] K[0][0]: KVManager has " << src_non_zero 
                        << "/1000 non-zero, kv_alloc has " << dst_non_zero << "/1000 non-zero\n";
              
              // Check position 0 for each dimension (should have data)
              std::cout << "  First 3 dimensions at position 0: ";
              for (int d = 0; d < 3; ++d) {
                uint8_t val = src_check[d * kv_cache_len];
                std::cout << (int)val << " ";
              }
              std::cout << "\n";
            }
          } else {
            std::memset(buf, 0, t.nbytes);
          }
        } else {
          std::memset(buf, 0, t.nbytes);
        }
      }
    }
    
    // Build QNN tensors for kv_forward
    std::vector<Qnn_Tensor_t> kv_inputs, kv_outputs;
    std::vector<std::unique_ptr<QnnTensorHolder>> kv_holders;
    
    int kv_input_count = 0;
    for (const auto& t : kv_graph.inputs) {
      void* buf = get_kv_buffer(t.name);
      if (!buf) continue;
      
      auto h = std::make_unique<QnnTensorHolder>();
      if (h->init_from_json(t, buf, t.nbytes, true)) {
        kv_inputs.push_back(h->tensor());
        kv_holders.push_back(std::move(h));
        kv_input_count++;
      }
    }
    
    if (gen_idx == 0) {
      std::cout << "[Debug] kv_forward prepared " << kv_input_count << " input tensors\n";
    }
    
    for (const auto& t : kv_graph.outputs) {
      void* buf = get_kv_buffer(t.name);
      if (!buf) continue;
      
      auto h = std::make_unique<QnnTensorHolder>();
      if (h->init_from_json(t, buf, t.nbytes, false)) {
        kv_outputs.push_back(h->tensor());
        kv_holders.push_back(std::move(h));
      }
    }
    
    // Execute kv_forward
    if (!loader.execute_graph(0, "kv_forward", kv_inputs, kv_outputs)) {
      std::cerr << "[Decode] kv_forward execution failed at step " << gen_idx << "\n";
      break;
    }
    
    // Extract logits (single token output)
    const QnnJsonTensorDesc* kv_logits_desc = nullptr;
    for (const auto& t : kv_graph.outputs) {
      if (t.name.find("squeeze") != std::string::npos || 
          t.name.find("logit") != std::string::npos) {
        kv_logits_desc = &t;
        break;
      }
    }
    
    if (!kv_logits_desc) {
      std::cerr << "[Decode] Logits output not found in kv_forward\n";
      break;
    }
    
    void* kv_logits_buf = get_kv_buffer(kv_logits_desc->name);
    const uint16_t* kv_logits_u16 = reinterpret_cast<const uint16_t*>(kv_logits_buf);
    
    // Argmax (single token, so offset = 0)
    uint16_t max_val = kv_logits_u16[0];
    next_token = 0;
    for (int32_t i = 1; i < vocab_size; ++i) {
      if (kv_logits_u16[i] > max_val) {
        max_val = kv_logits_u16[i];
        next_token = i;
      }
    }
    
    decoded = tokenizer.decode({next_token});
    std::cout << "[Decode] Token " << (gen_idx + 1) << ": " << next_token << " -> \"" << decoded << "\"\n";
    
    tokens.push_back(next_token);
    std::cout << decoded;
    std::cout.flush();
    
    // ========== Decode 후 KV Cache Update (SMART_MASK 방식) ==========
    // 
    // Decode 실행 후 상태:
    // - Input: next_token (1개)
    // - Output: logits [1, vocab_size], KV cache [각 layer/head당 V: [1,1,64], K: [1,64,1]]
    //
    // Update 작업:
    // - n_past = initial_tokens + gen_idx
    //   예) Decode Step 1: n_past = 11 + 0 = 11 (prefill 11개 토큰 후 첫 decode)
    // - 현재 토큰의 KV를 position n_past에 저장
    //
    // ExecutorchReader의 update_cache() 참고:
    // - update_key(k_cache, n_past, n_update=1)
    // - update_value(v_cache, n_past, n_update=1)
    
    int32_t pos_for_update = n_past; // 저장할 position (현재 처리한 토큰의 위치)
    
    std::cout << "[KV Update] Updating position " << pos_for_update 
              << " (total tokens in cache after update: " << (pos_for_update + 1) << ")\n";
    
    int v_out_idx = 0, k_out_idx = 0;
    int v_updated = 0, k_updated = 0;
    
    for (const auto& t : kv_graph.outputs) {
      std::string n = t.name;
      
      // ===== V Cache Output: [1, 1, head_dim=64] =====
      // Sequential layout: 단일 토큰의 64 bytes
      if (n.find("view_copy") != std::string::npos && 
          t.dims.size() == 3 && t.dims[1] == 1 && t.dims[2] == head_dim) {
        void* out_buf = get_kv_buffer(t.name);
        if (out_buf) {
          int layer = v_out_idx / num_heads;
          int head = v_out_idx % num_heads;
          
          if (layer >= 0 && layer < num_layers && head >= 0 && head < num_heads) {
            const auto& v_buf = kv_manager.get_v_cache(layer, head);
            
            // V cache update (sequential):
            // Output: [0..63] (64 bytes, sequential)
            // Input:  [...][pos*64 .. (pos+1)*64-1][...] (pos 위치에 삽입)
            uint8_t* src = reinterpret_cast<uint8_t*>(out_buf);
            uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + pos_for_update * head_dim;
            std::memcpy(dst, src, head_dim);  // 64 bytes 복사
            v_updated++;
          }
          v_out_idx++;
        }
      }
      // ===== K Cache Output: [1, head_dim=64, 1] =====
      // 단일 토큰이지만 strided layout
      else if (n.find("permute_copy") != std::string::npos && 
               t.dims.size() == 3 && t.dims[1] == head_dim && t.dims[2] == 1) {
        void* out_buf = get_kv_buffer(t.name);
        if (out_buf) {
          int layer = k_out_idx / num_heads;
          int head = k_out_idx % num_heads;
          
          if (layer >= 0 && layer < num_layers && head >= 0 && head < num_heads) {
            const auto& k_buf = kv_manager.get_k_cache(layer, head);
            
            // K cache update (strided):
            // Output: [1, head_dim=64, 1] → 메모리상 sequential [64 bytes]
            //   [dim0_val, dim1_val, ..., dim63_val]
            // 
            // Input: [1, head_dim=64, kv_cache_len=511] → strided layout (rearrange 후)
            //   [dim0: pos0, pos1, ..., pos510]
            //   [dim1: pos0, pos1, ..., pos510]
            //   ...
            //   [dim63: pos0, pos1, ..., pos510]
            //
            // Update: 각 dimension의 position pos_for_update에 값 삽입
            //   input[dim][pos_for_update] = output[dim]
            
            uint8_t* src = reinterpret_cast<uint8_t*>(out_buf);  // Sequential 64 bytes
            
            for (int32_t dim = 0; dim < head_dim; ++dim) {
              // Input의 [dim][pos_for_update] 위치 계산
              // = base + dim * stride + pos_for_update
              // ✅ stride=kv_cache_len (511, rearrange 후)
              uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) 
                             + dim * kv_cache_len + pos_for_update;
              dst[0] = src[dim];  // src는 sequential하므로 src[dim]
            }
            k_updated++;
          }
          k_out_idx++;
        }
      }
    }
    
    std::cout << "[KV Update] Updated " << v_updated << " V caches, " << k_updated << " K caches\n";
  }
  
  std::cout << "\n\n[DONE] Generated " << (max_gen) << " tokens\n";
  // Decode without special tokens
  std::cout << "[Full Output] " << tokenizer.decode(tokens, false) << "\n";

  // Cleanup
  prefill_alloc.release();
  kv_alloc.release();
  munmap(ctx_addr, st.st_size);
  close(fd);
  tokenizer.shutdown();

  return 0;
}

