#pragma once

#include "qnn_loader.h"
#include "qnn_qnnjson.h"
#include "qnn_tensor_util.h"
#include "io_alloc.h"
#include "llm_kv_cache_manager.h"
#include "llm_kv_cache_mapper.h"
#include "llm_stats.h"
#include "llm_output_processor.h"
#include "tokenizer_llama.h"

#include <string>
#include <vector>
#include <memory>

namespace llm_test {

/**
 * @brief Configuration for LLM Decode Runner
 */
struct LLMDecodeConfig {
  std::string ctx_dir;          // QNN context directory
  std::string backend_so;       // QNN backend library path
  std::string system_so;        // QNN system library path (optional)
  std::string tokenizer_path;   // Tokenizer model path
  int max_gen_tokens = 100;     // Maximum tokens to generate
  int log_level = 0;            // 0=quiet, 1=info, 2=debug
};

/**
 * @brief High-level API for LLM Prefill + Decode execution
 * 
 * Manages:
 * - QNN context loading and graph execution
 * - KV cache allocation and mapping
 * - Prefill → Decode transition (rearrange_cache)
 * - Token generation loop
 */
class LLMDecodeRunner {
 public:
  explicit LLMDecodeRunner(const LLMDecodeConfig& config);
  ~LLMDecodeRunner();
  
  /**
   * @brief Initialize QNN backend and load graphs
   * @return true on success
   */
  bool initialize();
  
  /**
   * @brief Run prefill + decode to generate text
   * @param prompt Input prompt string
   * @param output_text Generated text (output parameter)
   * @return true on success
   */
  bool generate(const std::string& prompt, std::string& output_text);
  
  /**
   * @brief Get last error message
   */
  const std::string& get_error() const { return error_msg_; }
  
  /**
   * @brief Get performance statistics
   */
  const LLMStats& get_stats() const { return stats_; }
  
 private:
  // Configuration
  LLMDecodeConfig config_;
  std::string error_msg_;
  
  // QNN components
  std::unique_ptr<QnnLoader> loader_;
  std::map<std::string, QnnJsonGraphDesc> graphs_;
  QnnJsonGraphDesc* prefill_graph_;
  QnnJsonGraphDesc* kv_graph_;
  
  // Model metadata
  int context_len_;
  int num_layers_;
  int num_heads_;
  int head_dim_;
  int prefill_ar_len_;
  int kv_ar_len_;
  int prefill_cache_len_;
  int kv_cache_len_;
  
  // KV cache
  std::unique_ptr<LLMKVCacheManager> kv_manager_;
  std::vector<KVCacheTensorInfo> prefill_kv_mapping_;
  std::vector<KVCacheTensorInfo> kv_kv_mapping_;
  std::map<std::string, void*> prefill_kv_override_;
  std::map<std::string, void*> kv_kv_override_;
  
  // I/O allocators
  std::unique_ptr<QNNIOAllocator> prefill_alloc_;
  std::unique_ptr<QNNIOAllocator> kv_alloc_;
  
  // Pre-built QNN tensors (reused across executions)
  std::vector<std::unique_ptr<QnnTensorHolder>> prefill_input_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> prefill_output_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> kv_input_holders_;
  std::vector<std::unique_ptr<QnnTensorHolder>> kv_output_holders_;
  
  // Tokenizer
  std::unique_ptr<LlamaTokenizer> tokenizer_;
  
  // Performance statistics
  LLMStats stats_;
  
  // Helper methods
  bool load_graphs();
  bool extract_metadata();
  bool setup_kv_cache();
  bool setup_io_allocators();
  
  bool run_prefill(const std::vector<int32_t>& tokens, 
                   int32_t& next_token,
                   int32_t& n_update);
  
  bool run_decode_step(int32_t token_in,
                       int32_t n_past,
                       int32_t& token_out);
};

} // namespace llm_test
