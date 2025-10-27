#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace llm_test {

/**
 * @brief LLM KV Cache Manager for SMART_MASK mode
 * 
 * Manages KV cache memory allocation and updates for dual-graph LLM inference.
 * Compatible with ExecuTorch's KVManager design.
 * 
 * Key Responsibilities:
 * - Allocate persistent KV cache buffers (input + output)
 * - Update cache: copy output → input after each step
 * - Update attention mask for each iteration
 * - Provide memory pointers for graph binding
 */
class LLMKVCacheManager {
public:
  struct Metadata {
    int32_t context_len;      // Total context length (e.g., 256)
    int32_t head_dim;         // Attention head dimension (e.g., 64)
    int32_t max_ar_len;       // Max autoregressive length (e.g., 32 for prefill)
    int32_t max_cache_len;    // Max cache length (context_len - min_ar_len, e.g., 224)
    int32_t num_heads;        // Number of attention heads per layer (e.g., 8)
    int32_t num_layers;       // Number of transformer layers (e.g., 16)
  };

  /**
   * @brief KV cache buffer pair for one head
   */
  struct KVCacheBuffer {
    void* input_buffer;   // Persistent storage [head_dim, max_cache_len]
    void* output_buffer;  // Temporary storage [head_dim, max_ar_len]
    size_t input_bytes;
    size_t output_bytes;
  };

  LLMKVCacheManager(const Metadata& metadata);
  ~LLMKVCacheManager();

  /**
   * @brief Allocate all KV cache memory
   * @return true if successful
   */
  bool allocate();

  /**
   * @brief Update KV cache: copy output → input
   * @param n_past Number of past tokens already in cache
   * @param n_update Number of new tokens to update
   */
  void update_cache(int32_t n_past, int32_t n_update);

  /**
   * @brief Initialize attention mask for prefill/decode
   * @param attention_mask Output buffer [ar_len, context_len] as uint16_t*
   * @param ar_len Current autoregressive length
   * @param n_past Number of past tokens
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Update attention mask after cache update
   * @param attention_mask Buffer [ar_len, context_len] as uint16_t*
   * @param ar_len Current autoregressive length
   * @param n_past Number of past tokens before update
   * @param n_update Number of newly added tokens
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);

  /**
   * @brief Get K cache buffer for a specific layer and head
   */
  const KVCacheBuffer& get_k_cache(int32_t layer, int32_t head) const {
    return k_cache_[layer][head];
  }

  /**
   * @brief Get V cache buffer for a specific layer and head
   */
  const KVCacheBuffer& get_v_cache(int32_t layer, int32_t head) const {
    return v_cache_[layer][head];
  }

  /**
   * @brief Get total allocated memory size
   */
  size_t total_cache_size() const { return total_cache_size_; }

  /**
   * @brief Get metadata
   */
  const Metadata& metadata() const { return metadata_; }

private:
  Metadata metadata_;
  size_t total_cache_size_;

  // KV cache storage: [num_layers][num_heads]
  std::vector<std::vector<KVCacheBuffer>> k_cache_;
  std::vector<std::vector<KVCacheBuffer>> v_cache_;

  // Helper functions
  void update_key_cache(
      const KVCacheBuffer& cache,
      int32_t n_past,
      int32_t n_update);
  
  void update_value_cache(
      const KVCacheBuffer& cache,
      int32_t n_past,
      int32_t n_update);
};

} // namespace llm_test

