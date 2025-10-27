#include "llm_kv_cache_manager.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace llm_test {

LLMKVCacheManager::LLMKVCacheManager(const Metadata& metadata)
    : metadata_(metadata), total_cache_size_(0) {
  // Resize storage
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);
  
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    k_cache_[layer].resize(metadata_.num_heads);
    v_cache_[layer].resize(metadata_.num_heads);
  }

  // Calculate total memory requirement (SMART_MASK mode)
  // Each cache: input_buffer + output_buffer
  size_t k_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t k_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  size_t v_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t v_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  
  size_t per_head = k_in_bytes + k_out_bytes + v_in_bytes + v_out_bytes;
  total_cache_size_ = per_head * metadata_.num_layers * metadata_.num_heads;
  
  std::cout << "[LLMKVCacheManager] Metadata:\n"
            << "  context_len: " << metadata_.context_len << "\n"
            << "  head_dim: " << metadata_.head_dim << "\n"
            << "  max_ar_len: " << metadata_.max_ar_len << "\n"
            << "  max_cache_len: " << metadata_.max_cache_len << "\n"
            << "  num_heads: " << metadata_.num_heads << "\n"
            << "  num_layers: " << metadata_.num_layers << "\n"
            << "  Total cache size: " << (total_cache_size_ / 1024.0 / 1024.0) << " MiB\n";
}

LLMKVCacheManager::~LLMKVCacheManager() {
  // Free all allocated memory
  for (auto& layer_k : k_cache_) {
    for (auto& head_k : layer_k) {
      if (head_k.input_buffer) free(head_k.input_buffer);
      if (head_k.output_buffer) free(head_k.output_buffer);
    }
  }
  for (auto& layer_v : v_cache_) {
    for (auto& head_v : layer_v) {
      if (head_v.input_buffer) free(head_v.input_buffer);
      if (head_v.output_buffer) free(head_v.output_buffer);
    }
  }
}

bool LLMKVCacheManager::allocate() {
  size_t k_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t k_out_bytes = metadata_.head_dim * metadata_.max_ar_len;
  size_t v_in_bytes = metadata_.head_dim * metadata_.max_cache_len;
  size_t v_out_bytes = metadata_.head_dim * metadata_.max_ar_len;

  std::cout << "[LLMKVCacheManager] Allocating memory...\n";
  
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int32_t head = 0; head < metadata_.num_heads; ++head) {
      // Allocate K cache
      k_cache_[layer][head].input_buffer = malloc(k_in_bytes);
      k_cache_[layer][head].output_buffer = malloc(k_out_bytes);
      k_cache_[layer][head].input_bytes = k_in_bytes;
      k_cache_[layer][head].output_bytes = k_out_bytes;
      
      if (!k_cache_[layer][head].input_buffer || !k_cache_[layer][head].output_buffer) {
        std::cerr << "[LLMKVCacheManager] Failed to allocate K cache for layer " 
                  << layer << ", head " << head << "\n";
        return false;
      }
      
      // Initialize to zero
      std::memset(k_cache_[layer][head].input_buffer, 0, k_in_bytes);
      std::memset(k_cache_[layer][head].output_buffer, 0, k_out_bytes);
      
      // Allocate V cache
      v_cache_[layer][head].input_buffer = malloc(v_in_bytes);
      v_cache_[layer][head].output_buffer = malloc(v_out_bytes);
      v_cache_[layer][head].input_bytes = v_in_bytes;
      v_cache_[layer][head].output_bytes = v_out_bytes;
      
      if (!v_cache_[layer][head].input_buffer || !v_cache_[layer][head].output_buffer) {
        std::cerr << "[LLMKVCacheManager] Failed to allocate V cache for layer " 
                  << layer << ", head " << head << "\n";
        return false;
      }
      
      // Initialize to zero
      std::memset(v_cache_[layer][head].input_buffer, 0, v_in_bytes);
      std::memset(v_cache_[layer][head].output_buffer, 0, v_out_bytes);
    }
  }
  
  std::cout << "[LLMKVCacheManager] Allocation complete: " 
            << (total_cache_size_ / 1024.0 / 1024.0) << " MiB\n";
  return true;
}

void LLMKVCacheManager::update_key_cache(
    const KVCacheBuffer& cache,
    int32_t n_past,
    int32_t n_update) {
  // Key cache layout (SMART_MASK):
  // Input:  [head_dim, max_cache_len]
  // Output: [head_dim, max_ar_len]
  //
  // Update: For each dimension, copy output[dim][0:n_update] → input[dim][n_past:n_past+n_update]
  
  uint8_t* write_ptr = reinterpret_cast<uint8_t*>(cache.input_buffer) + n_past;
  uint8_t* read_ptr = reinterpret_cast<uint8_t*>(cache.output_buffer);
  
  for (int32_t dim = 0; dim < metadata_.head_dim; ++dim) {
    std::memcpy(write_ptr, read_ptr, n_update);
    write_ptr += metadata_.max_cache_len;
    read_ptr += metadata_.max_ar_len;
  }
}

void LLMKVCacheManager::update_value_cache(
    const KVCacheBuffer& cache,
    int32_t n_past,
    int32_t n_update) {
  // Value cache layout (SMART_MASK):
  // Input:  [max_cache_len, head_dim] - sequential
  // Output: [max_ar_len, head_dim] - sequential
  //
  // Update: copy output[0:n_update*head_dim] → input[n_past*head_dim:(n_past+n_update)*head_dim]
  
  uint8_t* write_ptr = reinterpret_cast<uint8_t*>(cache.input_buffer) + n_past * metadata_.head_dim;
  uint8_t* read_ptr = reinterpret_cast<uint8_t*>(cache.output_buffer);
  
  std::memcpy(write_ptr, read_ptr, n_update * metadata_.head_dim);
}

void LLMKVCacheManager::update_cache(int32_t n_past, int32_t n_update) {
  for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int32_t head = 0; head < metadata_.num_heads; ++head) {
      update_key_cache(k_cache_[layer][head], n_past, n_update);
      update_value_cache(v_cache_[layer][head], n_past, n_update);
    }
  }
}

void LLMKVCacheManager::init_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past) {
  // SMART_MASK attention mask initialization
  // Shape: [ar_len, context_len]
  // Values: 0 = mask (don't attend), 65535 = attend
  //
  // Pattern: Causal mask at the END of context window
  // For prefill (ar_len > 1):
  //   Row i attends to [context_len - ar_len, context_len - ar_len + i]
  // For decode (ar_len = 1):
  //   Row 0 attends to [0, n_past]
  
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  
  // Clear all to 0 (mask)
  std::memset(attention_mask, 0, ar_len * metadata_.context_len * sizeof(uint16_t));
  
  for (int32_t i = 0; i < ar_len; ++i) {
    uint16_t* row = attention_mask + i * metadata_.context_len;
    
    // Attend to all past tokens (before this batch)
    for (int32_t j = 0; j < n_past; ++j) {
      row[j] = pos_val;
    }
    
    // Attend to tokens in current batch (causal)
    // New tokens are placed at context window end: [context_len - ar_len, context_len)
    int32_t new_token_start = metadata_.context_len - ar_len;
    for (int32_t j = 0; j <= i; ++j) {
      row[new_token_start + j] = pos_val;
    }
  }
}

void LLMKVCacheManager::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  // SMART_MASK attention mask update
  // After cache update, newly added tokens should be attended to
  //
  // Update pattern: For each row, fill [n_past, n_past + n_update) with 65535
  
  uint16_t pos_val = 65535;
  
  for (int32_t i = 0; i < ar_len; ++i) {
    uint16_t* row = attention_mask + i * metadata_.context_len;
    std::fill_n(row + n_past, n_update, pos_val);
  }
}

} // namespace llm_test

