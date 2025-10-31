# LLM QNN Test - Modularized Architecture

## 📁 Project Structure

```
llm_test/
├── include/              # Public headers
│   ├── qnn_loader.h     # QNN backend and context loading
│   ├── qnn_qnnjson.h    # JSON graph description parser
│   ├── io_alloc.h       # I/O buffer allocator
│   ├── qnn_tensor_util.h           # QNN tensor utilities
│   ├── tokenizer_llama.h           # Llama tokenizer wrapper
│   ├── llm_input_preparer.h        # Input tensor preparation
│   ├── llm_output_processor.h      # Output tensor processing
│   ├── llm_kv_cache_manager.h      # KV cache memory management
│   ├── llm_kv_cache_mapper.h       # ✨ KV cache tensor mapping
│   └── llm_decode_runner.h         # ✨ High-level prefill+decode API
├── src/                  # Implementation
│   ├── qnn_loader.cpp
│   ├── qnn_qnnjson.cpp
│   ├── io_alloc.cpp
│   ├── qnn_tensor_util.cpp
│   ├── tokenizer_llama.cpp
│   ├── llm_input_preparer.cpp
│   ├── llm_output_processor.cpp
│   ├── llm_kv_cache_manager.cpp
│   ├── llm_kv_cache_mapper.cpp     # ✨ NEW
│   └── llm_decode_runner.cpp       # ✨ NEW
└── apps/                 # Applications
    ├── qnn_llm_generate.cpp        # ✨ NEW: Simple generation API
    ├── qnn_decode_main.cpp         # Original decode implementation
    └── ...
```

## 🔧 Core Modules

### 1️⃣ **LLMKVCacheMapper** (`llm_kv_cache_mapper.h/cpp`)

**Purpose**: Maps QNN JSON KV cache tensors to KVManager buffers

**Key Functions**:
```cpp
// Build KV cache mapping from JSON graph
std::vector<KVCacheTensorInfo> build_mapping(
    const QnnJsonGraphDesc& graph,
    int num_heads,
    int head_dim);

// Create buffer override map (tensor_name → buffer pointer)
std::map<std::string, void*> create_buffer_override(
    const std::vector<KVCacheTensorInfo>& mapping,
    LLMKVCacheManager& kv_manager);
```

**Pattern Detected** (from Executorch analysis):
```
Input Order:
  input_2~9:   V cache L0 H0~7 [1, cache_len, 64]
  input_10~17: K cache L0 H0~7 [1, 64, cache_len]
  input_18:    attention_mask
  input_19~26: V cache L1 H0~7
  ...

Pattern: Each layer has V 8개 → K 8개
```

### 2️⃣ **LLMDecodeRunner** (`llm_decode_runner.h/cpp`)

**Purpose**: High-level API for LLM text generation (Prefill + Decode)

**Configuration**:
```cpp
struct LLMDecodeConfig {
  std::string ctx_dir;          // QNN context directory
  std::string backend_so;       // QNN backend library
  std::string system_so;        // QNN system library (optional)
  std::string tokenizer_path;   // Tokenizer model
  int max_gen_tokens = 100;     // Max tokens to generate
  int log_level = 0;            // 0=quiet, 1=info, 2=debug
};
```

**Usage**:
```cpp
LLMDecodeRunner runner(config);
if (!runner.initialize()) {
  std::cerr << "Error: " << runner.get_error() << "\n";
  return 1;
}

std::string output;
if (!runner.generate("The capital of France is", output)) {
  std::cerr << "Error: " << runner.get_error() << "\n";
  return 1;
}

std::cout << output << "\n";
```

**Internal Flow**:
1. Load QNN backend and context binaries
2. Parse JSON graph descriptions
3. Extract model metadata (context_len, num_layers, num_heads, etc.)
4. Allocate and map KV cache (zero-copy shared memory)
5. Setup I/O allocators
6. Load tokenizer
7. Execute:
   - Tokenize prompt
   - Run prefill
   - Update KV cache from prefill outputs
   - Rearrange cache (480 → 511)
   - Run decode loop
   - Update KV cache from decode outputs
   - Decode tokens and append

### 3️⃣ **LLMKVCacheManager** (`llm_kv_cache_manager.h/cpp`)

**Purpose**: Manages KV cache memory allocation and rearrangement

**Key Features**:
- **Zero-copy shared memory**: Input tensors directly point to KV cache buffers
- **Rearrange support**: Expands cache from prefill size (480) to decode size (511)
- **Layout**: 
  - V cache: `[cache_len, head_dim]` (contiguous)
  - K cache: `[head_dim, cache_len]` (strided)

**API**:
```cpp
struct Metadata {
  int context_len;        // 512
  int head_dim;           // 64
  int max_ar_len;         // 32 (prefill AR)
  int max_cache_len;      // 511 (decode cache)
  int num_heads;          // 8
  int num_layers;         // 16
};

LLMKVCacheManager manager(metadata);
manager.allocate();

const auto& k_buf = manager.get_k_cache(layer, head);
const auto& v_buf = manager.get_v_cache(layer, head);

// Rearrange: 480 → 511
manager.rearrange_cache(prefill_ar_len, kv_ar_len);
```

## 🚀 Build & Run

### Build

```bash
cd /home/chokwans99/llm_test

# Clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Modularized Application

```bash
./build/qnn_llm_generate \
  --ctx_dir models/llama_qnn_1b \
  --tokenizer models/llama_qnn_1b/tokenizer.model \
  --prompt "The capital of France is" \
  --backend_so /path/to/libQnnHtp.so \
  --max_gen 50 \
  --log_level 1
```

**Arguments**:
- `--ctx_dir`: QNN context directory (contains `forward_0.bin` and `forward_0_json.json`)
- `--tokenizer`: Tokenizer model path
- `--prompt`: Input prompt string
- `--backend_so`: QNN backend library (default: `libQnnHtp.so`)
- `--system_so`: QNN system library (optional)
- `--max_gen`: Maximum tokens to generate (default: 100)
- `--log_level`: 0=quiet, 1=info, 2=debug (default: 1)

## 📊 Architecture Improvements

### Before (qnn_decode_main.cpp)
- ❌ 1000+ lines of monolithic code
- ❌ Hardcoded KV cache mapping logic
- ❌ Manual memcpy for every decode step
- ❌ Difficult to maintain and extend

### After (Modularized)
- ✅ **LLMKVCacheMapper**: Automatic tensor → layer/head mapping
- ✅ **LLMDecodeRunner**: Clean high-level API
- ✅ **Zero-copy KV cache**: Direct shared memory access
- ✅ **Reusable modules**: Easy to extend and test
- ✅ **90 lines main app** vs 1000+ lines

## 🔍 Key Insights from Executorch Analysis

### KV Cache Input Order
ExecutorchReader logs revealed:
```
prefill_forward:
  [0] input_0_tokens_0
  [1] input_1_input_pos_0
  [2-9]   V cache L0 H0~7 [1,480,64]
  [10-17] K cache L0 H0~7 [1,64,480]
  [18]    attention_mask
  [19-26] V cache L1 H0~7
  [27-34] K cache L1 H0~7
  ... (repeat for all 16 layers)
```

### MethodMeta vs Context Binary
- **MethodMeta** (Executorch internal): K all → V all
- **Context Binary** (JSON): V/K interleaved per layer
- **Solution**: Use JSON order directly (context binary order)

### Zero-Copy Implementation
```cpp
// Old: Manual copy every time
std::memcpy(qnn_buffer, kv_cache_buffer, size);

// New: Direct pointer (zero-copy)
auto override_map = LLMKVCacheMapper::create_buffer_override(mapping, kv_manager);
// QNN inputs now directly point to KV cache buffers!
```

## 📝 TODO

- [ ] Add proper error handling for edge cases
- [ ] Support dynamic batch size
- [ ] Add multi-graph support (beyond prefill/decode)
- [ ] Performance profiling and optimization
- [ ] Unit tests for each module
- [ ] Support more model architectures

## 🎯 Next Steps

1. **Test with different models**: Llama 3B, 8B
2. **Add streaming support**: Token-by-token output
3. **Benchmark**: Compare with Executorch performance
4. **Android deployment**: Test on mobile devices
5. **Documentation**: API reference and examples
