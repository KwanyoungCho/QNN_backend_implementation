# Multi-Context (Sharding) 사용 가이드

## 개요

Multi-context 모드는 큰 모델을 여러 개의 shard로 나누어 실행하는 기능입니다. 각 shard는 독립적인 QNN context binary로 저장되며, 순차적으로 실행됩니다.

## Shard 구조 (16 layers, 8 shards)

### Shard 0 (forward_0.bin)
- **레이어**: Layer 0-1 (2 layers)
- **입력 (35개)**:
  - 32개: KV cache input (2 layers × 2 directions × 8 heads)
  - 1개: Token input
  - 1개: Position input
  - 1개: Attention mask
- **출력 (35개)**:
  - 32개: KV cache output
  - 1개: Hidden state
  - 2개: ROPE (cos, sin)

### Shard 1-7 (forward_1.bin ~ forward_7.bin)
- **레이어**: Shard N → Layer 2N ~ 2N+1 (각 2 layers)
- **입력 (36개)**:
  - 32개: KV cache input
  - 1개: Hidden state (from previous shard)
  - 2개: ROPE (cos, sin) (shared from Shard 0)
  - 1개: Attention mask
- **출력 (33개)**:
  - 32개: KV cache output
  - 1개: Hidden state

## Shared Data

### 1. KV Cache
- **관리**: Layer-wise, LLMKVCacheManager에서 전체 16 layers 관리
- **입력/출력**: 각 shard가 담당하는 layer의 KV cache만 처리
- **예시**: Shard 0 → Layer 0-1, Shard 1 → Layer 2-3, ...

### 2. ROPE (Rotary Position Embedding)
- **생성**: Shard 0의 prefill 단계에서 출력
- **공유**: 모든 shard (1-7)가 동일한 ROPE 버퍼 참조
- **크기**: `[context_len, head_dim/2]`

### 3. Hidden State
- **체이닝**: Shard N의 출력 → Shard N+1의 입력
- **업데이트**: 각 shard 실행 후 shared buffer 업데이트
- **크기**: `[1, hidden_dim]` (hidden_dim = num_heads × head_dim)

### 4. Attention Mask
- **관리**: Shared buffer에 저장
- **브로드캐스트**: 모든 shard가 동일한 attention mask 참조
- **업데이트**: Decode 단계마다 업데이트 (past tokens + current token)

## 사용 방법

### 1. 디렉토리 구조
```
ctx_dir/
├── forward_0.bin         # Shard 0 context binary
├── forward_0_json.json   # Shard 0 metadata
├── forward_1.bin         # Shard 1 context binary
├── forward_1_json.json   # Shard 1 metadata
├── ...
├── forward_7.bin         # Shard 7 context binary
├── forward_7_json.json   # Shard 7 metadata
└── tokenizer.model       # Tokenizer
```

### 2. 코드 예시

```cpp
#include "llm_decode_runner.h"

using namespace llm_test;

int main() {
  LLMDecodeConfig config;
  config.ctx_dir = "/path/to/ctx_dir";
  config.tokenizer_path = "/path/to/tokenizer.model";
  config.backend_so = "libQnnHtp.so";
  config.max_gen_tokens = 100;
  config.log_level = 1;
  
  // Enable multi-context mode
  config.use_multi_context = true;
  config.num_shards = 8;  // 16 layers ÷ 2 layers/shard = 8 shards
  
  LLMDecodeRunner runner(config);
  
  if (!runner.initialize()) {
    std::cerr << "Error: " << runner.get_error() << "\n";
    return 1;
  }
  
  std::string output;
  if (!runner.generate("Hello, how are you?", output)) {
    std::cerr << "Error: " << runner.get_error() << "\n";
    return 1;
  }
  
  std::cout << "Output: " << output << "\n";
  return 0;
}
```

### 3. 실행 (Android)

```bash
# Push binaries
adb push build-android/qnn_llm_generate /data/local/tmp/
adb push ctx_dir/ /data/local/tmp/ctx_dir/

# Run
adb shell "cd /data/local/tmp && \
  export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && \
  ./qnn_llm_generate \
    --ctx_dir ctx_dir \
    --tokenizer ctx_dir/tokenizer.model \
    --prompt 'Hello, how are you?' \
    --max_gen 50 \
    --log_level 1"
```

## 실행 흐름

### Prefill 단계
1. **Shard 0**:
   - Token, position, attention mask 입력
   - Layer 0-1 실행
   - Hidden state, ROPE (cos, sin) 출력
   - KV cache 업데이트 (Layer 0-1)

2. **Shard 1-7**:
   - Hidden state (from prev shard), ROPE (from Shard 0), attention mask 입력
   - Layer 2N ~ 2N+1 실행
   - Hidden state 출력
   - KV cache 업데이트 (해당 layer)

3. **After all shards**:
   - KV cache rearrange (480 → 511)
   - Extract next token from final shard logits

### Decode 단계 (각 step)
1. **Attention mask 업데이트**:
   - Past tokens attend
   - Current token (at context_len-1) attend

2. **Shard 0-7 순차 실행**:
   - 각 shard가 담당하는 layer의 KV cache 업데이트
   - Hidden state 체이닝

3. **Token 생성**:
   - Final shard의 logits에서 next token 추출
   - Tokenizer로 디코딩

## 메모리 관리

### KV Cache
- **전체 할당**: 16 layers × 8 heads × 2 directions × cache_len × head_dim
- **Shard별 접근**: 각 shard는 담당 layer의 KV cache만 접근
- **예시 (Layer 0-1 in Shard 0)**:
  ```cpp
  for (int layer = 0; layer < 2; ++layer) {
    for (int head = 0; head < 8; ++head) {
      auto k_buf = kv_manager_->get_k_cache(layer, head);
      auto v_buf = kv_manager_->get_v_cache(layer, head);
      // Bind to shard 0's KV cache inputs
    }
  }
  ```

### Shared Buffers
- **Hidden state**: `hidden_dim × sizeof(uint16_t)` (typically 512 × 2 = 1KB)
- **ROPE cos/sin**: `2 × context_len × head_dim × sizeof(uint16_t)` (2 × 512 × 64 × 2 = 128KB)
- **Attention mask**: `ar_len × context_len × sizeof(uint16_t)` (32 × 512 × 2 = 32KB)

## 디버깅

### 로그 레벨
- `log_level = 0`: Quiet (출력만)
- `log_level = 1`: Info (주요 단계 로그)
- `log_level = 2`: Debug (상세 로그, shard별 입출력)

### 확인 사항
1. **Context binary 파일 존재**: `forward_0.bin ~ forward_7.bin`
2. **JSON 메타데이터 존재**: `forward_0_json.json ~ forward_7_json.json`
3. **Shard별 입출력 크기**:
   - Shard 0: 35 inputs, 35 outputs
   - Shard 1-7: 36 inputs, 33 outputs
4. **KV cache 크기**: 각 shard가 2 layers를 담당하므로 32개 KV cache tensors
5. **ROPE 공유**: Shard 1-7이 Shard 0의 ROPE를 사용하는지 확인

## 성능 최적화

1. **Zero-copy KV cache**: QNN 입력이 KV cache 버퍼를 직접 참조
2. **Shared buffer 재사용**: Hidden state, ROPE, attention mask를 여러 shard가 공유
3. **Layer-wise KV cache 관리**: 불필요한 메모리 복사 최소화

## 주의사항

1. **Shard 순서**: 반드시 0부터 7까지 순차적으로 실행
2. **Hidden state 체이닝**: 각 shard의 출력이 다음 shard의 입력으로 전달되어야 함
3. **ROPE 공유**: Shard 0에서 생성된 ROPE를 모든 shard가 사용
4. **KV cache 레이어 매핑**: 각 shard가 담당하는 layer 범위를 정확히 지정
5. **Attention mask 업데이트**: Decode 단계마다 올바르게 업데이트

## TODO (구현 필요)

현재 `llm_decode_runner_multi_context.cpp`의 다음 부분이 TODO로 남아있습니다:

1. **`run_shard_prefill()` 구현**:
   - Shard별 입력 텐서 바인딩
   - Graph 실행
   - 출력에서 KV cache, hidden state, ROPE 추출

2. **`run_shard_decode()` 구현**:
   - Shard별 입력 텐서 바인딩
   - Graph 실행
   - 출력에서 KV cache, hidden state 추출

3. **Logits 추출 및 토큰 디코딩**:
   - Final shard (Shard 7)의 logits output에서 next token 추출
   - Argmax + dequantization

이러한 부분들은 single-context 구현 (`llm_decode_runner.cpp`의 `run_prefill()` 및 `run_decode_step()`)을 참고하여 구현할 수 있습니다.
