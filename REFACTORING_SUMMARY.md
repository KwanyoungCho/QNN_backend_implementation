# 🎯 코드 모듈화 완료 요약

## 📋 변경 사항

### ✨ 새로 추가된 모듈

#### 1. **LLMKVCacheMapper** (`llm_kv_cache_mapper.h/cpp`)
- **목적**: QNN JSON KV cache 텐서를 KVManager 버퍼에 자동 매핑
- **핵심 기능**:
  - `build_mapping()`: JSON 텐서 이름 분석 → layer/head 매핑 구축
  - `create_buffer_override()`: Zero-copy shared memory 매핑 생성
- **Executorch 분석 결과 반영**:
  ```
  Input Order Pattern (per layer):
    V cache 8개 (H0~H7) → K cache 8개 (H0~H7)
  ```

#### 2. **LLMDecodeRunner** (`llm_decode_runner.h/cpp`)
- **목적**: High-level Prefill + Decode 실행 API
- **핵심 기능**:
  - `initialize()`: QNN backend 로드, 그래프 파싱, KV cache 설정
  - `generate()`: Prompt → 텍스트 생성 (end-to-end)
  - `run_prefill()`: Prefill 단계 실행
  - `run_decode_step()`: Decode 단계 실행
- **자동화**:
  - Metadata 추출 (context_len, num_layers, num_heads, head_dim)
  - KV cache 할당 및 매핑
  - Rearrange cache (480 → 511)
  - KV cache update from outputs

#### 3. **qnn_llm_generate** (`apps/qnn_llm_generate.cpp`)
- **목적**: 간결한 사용자 인터페이스
- **특징**: 90줄 코드로 완전한 텍스트 생성
- **사용법**:
  ```bash
  ./build/qnn_llm_generate \
    --ctx_dir models/llama_qnn_1b \
    --tokenizer models/llama_qnn_1b/tokenizer.model \
    --prompt "The capital of France is" \
    --max_gen 50
  ```

---

## 📊 Before vs After

### Before (`qnn_decode_main.cpp`)
```
❌ 1000+ 줄 monolithic 코드
❌ 하드코딩된 KV cache 매핑 (tensor_idx - 2, tensor_idx - 130)
❌ 매 decode step마다 memcpy (V/K cache 복사)
❌ 유지보수 어려움
❌ 재사용 불가능
```

### After (Modularized)
```
✅ LLMKVCacheMapper: 자동 텐서 매핑
✅ LLMDecodeRunner: 재사용 가능한 API
✅ Zero-copy KV cache: 직접 shared memory 사용
✅ 90줄 main app (vs 1000+ 줄)
✅ 명확한 책임 분리
✅ 쉬운 확장 및 테스트
```

---

## 🗂️ 모듈 구조

```
llm_test/
├── include/
│   ├── qnn_loader.h              # QNN backend 로드 및 실행
│   ├── qnn_qnnjson.h             # JSON 파서
│   ├── io_alloc.h                # I/O 버퍼 할당
│   ├── qnn_tensor_util.h         # QNN 텐서 유틸
│   ├── tokenizer_llama.h         # Llama tokenizer
│   ├── llm_input_preparer.h      # Input 텐서 준비
│   ├── llm_output_processor.h    # Output 텐서 처리
│   ├── llm_kv_cache_manager.h    # KV cache 메모리 관리
│   ├── llm_kv_cache_mapper.h     # ✨ KV cache 매핑
│   └── llm_decode_runner.h       # ✨ High-level API
│
├── src/
│   ├── qnn_loader.cpp
│   ├── qnn_qnnjson.cpp
│   ├── io_alloc.cpp
│   ├── qnn_tensor_util.cpp
│   ├── tokenizer_llama.cpp
│   ├── llm_input_preparer.cpp
│   ├── llm_output_processor.cpp
│   ├── llm_kv_cache_manager.cpp
│   ├── llm_kv_cache_mapper.cpp   # ✨ NEW
│   └── llm_decode_runner.cpp     # ✨ NEW
│
└── apps/
    ├── qnn_llm_generate.cpp      # ✨ NEW: 간결한 생성 앱
    ├── qnn_decode_main.cpp       # Original (참고용)
    └── ...
```

---

## 🔍 핵심 개선 사항

### 1. **Zero-Copy KV Cache**

**Before**:
```cpp
// 매 decode step마다 복사
for (auto& kv_input : kv_inputs) {
  std::memcpy(qnn_buffer, kv_cache_buffer, size);  // 비효율
}
```

**After**:
```cpp
// 한 번만 매핑
auto kv_override = LLMKVCacheMapper::create_buffer_override(mapping, kv_manager);
// QNN input이 직접 KV cache 버퍼를 가리킴 (zero-copy!)
```

### 2. **자동 텐서 매핑**

**Before**:
```cpp
// 하드코딩된 인덱스
int v_offset = tensor_idx - 2;           // input_2가 V cache 시작
int k_offset = tensor_idx - (2 + 128);   // input_130이 K cache 시작
```

**After**:
```cpp
// 자동 분석 및 매핑
auto mapping = LLMKVCacheMapper::build_mapping(graph, num_heads, head_dim);
// Shape 기반 자동 감지: [1, cache_len, 64] = V, [1, 64, cache_len] = K
```

### 3. **간결한 사용자 코드**

**Before**: 1000+ lines
**After**: 90 lines

```cpp
int main(int argc, char** argv) {
  LLMDecodeConfig config;
  // ... parse args ...
  
  LLMDecodeRunner runner(config);
  if (!runner.initialize()) {
    std::cerr << runner.get_error() << "\n";
    return 1;
  }
  
  std::string output;
  if (!runner.generate(prompt, output)) {
    std::cerr << runner.get_error() << "\n";
    return 1;
  }
  
  std::cout << output << "\n";
  return 0;
}
```

---

## 🚀 사용 방법

### 빌드

```bash
cd /home/chokwans99/llm_test

# Clean build
./build.sh clean

# Or just rebuild
./build.sh
```

### 실행

```bash
./build/qnn_llm_generate \
  --ctx_dir models/llama_qnn_1b \
  --tokenizer models/llama_qnn_1b/tokenizer.model \
  --backend_so /path/to/libQnnHtp.so \
  --prompt "The capital of France is" \
  --max_gen 50 \
  --log_level 1
```

### 빠른 테스트

```bash
# test_generate.sh를 수정하여 경로 설정 후
./test_generate.sh
```

---

## 📈 성능 개선

| 항목 | Before | After | 개선 |
|------|--------|-------|------|
| KV cache 복사 | 매 step | 0 (zero-copy) | ∞ |
| 코드 가독성 | 1000+ lines | 90 lines | 11x |
| 유지보수성 | 낮음 | 높음 | ✅ |
| 재사용성 | 불가능 | 가능 | ✅ |
| 확장성 | 어려움 | 쉬움 | ✅ |

---

## 🎓 Executorch 분석 통해 얻은 인사이트

### KV Cache Input 순서
```
prefill_forward/kv_forward:
  input_0:     tokens
  input_1:     input_pos
  input_2~9:   V cache L0 H0~7 [1, cache_len, 64]
  input_10~17: K cache L0 H0~7 [1, 64, cache_len]
  input_18:    attention_mask
  input_19~26: V cache L1 H0~7
  ... (반복)
```

### MethodMeta vs Context Binary
- **MethodMeta** (Executorch internal): K all → V all
- **Context Binary** (JSON): V/K interleaved per layer
- **Solution**: JSON 순서 직접 사용 (context binary가 실제 실행 순서)

---

## 📝 다음 단계

1. ✅ **모듈화 완료**
2. ✅ **빌드 성공**
3. ⏭️ **실제 디바이스 테스트**
4. ⏭️ **성능 벤치마크**
5. ⏭️ **다른 모델 지원** (3B, 8B)
6. ⏭️ **Android 배포 테스트**

---

## 🔧 문제 해결

### 빌드 오류
```bash
# Clean rebuild
./build.sh clean
```

### 실행 오류
```bash
# QNN backend path 확인
ls /path/to/libQnnHtp.so

# Context 파일 확인
ls models/llama_qnn_1b/forward_0.bin
ls models/llama_qnn_1b/forward_0_json.json

# Tokenizer 확인
ls models/llama_qnn_1b/tokenizer.model
```

### 디버그 모드
```bash
./build/qnn_llm_generate \
  --ctx_dir models/llama_qnn_1b \
  --tokenizer models/llama_qnn_1b/tokenizer.model \
  --prompt "Test" \
  --log_level 2  # 상세 디버그 로그
```

---

## 📚 참고 문서

- `README_MODULES.md`: 상세 모듈 설명
- `src/README.md`: 기존 모듈 문서
- Executorch 분석 로그: checkpoint 4 참고

---

## ✅ 완료된 작업

1. ✅ LLMKVCacheMapper 모듈 생성
2. ✅ LLMDecodeRunner 모듈 생성
3. ✅ qnn_llm_generate 앱 생성
4. ✅ CMakeLists.txt 업데이트
5. ✅ 빌드 스크립트 생성
6. ✅ 테스트 스크립트 생성
7. ✅ 문서화 완료
8. ✅ 빌드 성공 확인

**모든 모듈화 작업이 완료되었습니다!** 🎉
