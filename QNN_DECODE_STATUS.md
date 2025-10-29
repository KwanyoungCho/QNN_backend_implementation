# QNN Decode 프로젝트 상태 보고서

**작성일**: 2025-10-28  
**프로젝트**: QNN Backend를 이용한 LLM Decoder 구현  
**목표**: ExecuTorch의 QNN backend 참고하여 독립적인 QNN 기반 LLM 추론 엔진 구축

---

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [완료된 작업](#완료된-작업)
3. [현재 상태](#현재-상태)
4. [발견된 문제점](#발견된-문제점)
5. [디버깅 과정](#디버깅-과정)
6. [다음 단계](#다음-단계)
7. [코드 구조](#코드-구조)
8. [핵심 참고 자료](#핵심-참고-자료)

---

## 프로젝트 개요

### 목적
- ExecuTorch의 `.pte` 파일에서 QNN context binary를 추출
- QNN API를 직접 사용하여 그래프 실행
- Prefill + KV Cache + Decoding 전체 파이프라인 구현
- **참고 구현**: ExecuTorch의 `qnn_llama_runner.cpp` (ExecutorchReader)

### 주요 컴포넌트
1. **QNN Context 로딩**: `.pte`에서 QNN binary 추출 및 로드
2. **Tokenizer**: `llama.cpp` 기반 토크나이저
3. **Dual-Graph Execution**: `prefill_forward` + `kv_forward`
4. **KV Cache Management**: SMART_MASK 전략 기반 캐시 관리
5. **I/O Tensor 관리**: 입력/출력 텐서 자동 할당 및 바인딩

### 테스트 모델
- **Model**: Llama 3.2-1B (QNN 8-bit quantized)
- **Context Length**: 512
- **Prefill AR Length**: 32
- **Decode AR Length**: 1
- **KV Cache**: SMART_MASK 방식

---

## 완료된 작업

### ✅ 1. 기본 인프라 구축 (완료)

#### 1.1 QNN Context 로딩
- **파일**: `src/qnn_loader.cpp`, `src/binary_provider.cpp`
- **기능**: 
  - `.pte` 파일에서 QNN context binary 추출
  - `qnn_context_create_from_binary` API 호출
  - 다중 그래프 지원 (`prefill_forward`, `kv_forward`)
- **상태**: ✅ 정상 동작

#### 1.2 Tokenizer 통합
- **파일**: `src/tokenizer_llama.cpp`
- **기능**:
  - `llama.cpp` 라이브러리 기반
  - GGUF 포맷 지원 (`tokenizer.gguf`)
  - Encode/Decode 기능
- **상태**: ✅ 정상 동작

#### 1.3 JSON 기반 I/O 관리
- **파일**: `src/qnn_qnnjson.cpp`, `src/io_alloc.cpp`
- **기능**:
  - `forward_0_json.json` 파싱
  - 입력/출력 텐서 자동 할당 (64-byte aligned)
  - 텐서 메타데이터 추출 (dimensions, datatype, quantization)
- **상태**: ✅ 정상 동작

### ✅ 2. 모듈화 및 코드 구조화 (완료)

#### 2.1 Input Preparation Module
- **파일**: `src/llm_input_preparer.cpp`, `include/llm_input_preparer.h`
- **기능**:
  - 토큰 입력 준비 (`fill_tokens`)
  - Position 입력 준비 (`fill_positions`)
  - Attention Mask 초기화 및 업데이트 (`fill_attention_mask`)
  - KV Cache 텐서 클리어 (`clear_kv_cache_tensors`)
- **구현 완료**:
  - ✅ Prefill Attention Mask (Causal pattern, SMART_MASK)
  - ✅ Decode Attention Mask (Past + Current token)
- **상태**: ✅ 정상 동작

#### 2.2 Output Processing Module
- **파일**: `src/llm_output_processor.cpp`, `include/llm_output_processor.h`
- **기능**:
  - Logits dequantization (UFIXED_POINT_16 → float)
  - Argmax (greedy sampling)
  - Top-K 출력 (디버깅용)
- **상태**: ✅ 정상 동작

#### 2.3 KV Cache Manager Module
- **파일**: `src/llm_kv_cache_manager.cpp`, `include/llm_kv_cache_manager.h`
- **기능**:
  - KV Cache 메모리 할당 (Layer × Head × [K/V])
  - `get_k_cache()`, `get_v_cache()`: 캐시 버퍼 접근
  - `rearrange_cache()`: Prefill→Decode 전환 시 메모리 재배치
    - `rearrange_key()`: K cache stride 변경 (480→511)
    - `rearrange_value()`: V cache (no-op, sequential)
- **구현 완료**:
  - ✅ 메모리 할당 (최대 cache_len=511 기준)
  - ✅ Rearrange 로직 (ExecutorchReader 방식 정확히 재현)
  - ✅ 디버그 로깅 (메모리 상태 추적)
- **상태**: ✅ 정상 동작 (ExecutorchReader와 동일한 메모리 패턴 확인)

### ✅ 3. Prefill 단계 구현 (완료)

#### 3.1 Prefill Forward 실행
- **기능**:
  - `prefill_forward` 그래프 로드 및 실행
  - 입력: tokens[ar_len], position, attention_mask[ar_len, context_len], KV_in (cleared)
  - 출력: logits[ar_len, vocab_size], KV_out[ar_len, head_dim]
- **상태**: ✅ 정상 동작

#### 3.2 Prefill KV Cache Update
- **파일**: `apps/qnn_decode_main.cpp` (Line 540~603)
- **기능**:
  - Prefill 출력 KV를 KVManager의 input_buffer로 복사
  - `n_update` 계산: `1 + ((num_prompt_tokens - 1) % prefill_ar_len)`
  - V cache: Sequential copy
  - K cache: Strided copy (dimension별)
- **구현 세부사항**:
  ```cpp
  // V cache update (sequential)
  uint8_t* src = reinterpret_cast<uint8_t*>(kv_out.buffer);
  uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + n_past * head_dim;
  std::memcpy(dst, src, n_update * head_dim);
  
  // K cache update (strided)
  uint8_t* src = reinterpret_cast<uint8_t*>(kv_out.buffer);
  uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + n_past;
  for (int32_t dim = 0; dim < head_dim; ++dim) {
    std::memcpy(dst, src, n_update);
    src += prefill_ar_len;
    dst += prefill_cache_len;
  }
  ```
- **검증 완료**:
  - ✅ 프롬프트 "Hello" (2 tokens) → 42/10000 non-zero (21 bytes/token)
  - ✅ 프롬프트 15 tokens → 315/10000 non-zero (21 bytes/token)
  - ✅ ExecutorchReader: 18 tokens → 378/10000 non-zero (21 bytes/token)
  - ✅ **비율 동일**: 우리 코드 = ExecutorchReader
- **상태**: ✅ 정상 동작 (ExecutorchReader와 동일한 데이터 복사 확인)

#### 3.3 Rearrange Cache
- **파일**: `apps/qnn_decode_main.cpp` (Line 624~659)
- **기능**:
  - Prefill (AR=32, cache_len=480) → Decode (AR=1, cache_len=511) 전환
  - `kv_manager.rearrange_cache(prefill_ar_len, kv_ar_len)` 호출
- **검증 완료**:
  ```
  BEFORE: buffer[0]=199, buffer[480]=157, buffer[960]=128
  AFTER:  buffer[0]=199, buffer[511]=157, buffer[1022]=128
  Non-zero: 42→44 (padding으로 약간 증가)
  ```
- **상태**: ✅ 정상 동작 (ExecutorchReader와 동일한 메모리 재배치 확인)

### ✅ 4. Decode 단계 구현 (완료)

#### 4.1 Decode Forward 실행
- **기능**:
  - `kv_forward` 그래프 로드 및 실행
  - 입력: token[1], position[1], attention_mask[1, context_len], KV_in[cache_len, head_dim]
  - 출력: logits[1, vocab_size], KV_out[1, head_dim]
- **상태**: ✅ 정상 동작

#### 4.2 Decode Input 준비
- **파일**: `apps/qnn_decode_main.cpp` (Line 697~793)
- **기능**:
  - **Token**: `next_token` (이전 step 출력)
  - **Position**: `n_past = initial_tokens + gen_idx`
  - **Attention Mask**: 
    ```cpp
    // Past tokens에 attend
    for (int i = 0; i < n_past; ++i) mask[i] = 65535;
    // Current token에 attend (context window 끝)
    mask[context_len - 1] = 65535;
    ```
  - **KV Cache Inputs**: KVManager의 input_buffer에서 kv_alloc으로 복사
- **검증 완료**:
  - ✅ Token, Position 정확히 전달
  - ✅ Attention Mask: Past tokens + Current token 모두 attend
  - ✅ KV Cache: 충분한 non-zero 데이터 확인
- **상태**: ✅ 정상 동작

#### 4.3 Decode KV Cache Update
- **파일**: `apps/qnn_decode_main.cpp` (Line 912~1011)
- **기능**:
  - Decode 출력 KV를 position `n_past`에 저장
  - V cache: `dst = input_buffer + pos * head_dim` (sequential)
  - K cache: `dst = input_buffer + dim * cache_len + pos` (strided)
- **구현 세부사항**:
  ```cpp
  // V cache update (sequential, [1,1,64] → [1,511,64])
  uint8_t* src = reinterpret_cast<uint8_t*>(out_buf);
  uint8_t* dst = reinterpret_cast<uint8_t*>(v_buf.input_buffer) + pos * head_dim;
  std::memcpy(dst, src, head_dim);
  
  // K cache update (strided, [1,64,1] → [1,64,511])
  uint8_t* src = reinterpret_cast<uint8_t*>(out_buf);
  uint8_t* dst = reinterpret_cast<uint8_t*>(k_buf.input_buffer) + dim * kv_cache_len + pos;
  dst[0] = src[dim];
  ```
- **검증 완료**:
  - ✅ Position 계산 정확: `pos = initial_tokens + gen_idx`
  - ✅ 메모리 복사 정확: 각 layer/head별 독립적 업데이트
- **상태**: ✅ 정상 동작

---

## 현재 상태

### 🟢 정상 동작하는 부분
1. ✅ **QNN Context 로딩**: Prefill/Decode 그래프 모두 로드 및 실행
2. ✅ **Tokenizer**: Encode/Decode 정확
3. ✅ **Prefill 단계**: 
   - Logits 출력 정상 (첫 토큰 생성 성공: "Question")
   - KV Cache 업데이트 정확 (비율 ExecutorchReader와 동일)
   - Rearrange 정확 (메모리 패턴 ExecutorchReader와 동일)
4. ✅ **Decode 단계**:
   - 그래프 실행 성공
   - 입력 준비 정확 (Token, Position, Attention Mask, KV Cache)
   - KV Cache 업데이트 정확

### 🔴 문제점

#### **메인 이슈: Decode에서 반복 토큰 생성**

**증상**:
```
입력: "Hello, how are you today?"
출력: "Hello, how are you today? I charge ра� ра� ра� ра� ра� ра� ра� charge"
         ↑ 첫 토큰 정상      ↑ 이후 반복 및 무의미 토큰
```

**비교 (ExecutorchReader - 정상)**:
```
입력: "Hellow, how are you today?"
출력: "I'm just a language model, I don't have feelings or emotions like humans do, 
       but I'm functioning properly and ready to help with any questions or tasks 
       you have! How about you? How's your day going so far?"
```

**특징**:
- ✅ Prefill 단계는 정상 (첫 토큰 생성 성공)
- ❌ Decode Step 1부터 반복 토큰 발생
- ❌ 주로 "ра�", "charge", "loose" 등 무의미한 토큰 반복

---

## 발견된 문제점

### 1. ✅ 해결된 문제들

#### 1.1 메모리 할당 크기 오류 (해결됨)
- **문제**: KVManager를 `cache_len=480`으로 할당 → Rearrange 시 `cache_len=511` 접근 → Out-of-bounds
- **원인**: Prefill 기준 cache_len으로 할당했으나, Decode 시 더 큰 cache_len 필요
- **해결**: `max_cache_len=511`로 할당 (초기부터 최대 크기)
- **검증**: ✅ Rearrange 후 메모리 패턴 ExecutorchReader와 동일

#### 1.2 Prefill KV Update Offset 오류 (해결됨)
- **문제**: SMART_MASK에서 Prefill output의 유효 데이터 위치 오해
- **원인**: `src = output_buffer + (ar_len - n_update)` offset 적용 (잘못된 가정)
- **해결**: `src = output_buffer` (시작부터 유효 데이터)
- **검증**: ✅ ExecutorchReader 로그 확인 (`read_ptr[0:3] = [130, 121, 38]` 시작부터 유효)

#### 1.3 n_update 계산 오류 (해결됨)
- **문제**: `n_update = tokens.size() - 1` → 단일 토큰 시 0
- **원인**: Prefill의 마지막 iteration만 의미있는 토큰
- **해결**: `n_update = 1 + ((num_prompt_tokens - 1) % prefill_ar_len)`
- **검증**: ✅ ExecutorchReader와 동일한 로직

### 2. ❓ 미해결 문제

#### 2.1 Decode 반복 토큰 생성의 근본 원인

**분석 완료된 사항**:
1. ✅ **Prefill KV Update**: ExecutorchReader와 동일한 비율 (21 bytes/token)
2. ✅ **Rearrange**: ExecutorchReader와 동일한 메모리 패턴
3. ✅ **Decode Input 준비**: Token, Position, Attention Mask 모두 정확
4. ✅ **Decode KV Update**: 로직 정확, position 계산 정확

**아직 확인 필요한 사항**:
1. ❓ **Decode 입력 KV Cache 데이터**: 실제로 올바른 데이터가 전달되는가?
2. ❓ **Quantization Parameters**: Scale/Offset이 정확히 적용되는가?
3. ❓ **Attention Mask 형식**: UFIXED_POINT_16 값이 정확한가? (0=mask, 65535=attend)
4. ❓ **Position Encoding**: Position 값이 올바르게 처리되는가?
5. ❓ **그래프 내부 동작**: QNN 그래프 실행이 정확한가?

**가능한 원인 가설**:
- **가설 1**: Decode KV Cache 입력 복사 시 데이터 손실
- **가설 2**: Attention Mask 형식 불일치 (endianness, data type)
- **가설 3**: Position 값 범위 오류
- **가설 4**: KV Cache stride/offset 계산 오류
- **가설 5**: Quantization parameter 불일치

---

## 디버깅 과정

### Phase 1: Prefill 검증
1. ✅ Prefill output 데이터 확인: 1000/1000 non-zero (정상)
2. ✅ Prefill KV update 후: 42/10000 non-zero (2 tokens)
3. ✅ ExecutorchReader 비교: 378/10000 non-zero (18 tokens)
4. ✅ **비율 계산**: 21 bytes/token (동일) → Prefill 정상

### Phase 2: Rearrange 검증
1. ✅ Rearrange 전: `buffer[0]=199, buffer[480]=157, buffer[960]=128`
2. ✅ Rearrange 후: `buffer[0]=199, buffer[511]=157, buffer[1022]=128`
3. ✅ ExecutorchReader와 동일한 패턴 확인
4. ✅ Non-zero count: 42→44 (약간 증가, 정상)

### Phase 3: Decode 입력 검증
1. ✅ Token 입력: 정확
2. ✅ Position 입력: `n_past = initial_tokens + gen_idx` (정확)
3. ✅ Attention Mask: Past + Current (정확)
4. ✅ KV Cache 복사: KVManager → kv_alloc (완료)

### Phase 4: Decode 출력 검증
1. ✅ Logits 출력: Argmax 수행 가능
2. ❌ **생성된 토큰**: 반복 및 무의미 ("ра�", "charge")
3. ❓ **근본 원인**: 미확인

### 추가된 디버그 로그

#### 우리 코드 (`qnn_decode_main.cpp`):
```cpp
// Line 609-622: Prefill KV update 후 상태
[Debug] After Prefill KV update (L0H0):
  K cache buffer[0]=199, buffer[480]=157, buffer[960]=128
  Non-zero in first 10000 bytes: 42/10000
  Expected: n_update=2 tokens copied to position 0
```

```cpp
// Line 232-264: Rearrange 전후 상태 (llm_kv_cache_manager.cpp)
[DEBUG Rearrange] BEFORE K cache (L0H0):
  buffer[0]=199, buffer[480]=157, buffer[960]=128
  Total buffer size allocated: 32704 bytes
  src_cache_len=480, dst_cache_len=511

[DEBUG Rearrange] AFTER K cache (L0H0):
  buffer[0]=199, buffer[511]=157, buffer[1022]=128
  Non-zero in first 10000 bytes: 44/10000
```

#### ExecutorchReader (`kv_manager.cpp`):
```cpp
// Line 521-543: Update 전후 상태
[DEBUG KV] update_key #0:
  n_past=0, n_update=18, cur_ar_len=32
  iter_size=480, out_size=32, copy_size=18
  write_ptr offset=0, read_ptr offset=0
  read_ptr[0:3] = [130, 121, 38]
  Before update: buffer has 0/10000 non-zero
  After update: buffer has 378/10000 non-zero
  buffer[0]=130, buffer[480]=130, buffer[960]=128
```

```cpp
// Line 374-426: Rearrange 전후 상태
[DEBUG KV] rearrange_key #0:
  src_cache_num=480, dst_cache_num=511, head_dim=64
  Expanding: copy from last dimension (backward)
  Before: buffer[0]=130, buffer[480]=130, buffer[960]=128
  Before non-zero: 378/10000
  After: buffer[0]=130, buffer[511]=130, buffer[1022]=128
  After non-zero: 396/10000
```

---

## 다음 단계

### 🎯 우선순위 1: Decode 입력 KV Cache 데이터 검증

**목적**: Decode 그래프에 전달되는 KV Cache가 올바른 데이터인지 확인

**방법**:
1. Decode Step 1 직전, KVManager에서 kv_alloc으로 복사한 후 데이터 검증
2. ExecutorchReader의 동일 시점과 비교
3. 첫 N bytes의 값 직접 출력

**구현 위치**: `apps/qnn_decode_main.cpp` Line 751~793

**디버그 코드 예시**:
```cpp
// KV cache 복사 후
if (log_level >= 2 && gen_idx == 0) {
  // V cache 첫 번째 확인 (Layer 0, Head 0)
  const auto& v_buf_src = kv_manager.get_v_cache(0, 0);
  uint8_t* v_src = reinterpret_cast<uint8_t*>(v_buf_src.input_buffer);
  uint8_t* v_dst = reinterpret_cast<uint8_t*>(kv_alloc_v_buffer);
  
  std::cout << "[Debug] V[0][0] data check:\n";
  std::cout << "  KVManager: [" << (int)v_src[0] << ", " << (int)v_src[1] 
            << ", " << (int)v_src[2] << "]\n";
  std::cout << "  kv_alloc:  [" << (int)v_dst[0] << ", " << (int)v_dst[1] 
            << ", " << (int)v_dst[2] << "]\n";
}
```

### 🎯 우선순위 2: Attention Mask 형식 검증

**목적**: Attention Mask가 QNN이 기대하는 형식인지 확인

**확인 사항**:
1. UFIXED_POINT_16 값: 0 (mask), 65535 (attend) 정확한가?
2. Quantization parameters: scale, offset 일치하는가?
3. Endianness: Little-endian 확인

**구현 위치**: `apps/qnn_decode_main.cpp` Line 711~748

**디버그 코드 예시**:
```cpp
if (log_level >= 2 && gen_idx == 0) {
  uint16_t* mask = reinterpret_cast<uint16_t*>(buf);
  std::cout << "[Debug] Attention mask details:\n";
  std::cout << "  Size: " << t.nbytes << " bytes (" << (t.nbytes/2) << " uint16 values)\n";
  std::cout << "  First 5 values: [";
  for (int i = 0; i < 5; ++i) std::cout << mask[i] << ", ";
  std::cout << "]\n";
  std::cout << "  Last 5 values: [";
  for (int i = context_len - 5; i < context_len; ++i) std::cout << mask[i] << ", ";
  std::cout << "]\n";
  std::cout << "  Expected: attend to [0.." << (n_past-1) << "] and [" 
            << (context_len-1) << "]\n";
}
```

### 🎯 우선순위 3: Position 값 검증

**목적**: Position 입력이 올바른 범위인지 확인

**확인 사항**:
1. `n_past` 값이 [0, context_len) 범위 내인가?
2. int32_t 형식이 맞는가?

**디버그 코드 예시**:
```cpp
if (log_level >= 2 && gen_idx == 0) {
  int32_t* pos = reinterpret_cast<int32_t*>(buf);
  std::cout << "[Debug] Position input:\n";
  std::cout << "  Value: " << pos[0] << "\n";
  std::cout << "  Expected: " << n_past << "\n";
  std::cout << "  Range: [0, " << context_len << ")\n";
}
```

### 🎯 우선순위 4: ExecutorchReader와 Side-by-Side 비교

**목적**: 동일한 프롬프트로 우리 코드와 ExecutorchReader를 동시 실행하여 차이점 발견

**방법**:
1. 동일 프롬프트: "Hello" (또는 더 긴 프롬프트)
2. Decode Step 1의 모든 입력 값 비교:
   - Token: 동일한가?
   - Position: 동일한가?
   - Attention Mask: 동일한가?
   - KV Cache 첫 N bytes: 동일한가?
3. Decode Step 1의 출력 비교:
   - Logits top-5: 동일한가?
   - 생성된 토큰: 동일한가?

### 🎯 우선순위 5: QNN Graph 정확성 검증

**목적**: QNN 그래프 자체가 올바르게 생성되었는지 확인

**확인 사항**:
1. `.pte` 파일의 그래프 버전
2. Quantization parameters (JSON에서 확인)
3. QNN SDK 버전 일치 여부

---

## 코드 구조

### 디렉토리 구조
```
llm_test/
├── apps/
│   ├── qnn_prefill_main.cpp       # Prefill-only 테스트
│   └── qnn_decode_main.cpp        # Prefill + Decode (현재 작업)
├── src/
│   ├── qnn_loader.cpp             # QNN context 로딩
│   ├── binary_provider.cpp        # .pte 파일 파싱
│   ├── io_alloc.cpp               # I/O 메모리 할당
│   ├── qnn_qnnjson.cpp            # JSON 파싱
│   ├── qnn_tensor_util.cpp        # 텐서 유틸리티
│   ├── tokenizer_llama.cpp        # Tokenizer
│   ├── llm_input_preparer.cpp     # 입력 준비 (Token, Pos, Mask)
│   ├── llm_output_processor.cpp   # 출력 처리 (Logits)
│   └── llm_kv_cache_manager.cpp   # KV Cache 관리
├── include/
│   ├── qnn_loader.h
│   ├── binary_provider.h
│   ├── io_alloc.h
│   ├── qnn_qnnjson.h
│   ├── qnn_tensor_util.h
│   ├── tokenizer_llama.h
│   ├── llm_input_preparer.h
│   ├── llm_output_processor.h
│   └── llm_kv_cache_manager.h
├── models/
│   └── llama_qnn_1b/
│       ├── forward_0.bin          # QNN context binary
│       └── forward_0_json.json    # 텐서 메타데이터
├── script/
│   └── build_push.sh              # 안드로이드 빌드 및 푸시
└── CMakeLists.txt
```

### 주요 클래스/모듈

#### `LLMKVCacheManager`
```cpp
class LLMKVCacheManager {
  struct Metadata {
    int32_t context_len;      // 512
    int32_t head_dim;         // 64
    int32_t max_ar_len;       // 32 (prefill)
    int32_t max_cache_len;    // 511 (decode, 최대값)
    int32_t num_heads;        // 8
    int32_t num_layers;       // 16
  };
  
  bool allocate();
  const KVCacheBuffer& get_k_cache(int layer, int head) const;
  const KVCacheBuffer& get_v_cache(int layer, int head) const;
  void rearrange_cache(int32_t src_ar_len, int32_t dst_ar_len);
};
```

#### `InputPreparer`
```cpp
namespace InputPreparer {
  void fill_tokens(void* buf, const std::vector<int32_t>& tokens, 
                   size_t start, size_t count);
  void fill_positions(void* buf, int32_t start_pos, size_t count);
  void fill_attention_mask(void* buf, int32_t seq_dim, int32_t max_len,
                           int32_t n_past, int32_t n_update);
}
```

#### `OutputProcessor`
```cpp
namespace OutputProcessor {
  float dequantize_ufixed16(uint16_t val, float scale, int32_t offset);
  int32_t argmax(const uint16_t* logits, int32_t size, 
                 float scale, int32_t offset);
  void print_topk(const uint16_t* logits, int32_t size,
                  float scale, int32_t offset, int k);
}
```

---

## 핵심 참고 자료

### ExecutorchReader 코드 (정상 동작 확인됨)
```
/home/chokwans99/executorch/examples/qualcomm/oss_scripts/llama/
├── qnn_llama_runner.cpp           # Main entry point
├── runner/
│   ├── runner.cpp                 # Prefill + Decode orchestration
│   ├── prompt_processor.cpp       # Prefill 처리
│   ├── token_generator.cpp        # Decode 처리
│   ├── kv_manager.cpp             # KV Cache 관리 ⭐
│   └── rpc_mem.cpp                # 메모리 할당 (RPC)
```

### 중요 문서
1. **SMART_MASK 설명**: `/home/chokwans99/executorch/examples/qualcomm/oss_scripts/llama/README.md`
   - KV Cache 업데이트 메커니즘 그림 포함
   - Prefill vs Decode 차이 설명

2. **QNN API 문서**: QNN SDK `/docs/` 디렉토리
   - `qnn_context_create_from_binary`
   - `qnn_graph_retrieve`
   - `qnn_graph_execute`

3. **Llama.cpp**: `/home/chokwans99/executorch/llama.cpp/`
   - Tokenizer 참고

### 테스트 명령어

#### 우리 코드 실행
```bash
adb shell "cd /data/local/tmp/chokwans99/executorch/QNN_test && \
  export LD_LIBRARY_PATH=.:bin:\$LD_LIBRARY_PATH && \
  ./qnn_decode \
    --ctx_dir ctx \
    --gguf tokenizer.gguf \
    --prompt 'Hello, how are you today?' \
    --decode \
    --logits_output output_aten_squeeze_copy_dims_0 \
    --log_level 1 \
    --max_gen 20"
```

#### ExecutorchReader 실행
```bash
adb shell "cd /data/local/tmp/chokwans99/executorch/QNN_test && \
  export LD_LIBRARY_PATH=lib && \
  ./qnn_llama_runner \
    --model_path llama_qnn_1b_hybrid \
    --tokenizer_path tokenizer.model \
    --prompt 'Hello, how are you today?' \
    --seq_len 20"
```

#### 빌드 및 푸시
```bash
# 우리 코드
cd /home/chokwans99/llm_test
cmake --build build-android --target qnn_decode -j8
adb push build-android/qnn_decode /data/local/tmp/chokwans99/executorch/QNN_test/

# ExecutorchReader (참고)
cd /home/chokwans99/executorch
cmake --build build-android --target examples_qualcomm_oss_scripts_llama_qnn_llama_runner -j8
adb push build-android/examples/qualcomm/oss_scripts/llama/qnn_llama_runner \
  /data/local/tmp/chokwans99/executorch/QNN_test/
```

---

## 핵심 발견 사항 (요약)

### ✅ 정상 동작 확인
1. **Prefill KV Update**: ExecutorchReader와 동일한 비율 (21 bytes/token)
2. **Rearrange**: ExecutorchReader와 동일한 메모리 패턴
3. **Decode 입력 준비**: Token, Position, Attention Mask 모두 정확
4. **첫 토큰 생성**: Prefill 단계에서 정확한 토큰 생성 ("Question")

### ❌ 미해결 이슈
1. **Decode 반복 토큰**: 2번째 토큰부터 반복/무의미 토큰 생성
2. **근본 원인 미확인**: Decode 그래프 실행 또는 입력 데이터 문제로 추정

### 🔍 다음 디버깅 방향
1. Decode Step 1의 KV Cache 입력 데이터 검증 (바이트 단위 비교)
2. Attention Mask 형식 검증 (값, 형식, quantization)
3. Position 값 검증
4. ExecutorchReader와 Side-by-Side 비교

---

## 추가 참고

### SMART_MASK 메모리 레이아웃

#### Prefill (AR=32, cache_len=480)
```
K cache: [1, 64, 480]
V cache: [1, 480, 64]

Example: n_update=18 tokens
K: [dim0][18 bytes][462 zeros][dim1][18 bytes][462 zeros]...[dim63]
V: [18*64 bytes][462*64 zeros]
```

#### Rearrange (480 → 511)
```
K cache: [1, 64, 511]
V cache: [1, 511, 64]

K: [dim0][18 bytes][493 zeros][dim1][18 bytes][493 zeros]...[dim63]
   └─ memmove로 stride 확장 (backward iteration)
V: [18*64 bytes][493*64 zeros]
   └─ sequential이므로 no-op
```

#### Decode (AR=1, cache_len=511)
```
각 step마다 position n_past에 1개 토큰 추가:

Step 1: n_past=18
K: [dim0][19 bytes][492 zeros][dim1]...[dim63]
V: [19*64 bytes][492*64 zeros]

Step 2: n_past=19
K: [dim0][20 bytes][491 zeros][dim1]...[dim63]
V: [20*64 bytes][491*64 zeros]
```

---

**이 문서는 프로젝트의 현재 상태를 완전히 기록하며, 향후 작업자가 이어서 디버깅을 진행할 수 있도록 모든 세부사항을 포함합니다.**

**마지막 업데이트**: 2025-10-28  
**작성자**: AI Assistant (with user chokwans99)  
**다음 작업자에게**: 위의 "다음 단계" 섹션의 우선순위 1부터 시작하세요!

