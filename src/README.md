## 개요

본 문서는 Executorch의 QNN 실행 흐름을 최대한 그대로 따르는 형태로 구현한 `llm_test` 프로젝트의 설계, 구현 세부, 사용법, 메모리/타입 처리, 문제 해결 가이드까지 모든 내용을 매우 자세히 설명합니다. 목표는 다음과 같습니다.

- QNN 컨텍스트 바이너리(멀티-그래프 포함)를 안전하게 복원하고, 그래프 핸들을 조회한 뒤, 각 그래프의 I/O 텐서를 올바른 크기와 포맷으로 바인딩하여 `graphExecute`를 성공적으로 호출
- Executorch의 실제 런타임 흐름과 동일한 QNN API 사용 순서와 바인딩 방식을 준수(예: `contextCreateFromBinary` → `graphRetrieve` → 등록된 텐서에 clientBuf만 채워 `graphExecute`)
- QNN SDK의 `qnn-context-binary-utility`가 생성한 JSON(`forward_{i}_json.json`)을 “I/O 메타데이터의 단일 소스”로 사용하여 I/O 메모리 크기(nbytes) 계산 및 텐서 바인딩
- 샤딩된 컨텍스트(예: 8개)가 있을 때도 각 샤드별로 동일한 방식으로 복원/바인딩/실행 가능


## 디렉토리 구조와 구성 요소

- 라이브러리 코어(`qnn_ctx_core`): 공용 로직을 모듈화
  - `include/qnn_loader.h` · `src/qnn_loader.cpp`
    - QNN SO 동적 로딩(`dlopen`/`dlsym`), provider 조회(`QnnInterface_getProviders`), backend/device 생성
    - `create_context_from_binary(const void* binary, size_t nbytes)`: 단일 컨텍스트 바이너리 복원(`qnn_context_create_from_binary`)
    - `retrieve_graph(size_t ctx_idx, const std::string& graph_name)`: 그래프 핸들 조회(`graphRetrieve`)
    - `execute_graph(size_t ctx_idx, const std::string& graph_name, const std::vector<Qnn_Tensor_t>& inputs, std::vector<Qnn_Tensor_t>& outputs)`: 실행(`graphExecute`)
    - 로그 레벨 설정, 기본 로거 사용(QNN 기본 로그를 사용. Android에선 logcat, Host에선 stderr)
  - `include/qnn_qnnjson.h` · `src/qnn_qnnjson.cpp`
    - `qnn-context-binary-utility`가 생성한 JSON(`forward_{i}_json.json`) 파서
    - 각 그래프와 텐서의 `id`, `name`, `dims`, `dataType(문자열/숫자코드)`, `bytesPerElement(있으면 신뢰)`, `nbytes(없으면 계산)`, `quantization` 필드 해석
  - `include/qnn_tensor_util.h` · `src/qnn_tensor_util.cpp`
    - `QnnTensorHolder`: JSON 메타 + 할당 버퍼를 사용해 `Qnn_Tensor_t`(v2)를 구성하는 유틸리티
    - 설정 요소: `id`, `name`, `type(APP_WRITE/APP_READ)`, `memType(RAW)`, `dataFormat(FLAT_BUFFER)`, `dims`, `dataType`, `clientBuf.data/.dataSize`, `quantizeParams`
  - `src/binary_provider.cpp`
    - `ctx_dir`에서 `forward_<i>.bin`과 `forward_<i>_json.json` 페어를 탐색
    - 컨텍스트 바이너리는 `mmap`으로 매핑하여 메모리 효율 개선

- 앱(`apps/`)
  - `qnn_io_plan_main.cpp`
    - 샤드 디렉토리(`--ctx_dir`)에 있는 컨텍스트/JSON 페어를 순회하며, 컨텍스트 복원 → 그래프 조회(`prefill_forward`, `kv_forward`) → JSON 기반 메모리 계획 및 텐서 생성 → `graphExecute` 수행
    - 상세 플래그는 아래 “실행 방법” 참고

## 빠른 시작(처음 보는 사람용)

1) QNN SDK 설치 확인(QNN 2.37.x 기준). `libQnnSystem.so`, `libQnnHtp.so`를 기기 혹은 작업 디렉토리에 배치
2) PTE 파일에서 QNN 컨텍스트 바이너리와 JSON을 준비
   - 컨텍스트 바이너리: Executorch 빌드 산출물에서 추출하거나, 제공된 `forward_<i>.bin` 사용
   - JSON: QNN SDK의 `qnn-context-binary-utility`로 각 컨텍스트의 그래프/텐서 I/O 정보를 덤프(아래 “JSON 파일 출처와 생성법” 참조)
3) 디렉토리 준비: `ctx_dir`에 `forward_0.bin` … `forward_N.bin`과 각 `forward_i_json.json`을 페어로 배치
4) 빌드 후 실행
   - Host: `./build/apps/qnn_io_plan --ctx_dir /path/to/ctx_dir --log_level 3`
   - Android: `adb push`로 파일 배치 후 `adb shell`에서 동일 실행(기본 SO는 현 디렉토리 참조)
5) 정상 시나리오: 각 샤드에서 `prefill_forward`, `kv_forward` 그래프가 순차 실행됨. 에러 시 “문제 해결 가이드” 확인

## 파일/모듈별 상세 역할

### 1) 로더
- `include/qnn_loader.h` / `src/qnn_loader.cpp`
  - 책임
    - QNN 공유 라이브러리 로딩(`dlopen`)과 심볼 로딩(`dlsym`)
    - Provider 조회(`QnnInterface_getProviders`) 및 첫 Provider 선택
    - Backend/Device 수명 주기 관리(`backendCreate`/`deviceCreate`/해제)
    - 컨텍스트 복원(`qnn_context_create_from_binary`)과 해제
    - 그래프 핸들 조회(`graphRetrieve`), 실행(`graphExecute`)
    - 로그 레벨 설정(QNN의 `QnnLog` 이용, 현재는 기본 로거 사용)
  - 왜 필요한가
    - Executorch가 내부에서 수행하는 초기화/복원/실행 과정을 독립 모듈로 분리하여 재사용성 확보
  - 주의
    - `graphExecute` 시 입력은 `APP_WRITE`, 출력은 `APP_READ`여야 하며, 등록된 텐서 `id`와 일치해야 함

### 2) JSON 파서
- `include/qnn_qnnjson.h` / `src/qnn_qnnjson.cpp`
  - 책임
    - `qnn-context-binary-utility`가 생성한 JSON(`forward_{i}_json.json`)에서 그래프/텐서 메타를 안전하게 파싱
    - `id`, `name`, `dims`, `dataType(문자열/정수)`, `bytesPerElement`, `quantization` 추출
    - `nbytes`가 없으면 계산(bpe × ∏dims)
  - 왜 필요한가
    - 표준 QNN C API만으로는 컨텍스트 내부 그래프 I/O 메타를 완전 탐색하기 어려움 → SDK JSON을 “단일 소스”로 채택
  - 주의
    - `dimensions`/`currentDimensions` 중 실제 런타임 차원에 해당하는 값을 사용(파일마다 차이가 있어 양쪽 지원)
    - `dataType`이 정수 코드로 오는 경우가 있음(예: 1032/1046/562). 매핑 규칙에 따라 bpe 도출

### 3) 텐서 유틸리티
- `include/qnn_tensor_util.h` / `src/qnn_tensor_util.cpp`
  - 책임
    - JSON 메타와 할당 버퍼로 `Qnn_Tensor_t(v2)`를 구성하는 `QnnTensorHolder` 제공
    - `id`/`name`/`type(APP_WRITE/APP_READ)`/`memType(RAW)`/`dataFormat(FLAT_BUFFER)`/`dims`/`dataType`/`clientBuf`/`quantizeParams` 채우기
  - 왜 필요한가
    - Executorch와 동일한 “등록 텐서에 clientBuf만 꽂아 실행” 모델을 안전하게 구현
  - 주의
    - QNN SDK 2.37에서는 v2 구조를 사용. `clientBuf.dataSize`는 32-bit일 수 있으므로 캐스팅 유의
    - per-axis quant는 encoding 메타까지만 채우고, 실제 스케일/오프셋 배열 포인터 연결은 추후 확장

### 4) 바이너리 공급자
- `src/binary_provider.cpp`
  - 책임
    - `ctx_dir`에서 `forward_<i>.bin`/`forward_<i>_json.json` 페어를 스캔
    - 바이너리를 `mmap`으로 매핑하여 파일 I/O 및 메모리 사용 최적화
  - 왜 필요한가
    - 대용량 컨텍스트를 효율적으로 다루기 위함. 불필요한 전체 로드/복사를 피함
  - 주의
    - 파일 디스크립터/매핑 해제 수명 관리 철저(종료 시 `munmap`/`close`)

### 5) 실행 앱
- `apps/qnn_io_plan_main.cpp`
  - 책임
    - CLI 파싱(그래프명은 기본 `prefill_forward`, `kv_forward`), SO 경로 기본값(현 디렉토리), 로그레벨 등
    - 컨텍스트 복원→그래프 조회→JSON 기반 I/O 계획→`Qnn_Tensor_t` 구성→`graphExecute`
  - 왜 필요한가
    - 전체 체인을 통합 검증하고, 다른 프레임워크(예: llama.cpp) 통합 시 참조 예제로 활용

## 구현 중심 해설(내부 동작 디테일)

- `qnn_loader.cpp` 내부 흐름
  - `dlopen(system_so)`, `dlopen(backend_so)` → `dlsym("QnnInterface_getProviders")`로 provider 배열 획득 → 첫 provider 선택
  - `provider->systemInterface.qnn_backend_create(...)` 로 backend 핸들 생성
  - `provider->systemInterface.qnn_device_create(...)` 로 device 핸들 생성
  - 컨텍스트 복원: `provider->context.qnn_context_create_from_binary(backend, device, config=null, binary, size, &ctx, profile=null)`
  - 그래프 조회: `provider->graph.qnn_graph_retrieve(ctx, graphName.c_str(), &graph)`
  - 실행: `provider->graph.qnn_graph_execute(graph, inputs, inCount, outputs, outCount, /*profile*/nullptr, /*signal*/nullptr)`
  - 로그: QNN 기본 로거를 사용하고, `set_log_level(int)`로 레벨만 조정(콜백 미사용)
  - 자원 수명: backend/device는 프로세스 동안 1회 생성, 컨텍스트/그래프 핸들은 샤드/그래프별로 벡터/맵에 보관 후 종료 시 파괴

- `qnn_qnnjson.cpp` 파싱 구현
  - 엄격한 키 탐색으로 `id/name/dims/dataType/bytesPerElement/quantization` 취득
  - `bytesPerElement` 부재 시 `dataType`(문자열 또는 정수 코드) 기반으로 바이트 수 도출
  - `dims`는 `dimensions`/`currentDimensions` 양쪽 지원, 0 차원 보호
  - `nbytes = bytesPerElement × ∏dims`로 계산. JSON의 `nbytes`가 있더라도 충돌 시 재계산 가능(현재는 bpe를 우선 신뢰)
  - quantization은 per-tensor(scale, offset) 즉시 매핑, per-axis는 encoding 메타까지만 보존(배열 포인터 미연결)

- `qnn_tensor_util.cpp` 텐서 구성
  - `Qnn_Tensor_t v2` 사용: `id/name/type(APP_WRITE|APP_READ)/dataFormat(FLAT_BUFFER)/memType(RAW)/dataType/dimensions` 채움
  - `clientBuf.data`에 호스트 버퍼 포인터, `clientBuf.dataSize`에 `nbytes(uint32_t 캐스팅)` 지정
  - `quantizeParams`는 JSON에 따라 per-tensor를 채움. per-axis는 후속 확장 포인트
  - 주의: `id`는 그래프 등록 텐서의 ID와 정확히 일치해야 하며, 그렇지 않으면 "Expected Tensor ID ... not found" 오류 발생

- `binary_provider.cpp` 매핑 로직
  - `open`→`mmap(PROT_READ, MAP_SHARED)`로 바이너리 read-only 매핑
  - `MappingOwner`가 fd와 매핑 포인터 수명 관리(`munmap`, `close`)
  - JSON 파일은 일반 파일 I/O로 로드

- `qnn_io_plan_main.cpp` 구동 순서
  - SO 경로 기본값: 현 디렉토리의 `./libQnnHtp.so`, `./libQnnSystem.so`
  - `--ctx_dir` 스캔: `forward_<i>.bin`과 `forward_<i>_json.json` 페어 찾기(인덱스 순)
  - 각 페어마다: 컨텍스트 복원 → 두 그래프명(`prefill_forward`, `kv_forward`) 시도 → 그래프별로 JSON I/O 메타 로드 → per‑tensor 메모리 할당 → `Qnn_Tensor_t` 구성 → `graphExecute`
  - 불필요 예시 출력 제거, 실패 시 에러 로그 후 다음 페어 진행


## 구현 철학(왜 이렇게 했는가)

- Executorch의 실제 흐름과 API 사용을 그대로 따릅니다.
  - 컨텍스트 복원은 `qnn_context_create_from_binary` 사용(멀티-그래프 컨텍스트 지원)
  - 그래프는 `graphRetrieve`로 핸들만 가져오고, 그래프의 등록 텐서(내부적으로 이미 정의된 텐서 ID/형상/타입)에 대해 애플리케이션이 `clientBuf`만 채워서 `graphExecute`를 호출
  - `QnnTensor_updateGraphTensors` 같은 UPDATEABLE 전용 경로는 제거(입출력 I/O는 APP_* 타입이므로 업데이트 대상이 아님)
- I/O 메타데이터의 단일 소스로 “QNN SDK JSON”을 사용합니다.
  - Executorch의 런타임 덤프 JSON은 참고용이며, 실제 바인딩은 `forward_{i}_json.json` 기준으로 수행
  - JSON에 `bytesPerElement`가 있으면 그대로 신뢰, 없으면 `dataType` 기준으로 계산
- 샤딩(복수 컨텍스트) 환경에서도 컨텍스트별로 동일한 절차를 반복합니다.
  - 샤드 간 자동 메모리 공유는 없습니다. 동일 버퍼 주소를 여러 컨텍스트에 명시적으로 바인딩하거나(QNN MEMHANDLE/ION 등) 애플리케이션 레벨에서 주소를 재사용해야 합니다.


## QNN 초기화 및 실행 흐름(Executorch 미러링)

1) QNN 라이브러리 로딩 및 Provider 획득
   - `libQnnSystem.so`, `libQnnHtp.so`를 `dlopen`
   - `QnnInterface_getProviders`로 provider 리스트 획득 → 첫 번째 provider 선택(Executorch 기본 동작과 동일)

2) Backend/Device 생성
   - `backendCreate`, `deviceCreate` 순서로 생성
   - 로깅은 QNN 기본 로거를 사용(안드로이드는 logcat, Host는 stderr). `--log_level`로 레벨 설정

3) 컨텍스트 복원
   - 각 샤드의 `forward_<i>.bin`을 `mmap`
   - `qnn_context_create_from_binary(backend, device, config, binary, nbytes, &context, profile=nullptr)` 호출

4) 그래프 핸들 조회
   - `graphRetrieve(context, graphName, &graphHandle)`
   - 본 구현은 `prefill_forward`, `kv_forward` 두 그래프명을 기본 사용(필요 시 확장 가능)

5) I/O 할당 및 텐서 바인딩
   - `forward_<i>_json.json`을 파싱하여 그래프별 입력/출력 텐서 메타 획득
   - 텐서마다 `bytesPerElement`와 `dims`로 `nbytes` 계산(아래 공식 및 매핑 참고)
   - 각 텐서에 맞는 크기의 호스트 메모리(정렬 선택적) 할당
   - `QnnTensorHolder`를 통해 `Qnn_Tensor_t v2`를 구성하고 `clientBuf.data/.dataSize` 채움
   - 입력 텐서 `type=APP_WRITE`, 출력 텐서 `type=APP_READ`

6) 실행
   - `graphExecute(graphHandle, inputs.data(), inputs.size(), outputs.data(), outputs.size(), /*profile*/nullptr, /*signal*/nullptr)`


## JSON 파싱 규칙(중요)

## JSON 파일 출처와 생성법

- 출처: QNN SDK의 `qnn-context-binary-utility`가 컨텍스트 바이너리에서 그래프/텐서 메타를 추출해 생성한 JSON입니다.
- 전제: 각 컨텍스트 바이너리(`forward_<i>.bin`)는 멀티-그래프(예: `prefill_forward`, `kv_forward`)를 포함할 수 있습니다.
- 일반적 사용 예시(환경/버전에 따라 옵션명이 다를 수 있으니 SDK 문서를 우선 확인하세요):

```bash
# QNN SDK 설치 경로 예시: $QNN_SDK/bin/x86_64-linux-clang
$QNN_SDK/bin/x86_64-linux-clang/qnn-context-binary-utility \
  --input forward_0.bin \
  --output forward_0_json.json \
  --dump_graph_io
```

- 결과: `forward_0_json.json` 내에 컨텍스트가 가진 그래프들의 I/O 텐서 정보(`id`, `name`, `dims`, `dataType`, `bytesPerElement`, `quantization` 등)가 포함됩니다.
- 본 프로젝트는 해당 JSON을 “단일 진실 소스(source of truth)”로 사용하여, Executorch 런타임 덤프(JSON)는 참고용으로만 봅니다.

- JSON 파일: `ctx_dir` 내 `forward_<i>_json.json`
- 그래프별로 `inputs`, `outputs` 섹션이 있고, 각 텐서는 최소한 다음 정보를 제공합니다.
  - `id`(정수), `name`(문자열), `dims`(정수 배열: `dimensions` 또는 `currentDimensions`), `dataType`(문자열 또는 숫자 코드)
  - `bytesPerElement`(있으면 그대로 사용), `nbytes`(있으면 참고하되, 없거나 불확실하면 재계산)
  - `quantization`(optional): `quantizationEncoding`과 per-tensor/per-axis 스펙(자세한 매핑은 아래 참조)
- 파서 동작 원칙
  - `bytesPerElement` 키가 존재하면 이를 신뢰
  - 없으면 `dataType` → 바이트 수 매핑으로 `bytesPerElement`를 도출
  - `nbytes` 최종값은 `bytesPerElement × ∏(dims)`로 계산
  - `dimensions`와 `currentDimensions`가 동시에 있을 경우, 실제 런타임 차원을 반영하는 `currentDimensions`를 우선 고려(파일 구조에 따라 상이할 수 있어 양쪽 대응)


## dataType → bytesPerElement 매핑(세부)

아래 매핑은 QNN SDK 2.37 환경과 본 프로젝트의 JSON 관찰값을 기반으로 합니다. JSON이 `bytesPerElement`를 제공하면 그 값을 우선 사용합니다. 제공하지 않는 경우에 한해 아래 매핑으로 도출합니다.

- 정수/부호 없는 정수
  - `QNN_DATATYPE_BOOL_8`: 1 byte
  - `QNN_DATATYPE_UINT_8` / `QNN_DATATYPE_INT_8`: 1 byte
  - `QNN_DATATYPE_UINT_16` / `QNN_DATATYPE_INT_16`: 2 bytes
  - `QNN_DATATYPE_UINT_32` / `QNN_DATATYPE_INT_32`: 4 bytes
  - `QNN_DATATYPE_UINT_64` / `QNN_DATATYPE_INT_64`: 8 bytes
- 부동소수점
  - `QNN_DATATYPE_FLOAT_16`: 2 bytes
  - `QNN_DATATYPE_FLOAT_32`: 4 bytes
- 고정소수점(예: JSON 숫자 코드 기준 관찰)
  - 코드 `1032`(예: `UFIXED_8` 계열): 1 byte
  - 코드 `1046`(예: `UFIXED_16` 계열): 2 bytes
- 기타(필요 시 확장): QNN SDK가 추가로 정의한 타입은 문서/헤더에 따라 바이트 수를 매핑

주의: 일부 JSON은 `dataType`을 문자열이 아닌 정수 코드로 표기합니다. 본 구현은 문자열/정수 모두 처리하며, 코드값이 알려진 범위를 벗어나면 보수적으로 실패를 로그로 알리거나, 안전한 기본값(예: 1 byte)으로 처리 후 경고를 남기는 방식으로 방어할 수 있습니다(현재 코드는 알려진 코드값에 대해 엄격 매핑).


## nbytes 계산 규칙

- 기본 공식: `nbytes = bytesPerElement × ∏(dims)`
- `dims`는 `size_t`로 누적 곱하며, 0 또는 음수(잘못된 값)가 감지되면 보정/에러를 보고합니다.
- `bytesPerElement`가 0이 되는 경우는 없어야 하며, 0이면 바인딩 실패(`clientBuf.dataSize==0`)로 이어집니다.
- JSON에 `nbytes`가 제공되더라도 상충 시 재계산값을 신뢰하도록 구현할 수 있습니다(본 구현은 JSON bpe 값을 신뢰하고, 없을 때 계산하여 사용). 


## Quantization 매핑(현재 상태)

- per-tensor(scale/offset) 형식
  - JSON이 `quantizationEncoding=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET`를 제공하고 `scale`/`offset`를 포함하면 `Qnn_QuantizeParams_t`의 `scaleOffsetEncoding`에 반영
  - 부호/비트폭은 JSON 메타에 맞게 보존(필요 시 확장)
- per-axis(axis-scale-offset) 형식
  - 현재는 `quantizationEncoding`과 `axis`/`bitwidth` 같은 메타만 보존하며, 실제 스케일/오프셋 배열 포인터 연결은 미구현(향후 확장: 별도 버퍼를 준비하고 `axisScaleOffsetEncoding`에 배열 포인터 연결)
- quant가 없거나 `UNDEFINED`인 경우
  - `quantizeParams.encodingDefinition`을 무효 또는 기본값으로 설정하여 비양자화 텐서로 취급


## 텐서 구성과 바인딩(핵심 API 필드)

- `Qnn_Tensor_t`(v2) 구성
  - `v2.id` = JSON의 텐서 `id`
  - `v2.name` = JSON의 텐서 `name`
  - `v2.type` = 입력: `QNN_TENSOR_TYPE_APP_WRITE`, 출력: `QNN_TENSOR_TYPE_APP_READ`
  - `v2.dataFormat` = `QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER`(연속 버퍼)
  - `v2.memType` = `QNN_TENSORMEMTYPE_RAW`(애플리케이션 제공 버퍼)
  - `v2.dimensions` = JSON `dims`
  - `v2.dataType` = JSON `dataType`(문자열/정수코드 → QNN enum으로 매핑)
  - `v2.clientBuf.data` = malloc/new/메모리 풀/ION 등으로 확보한 유효 주소
  - `v2.clientBuf.dataSize` = 위 `nbytes` 값(32-bit 필드인 경우 캐스팅 주의)
  - `v2.quantizeParams` = 상기 매핑 규칙에 따라 설정

- Executorch와 동일한 실행 방식
  - “등록된 텐서(그래프 내부 정의)의 `id`에 맞춰” `clientBuf`만 꽂아서 `graphExecute`
  - `QnnTensor_updateGraphTensors`는 I/O(APP_*)용이 아니므로 사용하지 않음


## 실행 방법

### 빌드

프로젝트 루트에서 CMake/Ninja 등을 사용해 빌드합니다. Android NDK 교차빌드도 지원합니다.

```bash
# 예시(Host)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build

# 예시(Android)
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DCMAKE_BUILD_TYPE=Release
ninja -C build-android
```

### 실행

`qnn_io_plan` 앱은 컨텍스트와 JSON이 함께 들어있는 디렉토리에서 페어 파일을 찾아 순차 처리합니다.

필수/옵션 인자:

- `--ctx_dir <PATH>`: 컨텍스트 및 JSON 페어가 들어있는 디렉토리
- `--backend_so <PATH>`: QNN Backend SO 경로(기본: 현재 디렉토리의 `./libQnnHtp.so`)
- `--system_so <PATH>`: QNN System SO 경로(기본: 현재 디렉토리의 `./libQnnSystem.so`)
- `--log_level <N>`: QNN 로그 레벨(0=ERROR, 1=WARN, 2=INFO, 3=DEBUG 등 SDK 정의에 따름)
- `--align <N>`: I/O 버퍼 정렬 바이트(선택, 기본 무정렬 또는 구현 기본값)
- `--bind_plan_out <PATH>`: (선택) 바인딩 계획을 JSON으로 덤프

예시:

```bash
# Host 예시
./build/apps/qnn_io_plan \
  --ctx_dir /home/user/ctx_verify \
  --log_level 3 \
  --backend_so ./libQnnHtp.so \
  --system_so ./libQnnSystem.so

# Android 예시(adb shell 내부)
./qnn_io_plan \
  --ctx_dir /data/local/tmp/ctx_verify \
  --log_level 3 \
  --backend_so ./libQnnHtp.so \
  --system_so ./libQnnSystem.so
```


## 샤딩(멀티 컨텍스트) 처리

- `ctx_dir`에 `forward_0.bin` … `forward_7.bin`과 각 `forward_i_json.json`이 페어로 존재한다고 가정
- 앱은 인덱스 순으로 각 컨텍스트를 독립적으로 복원하고, 동일한 그래프명(`prefill_forward`, `kv_forward`)을 대상으로 I/O 바인딩 및 실행
- 샤드 간 자동 버퍼 공유는 없으며, 동일 주소(혹은 QNN MEMHANDLE/ION) 바인딩으로 “의도적 공유”를 구현할 수 있음


## 메모리 할당/정렬

본 프로젝트의 메모리 할당은 “JSON 기반의 정적 바인딩”을 기본으로 하며, Executorch의 `AllocateTensor`가 하는 역할(등록된 텐서의 clientBuf만 채움)을 앱 레벨에서 재현합니다. 세부 동작은 다음과 같습니다.

1) nbytes 산출
   - 각 텐서별 `bytesPerElement`(JSON 제공 또는 dtype 매핑)와 `dims`의 곱으로 `nbytes`를 계산합니다.
   - 32-bit overflow를 피하기 위해 누적 곱은 `size_t`/`uint64_t`로 수행 후, `clientBuf.dataSize`에 대입 시 `uint32_t` 캐스팅을 수행합니다. 캐스팅 전 상한 검증으로 4GiB 초과 시 경고/에러를 낼 수 있습니다.

2) 할당기 정책
   - 기본: 각 텐서마다 독립적인 호스트 버퍼를 `new uint8_t[nbytes]` 또는 `std::aligned_alloc(align, rounded_size)`로 확보합니다.
   - 정렬: `--align` 인자를 통해 정렬 바이트를 지정할 수 있습니다(예: 64/128). 지정 시, `rounded_size = AlignUp(nbytes, align)`을 적용하여 오버런 없이 안전하게 할당합니다.
   - 수명: 입력/출력 텐서 버퍼는 그래프 실행이 끝날 때까지 유효해야 합니다. 현재 예제는 단일 호출 단위로 수명을 관리합니다. 반복 실행 시에는 재사용 풀(캐시)을 둘 수 있습니다.

3) 메모리 소유권과 클린업
   - `QnnTensorHolder`는 텐서 구조체와 별개로 버퍼 포인터만 참조합니다. 실제 소유권은 앱이 보유하며, 실행 종료 시 `delete[]` 혹은 `std::free`로 해제합니다(정렬 방법에 따라 일치하는 해제 함수 사용).
   - `mmap`은 컨텍스트 바이너리(읽기 전용)에만 사용합니다. I/O 텐서 버퍼는 일반 힙 할당을 사용합니다.

4) 멀티 텐서 연속 할당(옵션)
   - 성능과 캐시 친화성을 위해 여러 텐서를 하나의 큰 컨티구어스 블록에 패킹할 수 있습니다.
   - 절차: 총 바이트 수 = ∑ AlignUp(nbytes_i, align)로 계산 → 1회 대형 할당 → 각 텐서에 오프셋으로 슬라이스를 나누어 `clientBuf.data`에 지정.
   - 장점: `malloc`/`free` 호출 감소, TLB/캐시 효율 개선. 단점: 단일 텐서 재할당이 어려움.

5) 샤드 간/그래프 간 공유(심화)
   - 기본 구현은 샤드 간 자동 공유를 하지 않습니다. 동일 주소(or MEMHANDLE)를 명시적으로 바인딩해야 합니다.
   - 향후 ION/DMABUF 통합 시, 공통 ION 버퍼를 생성하고 각 샤드/그래프의 공통 I/O에 동일 핸들+오프셋을 바인딩함으로써 진정한 zero‑copy를 구현할 수 있습니다.
   - Executorch 유사 경로: `QnnManager::RegisterIonMem`/`RegisterCustomMem`를 참고하여 FD→memhandle 등록 후, 텐서 `memType=QNN_TENSORMEMTYPE_MEMHANDLE`로 설정 및 `clientBuf` 대신 핸들/오프셋을 사용합니다.

6) 동적 크기 텐서(옵션)
   - 동적 차원을 갖는 입력의 경우, 실행 전에 실제 `dims`로 `nbytes`를 재계산하고, 버퍼를 재할당하거나 충분한 상한 크기의 버퍼를 재사용합니다.
   - Executorch는 `MethodMeta` 기반으로 이를 관리합니다. 본 구현은 JSON+런타임 입력으로 치환하며, 차원/바이트 검증 로직을 추가해 안전성을 확보할 수 있습니다.

7) 보안/안정성
   - `nbytes==0` 또는 과대값(예: 비정상적으로 큰 값)인 경우 즉시 실패하도록 방어 로직을 두었습니다.
   - `memset`(0) 초기화 여부는 워크로드에 따라 선택합니다. 출력 텐서는 초기화 없이 상관없는 경우가 많지만, 디버깅 용이성을 위해 초기화를 권장할 수 있습니다.

8) 정렬 기준 추천치
   - CPU 측 캐시라인 정렬: 64B
   - HTP/버스 전송 친화: 64~128B(플랫폼에 따라 상이)
   - ION/DMABUF 사용 시, 페이지 정렬(4KiB) 또는 드라이버 제약을 우선 고려합니다.

향후 확장: ION/DMABUF 기반 MEMHANDLE 경로를 추가하여 NPU와의 zero‑copy를 실현하고, 텐서 묶음(특히 샤드 간 동일 I/O)에 대해 공유 핸들을 바인딩하는 플래너를 도입할 수 있습니다.


## 로깅

- QNN의 기본 로거 사용(안드로이드: logcat, Host: stderr)
- `--log_level`로 런타임에서 조절 가능
- 불필요한 예시/검증 출력은 제거하여 로그 간결화


## 문제 해결 가이드(Troubleshooting)

## 자주 하는 질문(FAQ)

- Q) 그래프 이름을 JSON에서 자동으로 가져오나요?
  - A) 현재 예제는 `prefill_forward`, `kv_forward`를 기본값으로 사용합니다. 필요 시 JSON에서 그래프명 목록을 파싱해 확장 가능합니다.

- Q) 샤드 간 버퍼를 자동 공유하나요?
  - A) 아니요. 동일 주소 바인딩(혹은 MEMHANDLE/ION)으로 명시적으로 공유해야 합니다. 자동 공유는 동일 컨텍스트 내의 mutable buffer 설계에서만 동작합니다(본 환경은 -1로 미사용).

- Q) 왜 `QnnTensor_updateGraphTensors`를 쓰지 않나요?
  - A) I/O(APP_*) 텐서는 업데이트 대상이 아니기 때문입니다. Executorch도 등록 텐서에 `clientBuf`만 채워 실행합니다.

- Q) `bytesPerElement`가 JSON에 없을 때 어떻게 하나요?
  - A) `dataType`을 문자열/정수 코드로 해석해 매핑 테이블로 도출합니다. 그 후 `nbytes = bpe × ∏dims`로 계산합니다.

- 오류: `Expected Tensor ID: #### not found in user-provided tensors.`
  - 원인: `Qnn_Tensor_t.v2.id`가 그래프 등록 텐서 ID와 불일치
  - 조치: JSON에서 제공하는 ID를 그대로 넣었는지 확인. 입력/출력 텐서 수와 순서도 점검

- 오류: `clientBuf is null` 또는 `dataSize == 0`
  - 원인: nbytes 과소 계산 또는 미할당
  - 조치: `bytesPerElement`/`dims` 파싱 재확인, JSON의 `bytesPerElement`가 없으면 dtype 매핑으로 도출. `nbytes = bpe × ∏dims` 검증

- 오류: `graphExecute` 실패(일반)
  - 원인: 텐서 타입/포맷/차원/dtype/quant 불일치
  - 조치: `type(APP_WRITE/APP_READ)`, `dataFormat(FLAT_BUFFER)`, `memType(RAW)`, `dims`·`dataType`·`quant`를 JSON에 맞게 엄격히 구성했는지 확인

- Android에서 로그가 보이지 않음
  - QNN 기본 로거는 logcat으로 출력. `adb logcat | grep Qnn` 등으로 확인. 호스트에서는 stderr


## 제한 사항 및 확장 계획

- 현재 per-axis quant는 메타만 보존하며 스케일/오프셋 배열 포인터 연결은 미구현(확장 예정)
- OPTIONAL I/O(마스크/특정 접두사)의 자동 제외 규칙은 선택 기능으로 추후 적용 가능
- 샤드 간 zero-copy는 자동이 아님. 동일 주소/ION 핸들 바인딩으로 의도적으로 구성 필요


## 설계 결정 요약(Executorch 대비)

- 동일: `contextCreateFromBinary` → `graphRetrieve` → 등록 텐서에 clientBuf만 채워 `graphExecute`
- 동일: provider 첫 항목 선택, 기본 로거 사용, 런타임 로그 레벨 조절
- 차이: I/O 메타데이터는 Executorch의 내부 MethodMeta 대신 QNN SDK JSON을 “단일 소스”로 사용
- 차이: 자동 mutable buffer 공유는 사용하지 않음(-1 기본). 필요 시 앱 레벨에서 버퍼 재사용 또는 MEMHANDLE 사용

## Executorch 구현과의 상세 비교(코드 기준)

- 라이브러리/Provider 로딩
  - Executorch: `backends/qualcomm/runtime/QnnImplementation.cpp`의 `QnnImplementation::StartBackend()`에서 `QnnInterface_getProviders` 호출 후 첫 provider 선택
  - 본 구현: 동일하게 `QnnInterface_getProviders`로 provider 배열을 받아 첫 항목 사용

- Backend/Device/Context 생성
  - Executorch: `QnnManager::Init()`에서 backend/device/context를 순차 구성. 컨텍스트는 `QnnContextCommon::Configure()`에서 캐시 상태에 따라
    - DESERIALIZE: `qnn_context_create_from_binary`
    - SERIALIZE/ONLINE_PREPARE/MULTI_GRAPH: `qnn_context_create` 후 그래프 등록 경로
  - 본 구현: “DESERIALIZE” 경로만 사용. 즉, 항상 `qnn_context_create_from_binary`로 복원(멀티-그래프 포함 단일 바이너리)

- 컨텍스트 바이너리 소스
  - Executorch: PTE 내부에 임베드된 QNN 컨텍스트 바이너리를 `QnnExecuTorchContextBinary`로 추출·보관(`QnnExecuTorch.h`, `QnnContextCommon::GetContextBinary()`)
  - 본 구현: 외부에서 추출된 `forward_<i>.bin` 파일을 그대로 사용(Python 추출 스크립트 등 별도 경로)

- 그래프 이름/검색
  - Executorch: `MethodMeta` 등 내부 메타에서 그래프 목록을 보유하고 순회 초기화
  - 본 구현: `prefill_forward`, `kv_forward`를 기본 상수로 사용. 필요 시 JSON에서 그래프명 파싱으로 확장 가능(현재는 상수)

- I/O 텐서 메타데이터 출처
  - Executorch: PTE의 `MethodMeta`·런타임 내부 캐시에서 입출력/업데이트 텐서 정보를 관리
  - 본 구현: QNN SDK `qnn-context-binary-utility` JSON을 “단일 소스”로 사용. Executorch 런타임 덤프 JSON은 참고용

- 텐서 바인딩 방식
  - Executorch: 그래프에 등록된 텐서 집합을 보유한 상태에서 “`AllocateTensor`가 각 텐서의 clientBuf만 채움”. 이후 `graphExecute`
  - 본 구현: JSON의 `id`를 그대로 사용해 `Qnn_Tensor_t v2`를 구성하고 `clientBuf`만 채워 `graphExecute`. `QnnTensor_updateGraphTensors`는 사용하지 않음(입출력은 APP_* 타입)

- Quantization 처리
  - Executorch: `MethodMeta` 기반으로 per-tensor/per-axis를 세밀하게 구성(필요 시 내부 버퍼 보관)
  - 본 구현: per-tensor(scale/offset) 즉시 반영, per-axis는 encoding 메타까지만(실제 배열 포인터 연결은 미구현)

- Mutable Buffer/공유 메모리
  - Executorch: 동일 컨텍스트 내 mutable buffer id를 기반으로 버퍼 공유(필요 시 ION/DMABUF 등록 경로 보유: `QnnManager::RegisterIonMem` 등)
  - 본 구현: 기본적으로 per‑tensor host 메모리 할당. 샤드 간 자동 공유 없음. 향후 ION/DMABUF 연계는 확장 포인트

- 프로파일/시그널
  - Executorch: 선택적으로 프로파일 핸들/시그널을 사용
  - 본 구현: 단순화를 위해 `nullptr` 사용

- 로깅
  - Executorch: `QnnLogger` 래퍼와 환경 변수/옵션을 통한 제어
  - 본 구현: QNN 기본 로거 사용, `--log_level`로 최소 제어(콜백 미사용)

- 에러 처리/검증
  - Executorch: `ET_CHECK_OR_RETURN_ERROR` 매크로 등으로 일관 처리
  - 본 구현: 함수별 반환 검사 및 에러 로그 출력(간결화)

- 샤딩 운영 모델
  - Executorch: 실행 시 내부 스케줄에 따라 샤드 전환·재사용, 캐시 상태 관리
  - 본 구현: `ctx_dir` 순회로 각 컨텍스트를 순차 복원·실행. 필요 시 수명 연장/재사용 가능하나 기본은 순차 처리



## 예시 확인(정상 동작 시)

- 타입/차원 예시:
  - `UFIXED_8` 계열(bpe 1), `dims=[1,32,128]` → `nbytes=4096`
  - `UFIXED_16` 계열(bpe 2), `dims=[1,32,4096]` → `nbytes=262144`
  - `FLOAT_32`(bpe 4), `dims=[32,64]` → `nbytes=8192`
  - 위 값들은 Executorch 런타임 덤프(참고용)와 일관되게 계산됨


## 참고 파일

- 로더: `llm_test/include/qnn_loader.h`, `llm_test/src/qnn_loader.cpp`
- JSON 파서: `llm_test/include/qnn_qnnjson.h`, `llm_test/src/qnn_qnnjson.cpp`
- 텐서 유틸: `llm_test/include/qnn_tensor_util.h`, `llm_test/src/qnn_tensor_util.cpp`
- 앱: `llm_test/apps/qnn_io_plan_main.cpp`


## 라이선스/주의

- QNN SDK의 사용 및 재배포 조건은 Qualcomm의 라이선스 조항을 따릅니다.
- 본 코드는 QNN SDK 2.37.x 환경에서 검증되었습니다. 다른 버전에서는 타입 코드/필드가 다를 수 있으니 헤더/문서를 확인하세요.


