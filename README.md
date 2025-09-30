## qnn_ctx_init

Executorch 경로를 따라 QNN API로 컨텍스트 바이너리(단일/다중)를 복구/초기화하는 최소 예제. 코어 로직은 라이브러리(`qnn_ctx_core`), 실행 엔트리는 `apps/qnn_ctx_init_main.cpp`(바이너리: `qnn_ctx_init`).

### 요구사항
- QNN SDK: `QNN_SDK_ROOT` 환경변수 지정 (예: `/home/chokwans99/QNN_SDK/qairt/2.37.1.250807`)
- Android NDK (on‑device 실행용 교차 빌드)

### 빌드 (Host Linux)
```bash
cd /home/chokwans99/llm_test
cmake -B build -DQNN_SDK_ROOT=$QNN_SDK_ROOT
cmake --build build -j 99
```

### 실행 (Host Linux)
```bash
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH \
  ./build/qnn_ctx_init \
  --ctx_file /path/to/ctx_all.bin \
  --backend_so $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --system_so $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnSystem.so \
  --log_level 5
```

### Android(NDK, arm64) 빌드
```bash
cd /home/chokwans99/llm_test
cmake -B build-android \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -D ANDROID_ABI=arm64-v8a \
  -D ANDROID_PLATFORM=android-30 \
  -D QNN_SDK_ROOT=$QNN_SDK_ROOT
cmake --build build-android -j 99
```

### Android 기기에서 실행(HTP)
```bash
adb push build-android/qnn_ctx_init /data/local/tmp/
adb shell mkdir -p /data/local/tmp/ctx_out
adb push /home/chokwans99/tmp/executorch/ctx_out/*.bin /data/local/tmp/ctx_out/

# 단일 바이너리 경로
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && \
  /data/local/tmp/qnn_ctx_init \
  --ctx_file /data/local/tmp/ctx_all.bin \
  --backend_so libQnnHtp.so \
  --system_so libQnnSystem.so \
  --log_level 4'

# 샤드 폴더에서 여러 컨텍스트 생성
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && \
  /data/local/tmp/qnn_ctx_init \
  --ctx_dir /data/local/tmp/ctx_out \
  --backend_so libQnnHtp.so \
  --system_so libQnnSystem.so \
  --log_level 3'
```

### 실행 인자
- `--ctx_file FILE`: 단일 컨텍스트 바이너리로 1개 컨텍스트 복원(권장)
- `--ctx_dir DIR`: `*_0.bin..*_N.bin`을 찾아 각 파일로 컨텍스트를 여러 개 생성
- `--backend_so PATH`, `--system_so PATH`: QNN SO 경로(안드로이드에선 파일명만 주면 현재 디렉터리에서 로드)
- `--provider NAME`(선택): 특정 provider 선택
- `--log_level N`: 1=ERROR, 2=WARN, 3=INFO, 4=VERBOSE, 5=DEBUG (기본 5)

### 구현 포인트(Executorch와 동일 흐름)
- `QnnInterface_getProviders` → 함수 테이블 사용
- `backendCreate` → `deviceCreate` → `contextCreateFromBinary`
- STDOUT 콜백 로깅 활성화(쉘에서 바로 로그 확인 가능)

### adb shell 내부 실행
/data/local/tmp/qnn_ctx_init \
  --ctx_dir /data/local/tmp/ctx_out \
  --backend_so libQnnHtp.so \
  --system_so libQnnSystem.so \
  --log_level 3