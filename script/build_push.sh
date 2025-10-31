cd /home/chokwans99/llm_test
rm -rf build-android
cmake -B build-android \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -D ANDROID_ABI=arm64-v8a \
  -D ANDROID_PLATFORM=android-30 \
  -D QNN_SDK_ROOT=$QNN_SDK_ROOT
cmake --build build-android -j 99

adb shell 'mkdir -p /data/local/tmp/chokwans99/executorch/QNN_test'
adb push build-android/qnn_ctx_init /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_graph_probe /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_io_plan /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/tok_encode /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_prefill /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_decode /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_llm_generate /data/local/tmp/chokwans99/executorch/QNN_test

adb push build-android/bin/libllama.so /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/bin/libggml.so /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/bin/libggml-cpu.so /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/bin/libggml-base.so /data/local/tmp/chokwans99/executorch/QNN_test

adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_ctx_init'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_graph_probe'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_io_plan'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/tok_encode'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_prefill'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_decode'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_llm_generate'