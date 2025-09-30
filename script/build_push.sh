cd /home/chokwans99/llm_test
cmake -B build-android \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -D ANDROID_ABI=arm64-v8a \
  -D ANDROID_PLATFORM=android-30 \
  -D QNN_SDK_ROOT=$QNN_SDK_ROOT
cmake --build build-android -j 99

adb push build-android/qnn_ctx_init /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_graph_probe /data/local/tmp/chokwans99/executorch/QNN_test
adb push build-android/qnn_io_plan /data/local/tmp/chokwans99/executorch/QNN_test

adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_ctx_init'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_graph_probe'
adb shell 'chmod +x /data/local/tmp/chokwans99/executorch/QNN_test/qnn_io_plan'