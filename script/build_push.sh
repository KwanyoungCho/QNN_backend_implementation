cd /home/jongjip/QNN_backend_implementation
rm -rf build-android
cmake -B build-android \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -D ANDROID_ABI=arm64-v8a \
  -D ANDROID_PLATFORM=android-30 \
  -D QNN_SDK_ROOT=$QNN_SDK_ROOT
cmake --build build-android -j 99

adb shell 'mkdir -p /data/local/tmp/jongjip/executorch/QNN_test'
adb push build-android/qnn_llm_generate /data/local/tmp/jongjip/executorch/QNN_test

adb push build-android/bin/libllama.so /data/local/tmp/jongjip/executorch/QNN_test
adb push build-android/bin/libggml.so /data/local/tmp/jongjip/executorch/QNN_test
adb push build-android/bin/libggml-cpu.so /data/local/tmp/jongjip/executorch/QNN_test
adb push build-android/bin/libggml-base.so /data/local/tmp/jongjip/executorch/QNN_test

adb shell 'chmod +x /data/local/tmp/jongjip/executorch/QNN_test/qnn_llm_generate'