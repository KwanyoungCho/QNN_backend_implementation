cmake_minimum_required(VERSION 3.16)

project(llm_qnn_init LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK root; allow override via env or cache
if(NOT DEFINED QNN_SDK_ROOT)
  if(DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
  else()
    set(QNN_SDK_ROOT "/home/chokwans99/QNN_SDK/qairt/2.37.1.250807")
  endif()
endif()

message(STATUS "Using QNN_SDK_ROOT=${QNN_SDK_ROOT}")

add_library(qnn_ctx_core STATIC
  src/qnn_loader.cpp
  src/binary_provider.cpp
  src/io_alloc.cpp
  src/qnn_qnnjson.cpp
  src/qnn_tensor_util.cpp
  src/llm_input_preparer.cpp
  src/llm_output_processor.cpp
  src/llm_kv_cache_manager.cpp
)

add_executable(qnn_ctx_init
  apps/qnn_ctx_init_main.cpp
)

target_include_directories(qnn_ctx_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${QNN_SDK_ROOT}/include/QNN
)

target_link_libraries(qnn_ctx_core PUBLIC dl)

if(ANDROID)
  target_link_libraries(qnn_ctx_core PUBLIC log)
else()
  target_link_libraries(qnn_ctx_core PUBLIC pthread)
endif()

target_link_libraries(qnn_ctx_init PRIVATE qnn_ctx_core)

add_executable(qnn_graph_probe
  apps/qnn_graph_probe_main.cpp
)
target_link_libraries(qnn_graph_probe PRIVATE qnn_ctx_core)

add_executable(qnn_io_plan
  apps/qnn_io_plan_main.cpp
)
target_link_libraries(qnn_io_plan PRIVATE qnn_ctx_core)

add_executable(qnn_prefill
  apps/qnn_prefill_main.cpp
)
target_link_libraries(qnn_prefill PRIVATE qnn_ctx_core tok_llama)

add_executable(qnn_decode
  apps/qnn_decode_main.cpp
)
target_link_libraries(qnn_decode PRIVATE qnn_ctx_core tok_llama)


# llama.cpp tokenizer wrapper and example
# Build llama.cpp (specinfer.cpp fork) as subproject
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(/home/chokwans99/dev/llm/specinfer.cpp ${CMAKE_BINARY_DIR}/third_party/specinfer-build)

add_library(tok_llama STATIC src/tokenizer_llama.cpp)

target_include_directories(tok_llama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(tok_llama PUBLIC llama)
if(ANDROID)
  target_link_libraries(tok_llama PUBLIC log)
endif()

add_executable(tok_encode
  apps/tok_encode_main.cpp
)
target_link_libraries(tok_encode PRIVATE tok_llama)

