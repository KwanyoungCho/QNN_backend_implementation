#include "qnn_loader.h"

#include <QnnInterface.h>
#include <QnnLog.h>
#include <cstdarg>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>

namespace llm_test {

namespace {
// QNN 로그를 표준출력으로 바로 내보내는 콜백
static void StdoutLogger(const char* fmt,
                         QnnLog_Level_t level,
                         uint64_t /*timestamp*/,
                         va_list args) {
  (void)level; // 레벨은 필요 시 필터링에 사용할 수 있음
  vfprintf(stdout, fmt, args);
  fputc('\n', stdout);
  fflush(stdout);
}
}

// 소멸자: 생성된 리소스들을 안전하게 해제
QnnLoader::~QnnLoader() { cleanup(); }

// QNN 공유 라이브러리 로드 및 provider 조회 함수 심볼 확인
bool QnnLoader::load(const std::string& backend_so_path, const std::string& system_so_path) {
  handles_.backend_so_handle = dlopen(backend_so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handles_.backend_so_handle) {
    std::cerr << "dlopen backend failed: " << dlerror() << "\n";
    return false;
  }
  handles_.system_so_handle = dlopen(system_so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handles_.system_so_handle) {
    std::cerr << "dlopen system failed: " << dlerror() << "\n";
    return false;
  }

  get_providers_fn_ = dlsym(handles_.backend_so_handle, "QnnInterface_getProviders");
  if (!get_providers_fn_) {
    get_providers_fn_ = dlsym(handles_.system_so_handle, "QnnInterface_getProviders");
  }
  if (!get_providers_fn_) {
    std::cerr << "Cannot resolve QnnInterface_getProviders: " << dlerror() << "\n";
    return false;
  }
  return true;
}

bool QnnLoader::get_graph_io(size_t ctx_index,
                             const std::string& graph_name,
                             std::vector<Qnn_Tensor_t>& inputs,
                             std::vector<Qnn_Tensor_t>& outputs) {
  if (!interface_provider_) return false;
  if (ctx_index >= contexts_.size()) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  if (!api.graphRetrieve) return false;
  void* graph = nullptr;
  if (api.graphRetrieve(reinterpret_cast<Qnn_ContextHandle_t>(contexts_[ctx_index]),
                        graph_name.c_str(),
                        reinterpret_cast<Qnn_GraphHandle_t*>(&graph)) != QNN_SUCCESS || !graph) {
    return false;
  }
  // 표준 C API로는 I/O 디스크립터를 직접 열람할 수 없습니다. 여기서는 두 벡터를 비워둡니다.
  // 주 호출자는 JSON 기반으로 Qnn_Tensor_t 배열을 구성한 뒤 update_graph_tensors로 등록 텐서를 갱신합니다.
  inputs.clear(); outputs.clear();
  return true;
}

bool QnnLoader::update_graph_tensors(size_t ctx_index,
                                     const std::string& graph_name,
                                     const std::vector<Qnn_Tensor_t>& tensors) {
  if (!interface_provider_) return false;
  if (ctx_index >= contexts_.size()) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  if (!api.graphRetrieve || !api.tensorUpdateGraphTensors) return false;
  void* graph = nullptr;
  if (api.graphRetrieve(reinterpret_cast<Qnn_ContextHandle_t>(contexts_[ctx_index]),
                        graph_name.c_str(),
                        reinterpret_cast<Qnn_GraphHandle_t*>(&graph)) != QNN_SUCCESS || !graph) {
    return false;
  }
  // tensorUpdateGraphTensors는 등록된 텐서 ID를 키로 clientBuf/quant 등을 갱신한다.
  std::vector<const Qnn_Tensor_t*> ptrs;
  ptrs.reserve(tensors.size());
  for (const auto& t : tensors) ptrs.push_back(&t);
  Qnn_ErrorHandle_t err = api.tensorUpdateGraphTensors(
      reinterpret_cast<Qnn_GraphHandle_t>(graph),
      ptrs.data(), static_cast<uint64_t>(ptrs.size()));
  return err == QNN_SUCCESS;
}

// QnnInterface_getProviders를 호출해 provider 리스트를 받아 적절한 provider를 선택
const void* QnnLoader::get_interface_provider(const char* provider_name) {
  if (!get_providers_fn_) return nullptr;
  using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto get_providers = reinterpret_cast<GetProvidersFn>(get_providers_fn_);
  const QnnInterface_t** providers = nullptr;
  uint32_t num = 0;
  auto err = get_providers(&providers, &num);
  if (err != QNN_SUCCESS || num == 0 || providers == nullptr) {
    std::cerr << "QnnInterface_getProviders failed or no providers\n";
    return nullptr;
  }
  if (provider_name == nullptr) {
    interface_provider_ = providers[0];
    return interface_provider_;
  }
  for (uint32_t i = 0; i < num; ++i) {
    if (providers[i] && providers[i]->providerName && std::string(providers[i]->providerName) == provider_name) {
      interface_provider_ = providers[i];
      return interface_provider_;
    }
  }
  interface_provider_ = providers[0];
  return interface_provider_;
}

// Executorch 흐름과 동일: logCreate → backendCreate → deviceCreate
bool QnnLoader::create_backend_and_device() {
  if (!interface_provider_) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  // 로거 생성(콜백 등록) 및 로그 레벨 설정 → 쉘 STDOUT로 바로 출력
  if (api.logCreate) {
    QnnLog_Level_t lvl = QNN_LOG_LEVEL_DEBUG;
    if (log_level_ >= 1 && log_level_ <= 5) {
      lvl = static_cast<QnnLog_Level_t>(log_level_);
    }
    api.logCreate(StdoutLogger, lvl,
                  reinterpret_cast<Qnn_LogHandle_t*>(&logger_));
  }
  if (logger_ && api.logSetLogLevel) {
    QnnLog_Level_t lvl = QNN_LOG_LEVEL_DEBUG;
    if (log_level_ >= 1 && log_level_ <= 5) {
      lvl = static_cast<QnnLog_Level_t>(log_level_);
    }
    api.logSetLogLevel(reinterpret_cast<Qnn_LogHandle_t>(logger_), lvl);
  }
  const QnnBackend_Config_t* backend_cfgs[] = {nullptr};
  if (!api.backendCreate) return false;
  if (api.backendCreate(reinterpret_cast<Qnn_LogHandle_t>(logger_), backend_cfgs,
                         reinterpret_cast<Qnn_BackendHandle_t*>(&backend_)) != QNN_SUCCESS || !backend_) {
    return false;
  }
  const QnnDevice_Config_t* dev_cfgs[] = {nullptr};
  if (!api.deviceCreate) return false;
  if (api.deviceCreate(reinterpret_cast<Qnn_LogHandle_t>(logger_), dev_cfgs,
                        reinterpret_cast<Qnn_DeviceHandle_t*>(&device_)) != QNN_SUCCESS || !device_) {
    return false;
  }
  return true;
}

// ListAsync 경로는 제거했습니다

// 생성 역순으로 리소스 해제 및 dlclose 수행
void QnnLoader::cleanup() {
  if (interface_provider_) {
    auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
    const auto& api = qnn->QNN_INTERFACE_VER_NAME;
    for (void* c : contexts_) {
      if (c && api.contextFree) api.contextFree(reinterpret_cast<Qnn_ContextHandle_t>(c), nullptr);
    }
    contexts_.clear();
    if (device_ && api.deviceFree) {
      api.deviceFree(reinterpret_cast<Qnn_DeviceHandle_t>(device_));
    }
    if (backend_ && api.backendFree) {
      api.backendFree(reinterpret_cast<Qnn_BackendHandle_t>(backend_));
    }
    if (logger_ && api.logFree) {
      api.logFree(reinterpret_cast<Qnn_LogHandle_t>(logger_));
    }
  }
  device_ = nullptr;
  backend_ = nullptr;
  logger_ = nullptr;
  if (handles_.backend_so_handle) { dlclose(handles_.backend_so_handle); handles_.backend_so_handle = nullptr; }
  if (handles_.system_so_handle) { dlclose(handles_.system_so_handle); handles_.system_so_handle = nullptr; }
}

bool QnnLoader::create_context_from_binary(const void* binary, size_t binary_size) {
  if (!interface_provider_ || !backend_ || !device_) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  if (!api.contextCreateFromBinary) return false;
  void* ctx = nullptr;
  auto err = api.contextCreateFromBinary(
      reinterpret_cast<Qnn_BackendHandle_t>(backend_),
      reinterpret_cast<Qnn_DeviceHandle_t>(device_),
      /*config*/nullptr,
      binary,
      static_cast<Qnn_ContextBinarySize_t>(binary_size),
      reinterpret_cast<Qnn_ContextHandle_t*>(&ctx),
      /*profile*/nullptr);
  if (err != QNN_SUCCESS) return false;
  contexts_.push_back(ctx);
  return true;
}

bool QnnLoader::retrieve_graph(size_t ctx_index, const std::string& graph_name) {
  if (!interface_provider_) return false;
  if (ctx_index >= contexts_.size()) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  if (!api.graphRetrieve) return false;
  void* graph = nullptr;
  auto err = api.graphRetrieve(
      reinterpret_cast<Qnn_ContextHandle_t>(contexts_[ctx_index]),
      graph_name.c_str(),
      reinterpret_cast<Qnn_GraphHandle_t*>(&graph));
  if (err != QNN_SUCCESS || graph == nullptr) return false;
  graphs_.push_back(graph);
  return true;
}

bool QnnLoader::execute_graph(size_t ctx_index,
                              const std::string& graph_name,
                              const std::vector<Qnn_Tensor_t>& inputs,
                              std::vector<Qnn_Tensor_t>& outputs) {
  if (!interface_provider_) return false;
  if (ctx_index >= contexts_.size()) return false;
  auto qnn = reinterpret_cast<const QnnInterface_t*>(interface_provider_);
  const auto& api = qnn->QNN_INTERFACE_VER_NAME;
  if (!api.graphRetrieve || !api.graphExecute) return false;
  void* graph = nullptr;
  if (api.graphRetrieve(reinterpret_cast<Qnn_ContextHandle_t>(contexts_[ctx_index]),
                        graph_name.c_str(),
                        reinterpret_cast<Qnn_GraphHandle_t*>(&graph)) != QNN_SUCCESS || !graph) {
    return false;
  }
  Qnn_ErrorHandle_t err = api.graphExecute(
      reinterpret_cast<Qnn_GraphHandle_t>(graph),
      inputs.data(), static_cast<uint32_t>(inputs.size()),
      outputs.data(), static_cast<uint32_t>(outputs.size()),
      /*profile*/nullptr,
      /*signal*/nullptr);
  return err == QNN_SUCCESS;
}

} // namespace llm_test


