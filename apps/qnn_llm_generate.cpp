/**
 * @file qnn_llm_generate.cpp
 * @brief Clean LLM text generation using modularized components
 */

#include "llm_decode_runner.h"
#include <iostream>
#include <string>

using namespace llm_test;

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << "\n"
            << "  --ctx_dir DIR          QNN context directory\n"
            << "  --tokenizer PATH       Tokenizer model path (tokenizer.model)\n"
            << "  --prompt STR           Input prompt\n"
            << "  [--params PATH]        params.json path (optional, for dynamic config)\n"
            << "  [--backend_so PATH]    QNN backend library (default: libQnnHtp.so)\n"
            << "  [--system_so PATH]     QNN system library (optional)\n"
            << "  [--max_gen N]          Maximum tokens to generate (default: 100)\n"
            << "  [--log_level N]        QNN log verbosity 1=ERROR, 2=WARN, 3=INFO, 4=VERBOSE, 5=DEBUG (default: 1)\n"
            << "  [--multi_context]      Enable multi-context (sharding) mode\n"
            << "  [--num_shards N]       Number of shards (0=auto-detect, default)\n"
            << "\n"
            << "Example (single-context):\n"
            << "  " << prog << " \\\n"
            << "    --ctx_dir models/llama_qnn_1b \\\n"
            << "    --tokenizer models/llama_qnn_1b/tokenizer.model \\\n"
            << "    --params models/llama_qnn_1b/params.json \\\n"
            << "    --prompt \"The capital of France is\" \\\n"
            << "    --max_gen 50\n"
            << "\n"
            << "Example (multi-context, auto-detect shards):\n"
            << "  " << prog << " \\\n"
            << "    --ctx_dir models/llama_qnn_1b \\\n"
            << "    --tokenizer models/llama_qnn_1b/tokenizer.model \\\n"
            << "    --params models/llama_qnn_1b/params.json \\\n"
            << "    --prompt \"Hello\" \\\n"
            << "    --multi_context \\\n"
            << "    --max_gen 50\n";
}

int main(int argc, char** argv) {
  LLMDecodeConfig config;
  config.backend_so = "libQnnHtp.so";
  config.max_gen_tokens = 100;
  config.log_level = 1;
  
  std::string prompt;
  
  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--ctx_dir" && i + 1 < argc) {
      config.ctx_dir = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      config.tokenizer_path = argv[++i];
    } else if (arg == "--backend_so" && i + 1 < argc) {
      config.backend_so = argv[++i];
    } else if (arg == "--system_so" && i + 1 < argc) {
      config.system_so = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--max_gen" && i + 1 < argc) {
      config.max_gen_tokens = std::stoi(argv[++i]);
    } else if (arg == "--log_level" && i + 1 < argc) {
      config.log_level = std::stoi(argv[++i]);
    } else if (arg == "--params" && i + 1 < argc) {
      config.params_path = argv[++i];
    } else if (arg == "--multi_context") {
      config.use_multi_context = true;
    } else if (arg == "--num_shards" && i + 1 < argc) {
      config.num_shards = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 1;
    }
  }
  
  // Validate required arguments
  if (config.ctx_dir.empty() || config.tokenizer_path.empty() || prompt.empty()) {
    std::cerr << "Error: Missing required arguments\n";
    usage(argv[0]);
    return 1;
  }
  
  // Initialize runner
  LLMDecodeRunner runner(config);
  
  if (!runner.initialize()) {
    std::cerr << "Error: " << runner.get_error() << "\n";
    return 1;
  }
  
  // Generate text
  std::string output;
  if (!runner.generate(prompt, output)) {
    std::cerr << "Error: " << runner.get_error() << "\n";
    return 1;
  }
  
  if (config.log_level == 0) {
    // Quiet mode: only output generated text
    std::cout << output << "\n";
  }
  
  return 0;
}
