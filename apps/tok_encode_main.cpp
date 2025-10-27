#include "tokenizer_llama.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace llm_test;

static void usage(const char* p) {
  std::cerr << "Usage: " << p << " --gguf PATH --prompt PROMPT [--system SYS] --out FILE\n";
}

int main(int argc, char** argv) {
  std::string gguf, prompt, system, out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
    else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
    else if (a == "--system" && i + 1 < argc) system = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
  }
  if (gguf.empty() || prompt.empty() || out.empty()) { usage(argv[0]); return 1; }

  LlamaTokenizer tok;
  if (!tok.init(gguf.c_str())) {
    std::cerr << "failed to load tokenizer gguf\n";
    return 2;
  }

  std::string formatted = format_llama32_prompt(prompt, system);
  auto toks = tok.encode(formatted, /*add_special*/true, /*parse_special*/true);

  std::ofstream f(out, std::ios::binary);
  if (!f) { std::cerr << "failed to open out file\n"; return 3; }
  for (int32_t v : toks) f.write(reinterpret_cast<const char*>(&v), sizeof(v));
  f.close();

  std::cout << "tokens: " << toks.size() << " saved to " << out << "\n";
  return 0;
}


