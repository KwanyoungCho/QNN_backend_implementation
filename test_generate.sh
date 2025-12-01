#!/bin/bash

# Quick test script for the new modularized qnn_llm_generate

CTX_DIR="models/llama_qnn_1b"
TOKENIZER="models/llama_qnn_1b/tokenizer.model"
BACKEND_SO="/path/to/libQnnHtp.so"  # Update this path!
PROMPT="The capital of France is"
MAX_GEN=50
LOG_LEVEL=1

echo "========================================="
echo "Testing Modularized LLM Generation"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Context Dir: $CTX_DIR"
echo "  Tokenizer:   $TOKENIZER"
echo "  Backend:     $BACKEND_SO"
echo "  Prompt:      \"$PROMPT\""
echo "  Max Gen:     $MAX_GEN"
echo ""

./build/qnn_llm_generate \
  --ctx_dir "$CTX_DIR" \
  --tokenizer "$TOKENIZER" \
  --backend_so "$BACKEND_SO" \
  --prompt "$PROMPT" \
  --max_gen "$MAX_GEN" \
  --log_level "$LOG_LEVEL"

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
