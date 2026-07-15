#!/usr/bin/env -S bash -x -euo pipefail
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-llama}"
LLAMA_MODEL="${LLAMA_MODEL:-./pretrained/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf}"
LLAMA_MMPROJ="${LLAMA_MMPROJ:-./pretrained/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf}"
LLAMA_PORT="${LLAMA_PORT:-6677}"

$LLAMA_BIN serve \
    -m "$LLAMA_MODEL" \
    --mmproj "$LLAMA_MMPROJ" \
    --port "$LLAMA_PORT"
