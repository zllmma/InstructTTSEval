#!/bin/bash
# ============================================================
# InstructTTSEval — 使用本地 Qwen3-Omni 模型评测
# ============================================================
# 前置条件：先启动 llama-server
#   llama-server \
#     -m pretrained/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf \
#     --mmproj pretrained/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf \
#     --port 6677 \
#     --n-gpu-layers 99 \
#     --host 0.0.0.0
# ============================================================

set -euo pipefail

# ====== 用户可配置变量 ======
INPUT_JSONL="${1:-example_en.jsonl}"
OUTPUT_JSONL="${2:-example_en_score.jsonl}"
PROMPT_FILE="eval_prompt.txt"
BASE_URL="${QWEN_BASE_URL:-http://localhost:6677/v1}"
MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-omni}"
INSTRUCTION_TYPE="${QWEN_INSTRUCTION_TYPE:-ALL}"

# ====== 运行评测 ======
echo "============================================"
echo "InstructTTSEval — Qwen3-Omni 评测"
echo "============================================"
echo "输入文件:     $INPUT_JSONL"
echo "输出文件:     $OUTPUT_JSONL"
echo "提示词模板:   $PROMPT_FILE"
echo "API 地址:     $BASE_URL"
echo "指令类型:     $INSTRUCTION_TYPE"
echo "============================================"

python eval.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$OUTPUT_JSONL" \
    --prompt_file "$PROMPT_FILE" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --instruction_type "$INSTRUCTION_TYPE"

echo "评测完成！结果已保存至 $OUTPUT_JSONL" 