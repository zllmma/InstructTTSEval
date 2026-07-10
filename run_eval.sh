#!/bin/bash
# ============================================================
# InstructTTSEval — 使用本地 Qwen3-Omni 模型评测（自动管理服务）
# 用法: ./run_eval.sh [input.jsonl] [output.jsonl]
# ============================================================

set -euo pipefail

# ====== 用户可配置变量 ======
INPUT_JSONL="${1:-example_en.jsonl}"
OUTPUT_JSONL="${2:-example_en_score.jsonl}"
PROMPT_FILE="eval_prompt.txt"
BASE_URL="${QWEN_BASE_URL:-http://localhost:6677/v1}"
MODEL_NAME="${QWEN_MODEL_NAME:-qwen3-omni}"
INSTRUCTION_TYPE="${QWEN_INSTRUCTION_TYPE:-ALL}"

# ====== llama 服务配置 ======
LLAMA_BIN="${LLAMA_BIN:-llama}"
LLAMA_MODEL="${LLAMA_MODEL:-pretrained/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf}"
LLAMA_MMPROJ="${LLAMA_MMPROJ:-pretrained/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf}"
LLAMA_PORT="${LLAMA_PORT:-6677}"

# ====== 清理函数 ======
cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "正在关闭 llama serve (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
        echo "llama serve 已关闭。"
    fi
}
trap cleanup EXIT INT TERM

# ====== 启动 llama serve ======
echo "============================================"
echo "InstructTTSEval — Qwen3-Omni 评测"
echo "============================================"
echo "输入文件:     $INPUT_JSONL"
echo "输出文件:     $OUTPUT_JSONL"
echo "提示词模板:   $PROMPT_FILE"
echo "API 地址:     $BASE_URL"
echo "指令类型:     $INSTRUCTION_TYPE"
echo "--------------------------------------------"
echo "启动 llama serve..."

$LLAMA_BIN serve \
    -m "$LLAMA_MODEL" \
    --mmproj "$LLAMA_MMPROJ" \
    --port "$LLAMA_PORT" &
SERVER_PID=$!

# ====== 等待服务就绪 ======
echo -n "等待服务就绪"
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${LLAMA_PORT}/v1/models" > /dev/null 2>&1; then
        echo " ✓ (${ELAPSED}s)"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -n "."
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo " ✗ 超时！"
    exit 1
fi

# ====== 运行评测 ======
echo "============================================"
echo "开始评测..."
echo "============================================"

PYTHON="${PYTHON:-.venv/bin/python}"

$PYTHON eval.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$OUTPUT_JSONL" \
    --prompt_file "$PROMPT_FILE" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --instruction_type "$INSTRUCTION_TYPE"

echo "============================================"
echo "评测完成！结果已保存至 $OUTPUT_JSONL"
echo "============================================" 