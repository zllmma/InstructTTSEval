# InstructTTSEval — TTS 风格指令遵循能力评估

使用大模型（Qwen3-Omni / llama.cpp）评测 TTS 合成语音的风格一致性，覆盖 12 个声学维度。任何 TTS 系统合成的音频均可评测。

- **论文**: [arXiv 2506.16381](https://arxiv.org/abs/2506.16381)
- **数据集**: [HuggingFace](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval)
- **上游仓库**: [GitHub](https://github.com/KexinHUANG19/InstructTTSEval)

## 环境配置

### 1. 安装 Python 依赖

```bash
uv sync
```

### 2. 下载评测模型

Qwen3-Omni GGUF 权重来自 [ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF)，下载至 `pretrained/` 目录：

- `Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf`
- `mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf`

### 3. 确保 llama 命令行可用

```bash
# llama.cpp 需编译安装，llama serve 命令应在 PATH 中
which llama
```

## 完整流程

### 步骤 1：TTS 合成音频

用任意 TTS 系统生成音频，每条样本按三种指令类型（APS/DSD/RP）合成三个音频文件，整理为 JSONL 格式。

项目中提供了基于 Qwen3-TTS 的合成脚本作为参考：

```bash
# 编辑 test_synthesize.py 中的 NUM_SAMPLES 控制样本数量
uv run test_synthesize.py
```

### 步骤 2：评测

```bash
./run_eval.sh gen_wav/zh_20.jsonl gen_wav/zh_20_output.jsonl
```

脚本会自动启动 `llama serve`、等待就绪、运行评测、关闭服务。

可通过环境变量覆盖默认配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `QWEN_BASE_URL` | `http://localhost:6677/v1` | API 地址 |
| `QWEN_MODEL_NAME` | `qwen3-omni` | 模型名称 |
| `QWEN_INSTRUCTION_TYPE` | `ALL` | 指令类型（APS/DSD/RP/ALL） |
| `LLAMA_MODEL` | `pretrained/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf` | 模型路径 |
| `LLAMA_MMPROJ` | `pretrained/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf` | 多模态投影 |
| `LLAMA_PORT` | `6677` | 服务端口 |
| `PYTHON` | `.venv/bin/python` | Python 解释器 |

## 数据格式

输入为 JSONL，每条记录包含 `id`、`text` 以及三种指令类型：

```json
{
  "id": "zh_0",
  "text": "待合成的文本内容",
  "APS": {
    "instruction": "性别: 男性.\n音高: 男性中高音...",
    "gen_path": "gen_wav/zh_0_APS.wav"
  },
  "DSD": {
    "instruction": "体现标准普通话的发音,带有戏剧化的哭腔...",
    "gen_path": "gen_wav/zh_0_DSD.wav"
  },
  "RP": {
    "instruction": "像一位在深夜单独打电话寻求朋友原谅的青年...",
    "gen_path": "gen_wav/zh_0_RP.wav"
  }
}
```

- `instruction`：风格描述文本（三种类型 APS/DSD/RP）
- `gen_path`：合成音频路径（相对路径）
- 音频格式：WAV/MP3/FLAC 等

## 输出格式

评测脚本将 `true`/`false` 写入各指令的 `gemini_score` 字段，并输出统计：

```
============================================================
评估统计
============================================================
     APS:  75.00% ( 15/ 20 有效,   0 空)
     DSD:  95.00% ( 19/ 20 有效,   0 空)
      RP:  85.00% ( 17/ 20 有效,   0 空)
     AVG:  85.00% (宏平均)
============================================================
```

日志同时写入 `eval.log` 和控制台。