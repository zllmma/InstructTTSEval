# InstructTTSEval — TTS Style Instruction Following Evaluation

Evaluates style consistency of synthesized TTS speech using a large model (Qwen3-Omni), covering 12 acoustic dimensions. Audio synthesized by any TTS system can be evaluated.

- **Paper**: [arXiv 2506.16381](https://arxiv.org/abs/2506.16381)
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval)
- **Upstream repo**: [InstructTTSEval](https://github.com/KexinHUANG19/InstructTTSEval)

## Setup

### 1. Install Python dependencies

Core dependencies (evaluation + ground truth extraction):

```bash
uv sync
```

If you also need TTS synthesis with Qwen3-TTS, install with the `tts` extra:

```bash
uv sync --extra tts
```

### 2. Download the evaluation model

Qwen3-Omni GGUF weights from [ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF), place under `pretrained/`:

- `Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf`
- `mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf`

### 3. Ensure the [llama](https://github.com/ggml-org/llama.cpp) CLI is available

## Full Workflow

### Step 1: Synthesize TTS audio

Generate audio with any TTS system. For each sample, synthesize three audio files (one per instruction type: APS/DSD/RP) and organize them as JSONL.

A reference synthesis script based on Qwen3-TTS is provided:

```bash
uv run test_synthesize.py --split zh --num_samples 20
```

### Step 2: Start llama serve

```bash
./run_llama_server.sh
```

The service runs in the foreground. Open another terminal to run evaluation.

### Step 3: Run evaluation

```bash
uv run eval.py --input_jsonl gen_wav/zh_20.jsonl --output_jsonl gen_wav/zh_20_output.jsonl --prompt_file eval_prompt.txt --instruction_type ALL
```

Default configuration can be overridden via environment variables:

| Env variable | Default | Description |
|-------------|---------|-------------|
| `LLAMA_BIN` | `llama` | Path to llama binary |
| `LLAMA_MODEL` | `pretrained/Qwen3-Omni-30B-A3B-Thinking-Q4_K_M.gguf` | Model path |
| `LLAMA_MMPROJ` | `pretrained/mmproj-Qwen3-Omni-30B-A3B-Thinking-bf16.gguf` | Multimodal projector |
| `LLAMA_PORT` | `6677` | Server port |

## Data Format

Input is JSONL, each record contains `id`, `text`, and three instruction types:

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

- `instruction`: style description text (three types: APS/DSD/RP)
- `gen_path`: path to synthesized audio (relative path)
- Audio format: WAV/MP3/FLAC, etc.

## Output Format

The evaluation script writes `true`/`false` into the `gemini_score` field of each instruction type, and outputs statistics:

```
============================================================
Evaluation Statistics
============================================================
     APS:  75.00% ( 15/ 20 valid,   0 null)
     DSD:  95.00% ( 19/ 20 valid,   0 null)
      RP:  85.00% ( 17/ 20 valid,   0 null)
     AVG:  85.00% (macro average)
============================================================
```

Logs are written to both `eval.log` and stdout.
