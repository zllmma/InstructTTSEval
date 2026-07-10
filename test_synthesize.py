"""步骤 1：TTS 合成 — 取前 N 条中文样本，用 Qwen3-TTS 合成音频"""
import os, json
import torch
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel

# ====== 配置 ======
NUM_SAMPLES = 20
OUTPUT_JSONL = "gen_wav/zh_20.jsonl"
# ==================

df = pd.read_parquet("datasets/InstructTTSEval/zh.parquet")
df = df.head(NUM_SAMPLES)
print(f"处理 {len(df)} 条样本")

# 加载 TTS 模型
print("加载 Qwen3-TTS VoiceDesign 模型...")
tts_model = Qwen3TTSModel.from_pretrained(
    "pretrained/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

os.makedirs("gen_wav", exist_ok=True)
results = []
inst_types = ["APS", "DSD", "RP"]

for _, row in tqdm(df.iterrows(), total=len(df), desc="合成中"):
    sample_id = row["id"]
    text = row["text"]
    record = {"id": sample_id, "text": text}

    for inst_type in inst_types:
        instruction = row[inst_type]
        output_path = f"gen_wav/{sample_id}_{inst_type}.wav"

        wavs, sr = tts_model.generate_voice_design(
            text=text,
            language="Chinese",
            instruct=instruction,
        )
        sf.write(output_path, wavs[0], sr)
        record[inst_type] = {"instruction": instruction, "gen_path": output_path}

    results.append(record)

# 写入 JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n✅ 合成完成！{len(results)} 条样本写入 {OUTPUT_JSONL}")
print(f"下一步: python eval.py --input_jsonl {OUTPUT_JSONL} --output_jsonl gen_wav/zh_20_output.jsonl --prompt_file eval_prompt.txt --instruction_type ALL")
