import os, json, argparse
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Extract reference_audio from test set and generate JSONL")
    p.add_argument("--split", default="zh", choices=["zh", "en"],
                   help="Dataset split (default zh)")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Number of samples to process (default 20)")
    p.add_argument("--output_jsonl", default=None,
                   help="Output JSONL path (default gt_wav/{split}_{num_samples}_gt.jsonl)")
    p.add_argument("--output_dir", default="gt_wav",
                   help="Audio output directory (default gt_wav)")
    return p.parse_args()


def main():
    args = parse_args()

    output_jsonl = args.output_jsonl or f"{args.output_dir}/{args.split}_{args.num_samples}_gt.jsonl"

    parquet_path = f"datasets/InstructTTSEval/{args.split}.parquet"
    df = pd.read_parquet(parquet_path)
    df = df.head(args.num_samples)
    print(f"Processing {len(df)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    inst_types = ["APS", "DSD", "RP"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        sample_id = row["id"]
        text = row["text"]
        record = {"id": sample_id, "text": text}

        ref_audio = row["reference_audio"]
        audio_bytes = ref_audio["bytes"]
        output_path = f"{args.output_dir}/{sample_id}_gt.wav"

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        for inst_type in inst_types:
            instruction = row[inst_type]
            record[inst_type] = {"instruction": instruction, "gen_path": output_path}

        results.append(record)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} samples written to {output_jsonl}")


if __name__ == "__main__":
    main()
