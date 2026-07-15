#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate style consistency of synthesized TTS speech using a local Qwen3-Omni model
(via llama-server).

- Loads a prompt template from file (containing placeholder <此处插入待评测的语音风格描述>)
- For each record and instruction type (APS/DSD/RP), fills instruction text into the
  template and calls the local model for scoring
- Parses JSON from the model response, extracting the "一致性" field (true/false)
- Streaming write: each result is flushed to disk immediately after processing
  to prevent data loss

Usage:
python eval.py \
  --input_jsonl   example_en.jsonl \
  --output_jsonl  example_en_score.jsonl \
  --prompt_file   eval_prompt.txt \
  --base_url      http://localhost:6677/v1 \
  --instruction_type ALL
"""
import os
import json
import time
import argparse
import logging
import base64
from tqdm import tqdm
from openai import OpenAI

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='eval.log',
    filemode='a'
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TTS speech style consistency with local Qwen3-Omni")
    p.add_argument("--input_jsonl", required=True, help="Path to input JSONL file")
    p.add_argument("--output_jsonl", required=True, help="Path to output JSONL file")
    p.add_argument("--prompt_file", required=True, help="Path to prompt template file (contains '<此处插入待评测的语音风格描述>' placeholder)")
    p.add_argument("--base_url", default="http://localhost:6677/v1",
                   help="OpenAI-compatible API base URL for llama-server (default http://localhost:6677/v1)")
    p.add_argument("--model_name", default="qwen3-omni",
                   help="Model name parameter passed to the API (llama-server usually ignores this)")
    p.add_argument("--instruction_type", default="ALL",
                   choices=["APS", "DSD", "RP", "ALL"],
                   help="Process only the specified instruction type; ALL processes all three")
    return p.parse_args()


def load_template(path: str) -> str:
    """Load the prompt template from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_json_from_text(text):
    """Extract JSON from model response. Qwen3-Omni returns clean JSON, so parse directly."""
    logging.info(f"Raw response:\n{text}")

    try:
        parsed = json.loads(text.strip())
        if '一致性' in parsed:
            logging.info(f"JSON parsed successfully: {parsed}")
            return parsed
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")

    logging.error("Failed to extract valid JSON from response")
    return None


def encode_audio_base64(audio_path: str) -> dict:
    """Encode audio file to base64, returning a llama.cpp-compatible audio dict."""
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    format_map = {'wav': 'wav', 'mp3': 'mp3', 'flac': 'flac', 'ogg': 'ogg', 'm4a': 'mp4'}
    audio_format = format_map.get(ext, 'wav')

    with open(audio_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    return {'data': data, 'format': audio_format}


def score_with_prompt(client: OpenAI, model: str, prompt_text: str,
                      audio_path: str, max_retries: int = 5):
    """Evaluate style consistency of an audio file using the local Qwen3-Omni model."""
    prompt_summary = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
    logging.info(f"Preparing audio: {audio_path}, prompt preview: {prompt_summary}")

    try:
        audio_input = encode_audio_base64(audio_path)
    except Exception as e:
        logging.error(f"Audio encoding failed [{audio_path}]: {e}")
        return None

    for retry in range(max_retries):
        try:
            logging.info(f"Processing audio: {audio_path}, attempt {retry+1}/{max_retries}")

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "input_audio", "input_audio": audio_input},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=4096,
            )

            text = resp.choices[0].message.content.strip()
            usage = resp.usage

            if retry != 0:
                logging.info(f"Attempt {retry+1} response:\n{text}")
                print(f"Attempt {retry+1} response:\n{text}")
                print(f"Audio path: {audio_path}")

            ctx = f"[Audio: {audio_path}]"
            logging.info(f"{ctx} API raw response: {text}")

            result = extract_json_from_text(text)

            if result and '一致性' in result:
                logging.info(f"{ctx} Successfully extracted consistency: {result['一致性']}")
                return result['一致性']

            logging.error(f"{ctx} Failed to extract consistency field: {text}")
            time.sleep(1)

        except Exception as e:
            logging.error(f"[Audio: {audio_path}] Exception: {str(e)}")
            time.sleep(2)

    logging.error(f"Failed to get a valid score after {max_retries} attempts")
    return None


def process_single_item(item: dict, template: str, client: OpenAI,
                        model: str, instruction_type: str) -> dict:
    """Process a single data item."""
    result = item.copy()

    instruction_types = (["APS", "DSD", "RP"] if instruction_type == "ALL"
                         else [instruction_type])

    for inst_type in instruction_types:
        if inst_type not in result:
            continue

        inst_data = result[inst_type]
        instruction = inst_data.get('instruction')
        audio_path = inst_data.get('gen_path')

        if not audio_path or not os.path.isfile(audio_path):
            logging.warning(f"Audio file not found: {audio_path}")
            result[inst_type]['gemini_score'] = None
            continue

        if not instruction:
            logging.warning(f"Instruction is empty: {inst_type}")
            result[inst_type]['gemini_score'] = None
            continue

        logging.info(f"Processing {inst_type}, instruction: {instruction[:50]}...")
        prompt_text = template.replace(
            '<此处插入待评测的语音风格描述>', instruction)

        score = score_with_prompt(client, model, prompt_text, audio_path)
        logging.info(f"Score obtained: {score}")
        result[inst_type]['gemini_score'] = score

    return result


def calculate_statistics(results):
    """Calculate scoring statistics for APS/DSD/RP."""
    stats = {
        'APS': {'total': 0, 'true_count': 0, 'null_count': 0},
        'DSD': {'total': 0, 'true_count': 0, 'null_count': 0},
        'RP': {'total': 0, 'true_count': 0, 'null_count': 0},
    }

    for result in results:
        for inst_type in ['APS', 'DSD', 'RP']:
            if inst_type in result:
                stats[inst_type]['total'] += 1
                score = result[inst_type].get('gemini_score')
                if score is True:
                    stats[inst_type]['true_count'] += 1
                elif score is None:
                    stats[inst_type]['null_count'] += 1

    for inst_type in stats:
        total = stats[inst_type]['total']
        true_count = stats[inst_type]['true_count']
        null_count = stats[inst_type]['null_count']

        if total > 0:
            valid_count = total - null_count
            percentage = (true_count / valid_count * 100) if valid_count > 0 else 0.0
        else:
            percentage = 0.0
            valid_count = 0

        stats[inst_type]['percentage'] = percentage
        stats[inst_type]['valid_count'] = valid_count

    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "=" * 60)
    print("Evaluation Statistics")
    print("=" * 60)

    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        print(f"{inst_type:>8}: {data['percentage']:6.2f}% "
              f"({data['true_count']:3d}/{data['valid_count']:3d} valid, "
              f"{data['null_count']:3d} null)")

    valid_types = [t for t in ['APS', 'DSD', 'RP'] if stats[t]['valid_count'] > 0]
    if valid_types:
        macro_avg = sum(stats[t]['percentage'] for t in valid_types) / len(valid_types)
        print(f"{'AVG':>8}: {macro_avg:6.2f}% (macro average)")

    print("=" * 60)

    logging.info("Evaluation statistics:")
    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        logging.info(f"{inst_type}: {data['percentage']:.2f}% "
                     f"({data['true_count']}/{data['valid_count']} valid, "
                     f"{data['null_count']} null)")
    if valid_types:
        logging.info(f"AVG: {macro_avg:.2f}% (macro average)")


def main():
    args = parse_args()
    logging.info(f"Starting evaluation, args: {vars(args)}")

    template = load_template(args.prompt_file)
    logging.info(f"Prompt template loaded, length: {len(template)} chars")

    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    logging.info(f"Read {len(lines)} records")

    items = [json.loads(line) for line in lines]
    logging.info(f"Parsed {len(items)} items")

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    results = []
    with open(args.output_jsonl, 'w', encoding='utf-8', buffering=1) as fout:
        for item in tqdm(items, desc="Scoring"):
            try:
                result = process_single_item(
                    item, template, client, args.model_name, args.instruction_type)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process item: {str(e)}")
                results.append(item)
            fout.write(json.dumps(results[-1], ensure_ascii=False) + '\n')

    logging.info(f"Processing complete, results saved to: {args.output_jsonl}")
    logging.info(f"Total samples processed: {len(results)}")

    stats = calculate_statistics(results)
    print_statistics(stats)


if __name__ == "__main__":
    main()
