#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地 Qwen3-Omni 模型（通过 llama-server）评测 TTS 合成语音的风格一致性。
- 从提示词文件加载模板（含占位符 <此处插入待评测的语音风格描述>）
- 对每条记录和每种指令类型（APS/DSD/RP），将指令文本填入模板，调用本地模型评分
- 从返回文本中解析 JSON，提取"一致性"字段（true/false）
- 流式写入：每处理完一条立即写盘，防止数据丢失

用法示例：
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

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='eval.log',
    filemode='a'
)


def parse_args():
    p = argparse.ArgumentParser(description="使用本地 Qwen3-Omni 模型评测 TTS 语音风格一致性")
    p.add_argument("--input_jsonl", required=True, help="输入 JSONL 文件路径")
    p.add_argument("--output_jsonl", required=True, help="输出 JSONL 文件路径")
    p.add_argument("--prompt_file", required=True, help="包含占位符的提示词模板文件路径")
    p.add_argument("--base_url", default="http://localhost:6677/v1",
                   help="llama-server 的 OpenAI 兼容 API 地址（默认 http://localhost:6677/v1）")
    p.add_argument("--model_name", default="qwen3-omni",
                   help="模型名称（传给 API 的 model 参数，llama-server 通常忽略此值）")
    p.add_argument("--instruction_type", default="ALL",
                   choices=["APS", "DSD", "RP", "ALL"],
                   help="只处理指定指令类型，ALL 表示处理全部三种")
    return p.parse_args()


def load_template(path: str) -> str:
    """加载提示词模板"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_json_from_text(text):
    """从模型响应中提取 JSON 对象。Qwen3-Omni 返回纯净 JSON，直接解析即可。"""
    logging.info(f"原始响应内容:\n{text}")

    try:
        parsed = json.loads(text.strip())
        if '一致性' in parsed:
            logging.info(f"JSON 解析成功: {parsed}")
            return parsed
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解析失败: {e}")

    logging.error("未能从响应中提取有效的 JSON")
    return None


def encode_audio_base64(audio_path: str) -> dict:
    """将音频文件编码为 base64，返回 llama.cpp 兼容的音频数据 dict"""
    # 从文件扩展名推断音频格式
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    format_map = {'wav': 'wav', 'mp3': 'mp3', 'flac': 'flac', 'ogg': 'ogg', 'm4a': 'mp4'}
    audio_format = format_map.get(ext, 'wav')

    with open(audio_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    return {'data': data, 'format': audio_format}


def score_with_prompt(client: OpenAI, model: str, prompt_text: str,
                      audio_path: str, max_retries: int = 5):
    """使用本地 Qwen3-Omni 模型对音频进行风格一致性评分"""
    prompt_summary = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
    logging.info(f"准备处理音频: {audio_path}, 提示词摘要: {prompt_summary}")

    # 预编码音频（只编码一次，重试时复用）
    try:
        audio_input = encode_audio_base64(audio_path)
    except Exception as e:
        logging.error(f"音频文件编码失败 [{audio_path}]: {e}")
        return None

    for retry in range(max_retries):
        try:
            logging.info(f"处理音频: {audio_path}, 第 {retry+1}/{max_retries} 次尝试")

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
                logging.info(f"第 {retry+1} 次尝试响应:\n{text}")
                print(f"第 {retry+1} 次尝试响应:\n{text}")
                print(f"音频路径: {audio_path}")

            ctx = f"[Audio: {audio_path}]"
            logging.info(f"{ctx} API 返回原始文本: {text}")

            result = extract_json_from_text(text)

            if result and '一致性' in result:
                logging.info(f"{ctx} 成功提取一致性: {result['一致性']}")
                return result['一致性']

            logging.error(f"{ctx} 未能从文本中提取一致性字段: {text}")
            time.sleep(1)

        except Exception as e:
            logging.error(f"[Audio: {audio_path}] 处理异常: {str(e)}")
            time.sleep(2)

    logging.error(f"经过 {max_retries} 次尝试仍无法获得评分结果")
    return None


def process_single_item(item: dict, template: str, client: OpenAI,
                        model: str, instruction_type: str) -> dict:
    """处理单条数据"""
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
            logging.warning(f"音频文件不存在: {audio_path}")
            result[inst_type]['gemini_score'] = None
            continue

        if not instruction:
            logging.warning(f"指令为空: {inst_type}")
            result[inst_type]['gemini_score'] = None
            continue

        logging.info(f"处理 {inst_type} 类型, 指令: {instruction[:50]}...")
        prompt_text = template.replace(
            '<此处插入待评测的语音风格描述>', instruction)

        score = score_with_prompt(client, model, prompt_text, audio_path)
        logging.info(f"获得评分结果: {score}")
        result[inst_type]['gemini_score'] = score

    return result


def calculate_statistics(results):
    """计算 APS/DSD/RP 各项的评分统计"""
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
    """格式化输出统计数据"""
    print("\n" + "=" * 60)
    print("评估统计")
    print("=" * 60)

    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        print(f"{inst_type:>8}: {data['percentage']:6.2f}% "
              f"({data['true_count']:3d}/{data['valid_count']:3d} 有效, "
              f"{data['null_count']:3d} 空)")

    valid_types = [t for t in ['APS', 'DSD', 'RP'] if stats[t]['valid_count'] > 0]
    if valid_types:
        macro_avg = sum(stats[t]['percentage'] for t in valid_types) / len(valid_types)
        print(f"{'AVG':>8}: {macro_avg:6.2f}% (宏平均)")

    print("=" * 60)

    logging.info("评估统计:")
    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        logging.info(f"{inst_type}: {data['percentage']:.2f}% "
                     f"({data['true_count']}/{data['valid_count']} 有效, "
                     f"{data['null_count']} 空)")
    if valid_types:
        logging.info(f"AVG: {macro_avg:.2f}% (宏平均)")


def main():
    args = parse_args()
    logging.info(f"启动评测程序，参数: {vars(args)}")

    # 加载提示词模板
    template = load_template(args.prompt_file)
    logging.info(f"已加载提示词模板，长度: {len(template)} 字符")

    # 读取输入数据
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    logging.info(f"读取 {len(lines)} 条记录")

    items = [json.loads(line) for line in lines]
    logging.info(f"解析 {len(items)} 条数据")

    # 初始化 OpenAI 客户端（指向本地 llama-server）
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # 单进程顺序处理 + 流式写入
    results = []
    with open(args.output_jsonl, 'w', encoding='utf-8', buffering=1) as fout:
        for item in tqdm(items, desc="评分中"):
            try:
                result = process_single_item(
                    item, template, client, args.model_name, args.instruction_type)
                results.append(result)
            except Exception as e:
                logging.error(f"处理条目失败: {str(e)}")
                results.append(item)
            # 每条结果立即落盘
            fout.write(json.dumps(results[-1], ensure_ascii=False) + '\n')

    logging.info(f"处理完成，结果已保存至: {args.output_jsonl}")
    logging.info(f"共处理 {len(results)} 条样本")

    # 统计汇总
    stats = calculate_statistics(results)
    print_statistics(stats)


if __name__ == "__main__":
    main()
