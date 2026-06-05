"""
统一 API 生成脚本

用法:
    # 获取数据总条数
    python script/gen/api_gen.py --count_only --benchmark mssbench --data_dir data/mssbench

    # API 生成
    python script/gen/api_gen.py \
        --benchmark mssbench \
        --data_dir data/mssbench \
        --output_dir output/api_gen/mssbench \
        --model Qwen2.5-VL-7B-Instruct \
        --shard_id 0 --num_shards 1 \
        --max_tokens 512 \
        --temperature 0.0
    
    # 使用 constitution 模式（在生成时添加 default policy）
    python script/gen/api_gen.py \
        --benchmark mssbench \
        --data_dir data/mssbench \
        --output_dir output/api_gen/mssbench \
        --model Qwen2.5-VL-7B-Instruct \
        --constitution
"""

import argparse
import json
import os
import sys
import logging
import time

# 将 script/ 加入 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from data_adapter import get_adapter, set_constitution_mode
from utils.common import load_jsonl, save_jsonl, get_existing_ids
from utils.api_client import call_gen_api


def parse_args():
    parser = argparse.ArgumentParser(description="统一 API 生成脚本")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark 名称")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--model", type=str, default=None, help="生成模型名称")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片 ID")
    parser.add_argument("--num_shards", type=int, default=1, help="总分片数")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--batch_size", type=int, default=1, help="批量处理大小（API 生成建议为1）")
    parser.add_argument("--count_only", action="store_true", help="仅输出数据总条数后退出")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续跑")
    parser.add_argument("--delay", type=float, default=0.5, help="API 调用间隔（秒）")
    parser.add_argument("--constitution_mode", type=str, default="none",
                        choices=["none", "prefix", "system"],
                        help="Constitution 模式: none=不嵌入policy, prefix=在user message前拼接policy, system=将policy作为system message插入")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Shard {args.shard_id}] %(levelname)s - %(message)s",
    )

    # 1. 加载数据
    # 如果启用了 constitution 模式，设置 default policy
    if args.constitution_mode != "none":
        from data.policy_category import Default_Policy
        set_constitution_mode(mode=args.constitution_mode, policy_text=Default_Policy.strip())
        logging.info(f'Constitution mode enabled: {args.constitution_mode} with Default_Policy')
    
    adapter = get_adapter(args.benchmark)
    all_samples = adapter.load(args.data_dir)
    total_count = len(all_samples)

    # --count_only 模式：仅输出总条数
    if args.count_only:
        print(total_count)
        return

    logging.info(f"Benchmark: {args.benchmark}, Total samples: {total_count}")

    # 2. 分片
    shard_samples = [
        s for i, s in enumerate(all_samples)
        if i % args.num_shards == args.shard_id
    ]
    logging.info(
        f"Shard {args.shard_id}/{args.num_shards}: {len(shard_samples)} samples"
    )

    if not shard_samples:
        logging.warning("No samples for this shard, exiting.")
        return

    # 3. 断点续跑
    os.makedirs(args.output_dir, exist_ok=True)
    shard_output = os.path.join(
        args.output_dir, f"gen_shard_{args.shard_id}.jsonl"
    )

    existing_ids = set()
    if args.resume:
        existing_ids = get_existing_ids(shard_output)
        if existing_ids:
            logging.info(f"Resuming: found {len(existing_ids)} existing results")
            shard_samples = [s for s in shard_samples if s["id"] not in existing_ids]
            logging.info(f"Remaining: {len(shard_samples)} samples to process")

    if not shard_samples:
        logging.info("All samples already processed, exiting.")
        return

    # 4. 遍历生成
    results = []
    for idx, item in enumerate(shard_samples):
        # 获取消息格式
        messages = adapter.get_gen_messages(item)

        # 获取图片（如果有）
        image_path = item.get("image_path")
        image = None
        if image_path and os.path.exists(image_path):
            from PIL import Image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logging.error(f"Failed to load image {image_path}: {e}")

        # 调用 API 生成
        logging.info(f"[{idx + 1}/{len(shard_samples)}] Processing item: {item.get('id', 'unknown')}")

        try:
            response = call_gen_api(
                messages=messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                image=image,
                max_retries=3,
            )
        except Exception as e:
            logging.error(f"API call failed: {e}")
            response = None

        # 保存结果
        result = {
            "id": item["id"],
            "benchmark": args.benchmark,
            "question": item["question"],
            "image_path": item.get("image_path", ""),
            "response": response or "",
            "metadata": item.get("metadata", {}),
        }
        results.append(result)

        # 追加写入（支持断点续跑）
        mode = "a" if existing_ids or results else "w"
        with open(shard_output, mode, encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # API 调用间隔
        if args.delay > 0 and idx < len(shard_samples) - 1:
            time.sleep(args.delay)

        # 定期清理内存
        if idx > 0 and idx % 100 == 0:
            import gc
            gc.collect()

    logging.info(
        f"Shard {args.shard_id} done! {len(results)} results saved to {shard_output}"
    )


if __name__ == "__main__":
    main()