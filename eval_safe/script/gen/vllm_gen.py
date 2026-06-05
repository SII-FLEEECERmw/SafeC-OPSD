"""
统一 vLLM 生成脚本（单分片，被 dp_launcher.sh 调用）

用法:
    # 获取数据总条数
    python script/gen/vllm_gen.py --count_only --benchmark mssbench --data_dir data/mssbench

    # 单分片推理
    python script/gen/vllm_gen.py \
        --model_path /path/to/model \
        --benchmark mssbench \
        --data_dir data/mssbench \
        --output_dir output/model_name/mssbench \
        --shard_id 0 --num_shards 1 \
        --tp_size 1
"""

import argparse
import json
import os
import sys
import logging
import multiprocessing

# 将 script/ 加入 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from data_adapter import get_adapter, set_constitution_mode
from utils.common import load_jsonl, save_jsonl, get_existing_ids


def parse_args():
    parser = argparse.ArgumentParser(description="统一 vLLM 生成脚本")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark 名称")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片 ID")
    parser.add_argument("--num_shards", type=int, default=1, help="总分片数")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096, help="最大模型长度")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--batch_size", type=int, default=64, help="批量推理大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU 内存利用率")
    # 多模态相关参数
    parser.add_argument("--limit_mm_per_prompt", type=str, default="image:1",
                        help="多模态输入限制，格式如 'image:5,video:1'")
    parser.add_argument("--min_pixels", type=int, default=None,
                        help="动态分辨率最小像素值，如 28*28*4=3136")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="动态分辨率最大像素值，如 28*28*256=200704")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="是否信任远程代码（默认 True）")
    parser.add_argument("--count_only", action="store_true", help="仅输出数据总条数后退出")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续跑")
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

    # 4. 延迟导入重型库
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from PIL import Image

    # 5. 初始化模型
    logging.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    llm_kwargs = {
        "model": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    # 解析 limit_mm_per_prompt 参数（格式: "image:5,video:1"）
    if args.limit_mm_per_prompt:
        limit_dict = {}
        for item in args.limit_mm_per_prompt.split(","):
            if ":" in item:
                key, val = item.strip().split(":")
                limit_dict[key.strip()] = int(val.strip())
        if limit_dict:
            llm_kwargs["limit_mm_per_prompt"] = limit_dict

    # 设置动态分辨率参数（用于 Qwen3-VL 等模型）
    mm_processor_kwargs = {}
    if args.min_pixels is not None:
        mm_processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        mm_processor_kwargs["max_pixels"] = args.max_pixels
    if mm_processor_kwargs:
        llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        logging.info(f"Using mm_processor_kwargs: {mm_processor_kwargs}")

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 6. 构造 vLLM 输入
    vllm_inputs = []
    valid_samples = []
    skipped = 0

    for item in shard_samples:
        messages = adapter.get_gen_messages(item)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        mm_data = {}
        if item.get("image_path") and os.path.exists(item["image_path"]):
            try:
                pil_image = Image.open(item["image_path"]).convert("RGB")
                mm_data["image"] = pil_image
            except Exception as e:
                logging.error(f"Failed to load image {item['image_path']}: {e}")
                skipped += 1
                continue

        input_dict = {"prompt": prompt_text}
        if mm_data:
            input_dict["multi_modal_data"] = mm_data

        vllm_inputs.append(input_dict)
        valid_samples.append(item)

    logging.info(f"Prepared {len(vllm_inputs)} inputs, skipped {skipped}")

    # 7. 批量推理
    logging.info("Starting batch inference...")
    outputs = llm.generate(vllm_inputs, sampling_params)

    # 8. 保存结果（追加模式）
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        item = valid_samples[i]
        results.append({
            "id": item["id"],
            "benchmark": args.benchmark,
            "question": item["question"],
            "image_path": item.get("image_path", ""),
            "response": generated_text,
            "metadata": item.get("metadata", {}),
        })

    # 追加写入（支持断点续跑）
    mode = "a" if existing_ids else "w"
    with open(shard_output, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(
        f"Shard {args.shard_id} done! {len(results)} results saved to {shard_output}"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
