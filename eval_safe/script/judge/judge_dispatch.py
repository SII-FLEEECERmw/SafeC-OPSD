"""
Judge 分发器
根据 benchmark 配置选择 judge 方式，统一处理 gen_results.jsonl -> judge_results.jsonl

用法:
    python script/judge/judge_dispatch.py \
        --benchmark mssbench \
        --input output/model/mssbench/gen_results.jsonl \
        --output output/model/mssbench/judge_results.jsonl \
        --config script/config/benchmarks.yaml \
        --threads 32

可选参数:
    --baseline_gen_results: 覆盖 yaml 中配置的 baseline gen_results 路径
                            （beavertails / spavl 等胜和率任务使用）
"""

import argparse
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from utils.common import load_jsonl, save_jsonl, load_yaml, get_existing_ids
from judge.judge_api import (
    judge_api_safe,
    judge_api_effective,
    judge_api_overrefusal,
    judge_api_vlsbench,
    judge_api_winrate,
)
from judge.judge_keyword import (
    judge_keyword_vlguard,
    judge_keyword_vlsbench,
)


def judge_single_item(item: dict, benchmark: str, config: dict) -> dict:
    """
    对单个样本执行完整的 judge 流程。

    Args:
        item: gen_results 中的一条记录
        benchmark: benchmark 名称
        config: benchmark 配置

    Returns:
        追加了 judge 结果的 item
    """
    judge_modes = config.get("judge_modes", ["safe"])

    try:
        if benchmark == "vlsbench":
            # 关键词匹配 + API 兜底
            result = judge_keyword_vlsbench(item)
            if result is not None:
                return result
            # 关键词未匹配，走 API
            return judge_api_vlsbench(item)

        elif benchmark == "vlguard":
            # 根据 subset 分别走 safe / overrefusal API 评测
            subset = item.get("metadata", {}).get("subset", "")
            if subset == "safe_safes":
                # 安全样本：判过度拒绝
                return judge_api_overrefusal(item, benchmark)
            else:
                # unsafes / safe_unsafes：判 safe，并把 reference 送进 prompt
                return judge_api_safe(item, benchmark)

        elif benchmark in ("beavertails", "spavl"):
            # 胜和率评测：必须有 baseline_response
            for mode in judge_modes:
                if mode in ("winrate_harm", "winrate_help"):
                    eval_type = mode.replace("winrate_", "")
                    item = judge_api_winrate(item, benchmark, eval_type)
                elif mode == "safe":
                    item = judge_api_safe(item, benchmark)
            return item

        else:
            # 通用 API judge (mssbench, mssembodied, siuo)
            for mode in judge_modes:
                if mode == "safe":
                    item = judge_api_safe(item, benchmark)
                elif mode == "effective":
                    item = judge_api_effective(item, benchmark)
            return item

    except Exception as e:
        logging.error(f"Judge failed for {item.get('id', 'unknown')}: {e}")
        item["judge_error"] = str(e)
        item["judge_safe"] = False
        return item


def attach_baseline_responses(data: list, baseline_path: str) -> list:
    """
    将 baseline gen_results 中的 response 注入到 data 的每条记录中作为 baseline_response。
    匹配优先按 id，其次按 question。
    """
    if not baseline_path or not os.path.exists(baseline_path):
        logging.error(f"Baseline gen_results 不存在: {baseline_path}")
        return data

    baseline_data = load_jsonl(baseline_path)
    by_id = {b.get("id"): b.get("response", "") for b in baseline_data if b.get("id")}
    by_question = {b.get("question"): b.get("response", "") for b in baseline_data if b.get("question")}

    matched = 0
    for item in data:
        bid = item.get("id")
        bresp = None
        if bid and bid in by_id:
            bresp = by_id[bid]
        elif item.get("question") in by_question:
            bresp = by_question[item.get("question")]
        item["baseline_response"] = bresp or ""
        if bresp:
            matched += 1

    logging.info(f"Baseline 匹配: {matched}/{len(data)} 条记录")
    return data


def main():
    parser = argparse.ArgumentParser(description="Judge 分发器")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark 名称")
    parser.add_argument("--input", type=str, required=True, help="gen_results.jsonl 路径")
    parser.add_argument("--output", type=str, required=True, help="judge_results.jsonl 输出路径")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(SCRIPT_DIR, "config", "benchmarks.yaml"),
        help="benchmarks.yaml 配置路径",
    )
    parser.add_argument("--threads", type=int, default=32, help="并发线程数")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续跑")
    parser.add_argument(
        "--baseline_gen_results", type=str, default=None,
        help="覆盖 yaml 中配置的 baseline gen_results 路径（用于胜和率任务）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 加载配置
    all_config = load_yaml(args.config)
    if args.benchmark not in all_config:
        logging.error(f"Benchmark '{args.benchmark}' not found in config")
        sys.exit(1)
    bench_config = all_config[args.benchmark]

    # 加载输入数据
    data = load_jsonl(args.input)
    logging.info(f"Loaded {len(data)} samples from {args.input}")

    # 胜和率任务：注入 baseline_response
    judge_modes = bench_config.get("judge_modes", [])
    needs_baseline = any(m.startswith("winrate") for m in judge_modes)
    if needs_baseline:
        baseline_path = args.baseline_gen_results or bench_config.get("baseline_gen_results")
        if not baseline_path:
            logging.error(
                f"Benchmark {args.benchmark} 需要 baseline_gen_results，但未在配置中找到。"
            )
            sys.exit(1)
        # 检查是否对比同一个文件（即 baseline 自身评测自身）
        if os.path.abspath(baseline_path) == os.path.abspath(args.input):
            logging.warning(
                "当前评测的 gen_results 与 baseline 相同，胜和率结果将无意义，跳过。"
            )
            return
        logging.info(f"使用 baseline: {baseline_path}")
        data = attach_baseline_responses(data, baseline_path)

    # 断点续跑
    existing_ids = set()
    existing_results = []
    if args.resume and os.path.exists(args.output):
        existing_results = load_jsonl(args.output)
        existing_ids = {item["id"] for item in existing_results if "id" in item}
        logging.info(f"Resuming: found {len(existing_ids)} existing judge results")

    # 过滤已完成的
    pending = [item for item in data if item.get("id") not in existing_ids]
    logging.info(f"Pending: {len(pending)} samples to judge")

    if not pending:
        logging.info("All samples already judged.")
        return

    # 并行执行 judge
    new_results = []
    judge_type = bench_config.get("judge_type", "api")

    if "keyword" in judge_type and "api" not in judge_type:
        # 纯关键词匹配，不需要多线程
        for item in tqdm(pending, desc=f"Judging {args.benchmark} (keyword)"):
            result = judge_single_item(item, args.benchmark, bench_config)
            new_results.append(result)
    else:
        # API judge，使用多线程
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(
                    judge_single_item, item, args.benchmark, bench_config
                ): item
                for item in pending
            }

            for future in tqdm(
                as_completed(futures),
                total=len(pending),
                desc=f"Judging {args.benchmark}",
            ):
                try:
                    result = future.result()
                    new_results.append(result)
                except Exception as e:
                    item = futures[future]
                    logging.error(f"Judge exception for {item.get('id')}: {e}")
                    item["judge_error"] = str(e)
                    item["judge_safe"] = False
                    new_results.append(item)

    # 合并并保存
    all_results = existing_results + new_results
    save_jsonl(all_results, args.output)
    logging.info(f"Judge complete! {len(all_results)} results saved to {args.output}")


if __name__ == "__main__":
    main()
