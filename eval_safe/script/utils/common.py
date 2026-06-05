"""
通用工具函数：文件读写、日志配置、YAML 加载等
"""

import json
import os
import logging
import yaml
from typing import List, Dict, Any, Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """配置日志"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_yaml(path: str) -> Dict:
    """加载 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存为 JSONL 文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """保存为 JSON 文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def get_existing_ids(output_path: str) -> set:
    """从已有的输出文件中获取已完成的 ID 集合（用于断点续跑）"""
    if not os.path.exists(output_path):
        return set()
    existing = load_jsonl(output_path)
    return {item["id"] for item in existing if "id" in item}


def merge_jsonl_files(file_list: List[str], output_path: str):
    """合并多个 JSONL 文件为一个"""
    all_data = []
    for fpath in sorted(file_list):
        if os.path.exists(fpath):
            all_data.extend(load_jsonl(fpath))
    save_jsonl(all_data, output_path)
    return len(all_data)


def get_benchmark_config(config_path: str, benchmark: str) -> Dict:
    """从 benchmarks.yaml 获取指定 benchmark 的配置"""
    all_config = load_yaml(config_path)
    if benchmark not in all_config:
        raise ValueError(
            f"Benchmark '{benchmark}' not found in config. "
            f"Available: {list(all_config.keys())}"
        )
    return all_config[benchmark]


def get_all_benchmarks(config_path: str) -> List[str]:
    """获取所有已注册的 benchmark 名称"""
    all_config = load_yaml(config_path)
    return list(all_config.keys())
