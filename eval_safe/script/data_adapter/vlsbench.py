"""
VLSBench 数据适配器
数据格式: JSON 列表, 每项含 {instruction_id, instruction, image_path, category, sub_category, source, ...}
图片目录: data/vlsbench/ (image_path 字段为相对路径如 "imgs/0.png")
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class VLSBenchAdapter(BenchmarkAdapter):
    benchmark_name = "vlsbench"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        # 优先 data/data.json，兼容 data.json
        data_file = os.path.join(data_dir, "data", "data.json")
        if not os.path.exists(data_file):
            data_file = os.path.join(data_dir, "data.json")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            img_rel_path = item["image_path"]  # e.g. "imgs/0.png"
            img_path = os.path.join(data_dir, img_rel_path)

            samples.append({
                "id": f"vlsbench_{item['instruction_id']}",
                "question": item["instruction"],
                "image_path": img_path if os.path.exists(img_path) else None,
                "metadata": {
                    "instruction_id": item["instruction_id"],
                    "category": item.get("category", ""),
                    "sub_category": item.get("sub_category", ""),
                    "source": item.get("source", ""),
                    "safety_reason": item.get("safety_reason", ""),
                    "image_description": item.get("image_description", ""),
                },
            })

        return samples
