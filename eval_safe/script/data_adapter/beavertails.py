"""
BeaverTails-V 数据适配器
数据格式: JSON 列表, 每项含 {question, image, category, image_severity, is_response_safe}
图片目录: data/beavertails/images/
注意: image 字段格式为 "evaluation/animal_abuse_evaluation_0.jpg"
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class BeaverTailsAdapter(BenchmarkAdapter):
    benchmark_name = "beavertails"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        data_file = os.path.join(data_dir, "evaluation_cleaned.json")
        image_dir = os.path.join(data_dir, "images")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for idx, item in enumerate(raw_data):
            image_rel = item["image"]  # e.g. "evaluation/animal_abuse_evaluation_0.jpg"
            image_path = os.path.join(image_dir, image_rel)

            # 兼容: 如果带子目录的路径不存在，尝试只用文件名
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, os.path.basename(image_rel))

            samples.append({
                "id": f"beavertails_{idx}",
                "question": item["question"],
                "image_path": image_path if os.path.exists(image_path) else None,
                "metadata": {
                    "category": item.get("category", ""),
                    "image_severity": item.get("image_severity", 0),
                    "is_response_safe": item.get("is_response_safe", ""),
                    "image_rel": image_rel,
                },
            })

        return samples
