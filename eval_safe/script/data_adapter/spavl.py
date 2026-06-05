"""
SPA-VL 数据适配器
数据格式: JSON 列表, 每项含 {question, class1, class2, class3, image}
图片目录: data/spavl/images/
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class SPAVLAdapter(BenchmarkAdapter):
    benchmark_name = "spavl"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        data_file = os.path.join(data_dir, "evaluation.json")
        image_dir = os.path.join(data_dir, "images")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for idx, item in enumerate(raw_data):
            image_name = item["image"]
            image_path = os.path.join(image_dir, image_name)

            samples.append({
                "id": f"spavl_{idx}",
                "question": item["question"],
                "image_path": image_path if os.path.exists(image_path) else None,
                "metadata": {
                    "category": item.get("class1", ""),
                    "class2": item.get("class2", ""),
                    "class3": item.get("class3", ""),
                    "image_name": image_name,
                },
            })

        return samples
