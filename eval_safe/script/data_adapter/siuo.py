"""
SIUO 数据适配器
数据格式: JSON 列表, 每项含 {question_id, image, question, category, safety_warning, reference_answer, ...}
图片目录: data/siuo/images/
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class SIUOAdapter(BenchmarkAdapter):
    benchmark_name = "siuo"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        data_file = os.path.join(data_dir, "siuo_gen_bailing_eval.json")
        image_dir = os.path.join(data_dir, "images")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            image_name = item["image"]
            image_path = os.path.join(image_dir, image_name)

            samples.append({
                "id": f"siuo_{item['question_id']}",
                "question": item["question"],
                "image_path": image_path if os.path.exists(image_path) else None,
                "metadata": {
                    "question_id": item["question_id"],
                    "category": item.get("category", ""),
                    "safety_warning": item.get("safety_warning", ""),
                    "reference_answer": item.get("reference_answer", ""),
                    "image_name": image_name,
                },
            })

        return samples
