"""
MSS-Bench Chat 数据适配器
数据格式: JSON 列表, 每项含 {intent, unsafe_image, image, Type, question}
图片目录: data/mssbench/chat/
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class MSSBenchAdapter(BenchmarkAdapter):
    benchmark_name = "mssbench"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        data_file = os.path.join(data_dir, "mss_bench_chat.json")
        image_dir = os.path.join(data_dir, "chat")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for idx, item in enumerate(raw_data):
            image_name = item["image"]
            image_path = os.path.join(image_dir, image_name)

            samples.append({
                "id": f"mssbench_{idx}",
                "question": item["question"],
                "image_path": image_path if os.path.exists(image_path) else None,
                "metadata": {
                    "category": item.get("Type", ""),
                    "intent": item.get("intent", ""),
                    "unsafe_image": item.get("unsafe_image", ""),
                    "image_name": image_name,
                },
            })

        return samples
