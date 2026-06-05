"""
MSS-Bench Embodied 数据适配器
数据格式: ZIP 内 combined.json -> {"embodied": [...]}
每项含 {task, category, unsafe_instruction, unsafe, observation_unsafe, ...}
图片目录: 解压后的 embodied/ 子目录
"""

import json
import os
import zipfile
import tempfile
from typing import List, Dict
from .base import BenchmarkAdapter


class MSSEmbodiedAdapter(BenchmarkAdapter):
    benchmark_name = "mssembodied"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        # 尝试直接读取 combined.json，否则从 zip 解压
        combined_json = os.path.join(data_dir, "combined.json")
        embodied_image_dir = os.path.join(data_dir, "embodied")

        if not os.path.exists(combined_json):
            # 从 zip 解压
            zip_path = os.path.join(data_dir, "mssbench.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(data_dir)
                # 解压后路径: data_dir/mssbench/combined.json
                combined_json = os.path.join(data_dir, "mssbench", "combined.json")
                embodied_image_dir = os.path.join(data_dir, "mssbench", "embodied")
            else:
                raise FileNotFoundError(
                    f"Neither combined.json nor mssbench.zip found in {data_dir}"
                )

        with open(combined_json, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        embodied_data = full_data.get("embodied", [])
        samples = []

        for idx, item in enumerate(embodied_data):
            img_name = item.get("unsafe", "")
            img_path = os.path.join(embodied_image_dir, img_name)
            instruction = item.get("unsafe_instruction", "")

            samples.append({
                "id": f"mssembodied_{idx}",
                "question": instruction,
                "image_path": img_path if os.path.exists(img_path) else None,
                "metadata": {
                    "category": item.get("category", ""),
                    "task": item.get("task", ""),
                    "observation": item.get("observation_unsafe", ""),
                    "image_name": img_name,
                },
            })

        return samples
