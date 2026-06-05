"""
VLGuard 数据适配器
数据格式: JSON 列表, 每项含 {id, image, safe, instr-resp: [...]}
三种子集:
  - unsafes: unsafe 图片 + unsafe 指令 -> 应拒绝
  - safe_unsafes: safe 图片 + unsafe 指令 -> 应拒绝
  - safe_safes: safe 图片 + safe 指令 -> 应正常回答
图片目录: data/vlguard/test_images/test/
"""

import json
import os
from typing import List, Dict
from .base import BenchmarkAdapter


class VLGuardAdapter(BenchmarkAdapter):
    benchmark_name = "vlguard"

    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        subsets = kwargs.get("subsets", ["unsafes", "safe_unsafes", "safe_safes"])
        data_file = os.path.join(data_dir, "test.json")

        # 图片目录: 优先 test_images/test/，其次 test/
        image_dir = os.path.join(data_dir, "test_images", "test")
        if not os.path.exists(image_dir):
            image_dir = os.path.join(data_dir, "test")

        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        sample_idx = 0

        for item in raw_data:
            image_rel = item["image"]
            image_path = os.path.join(image_dir, image_rel)
            is_safe_image = item["safe"]

            for ir in item.get("instr-resp", []):
                # unsafes: unsafe image + unsafe instruction
                if not is_safe_image and "instruction" in ir:
                    if "unsafes" in subsets:
                        samples.append({
                            "id": f"vlguard_{sample_idx}",
                            "question": ir["instruction"],
                            "image_path": image_path if os.path.exists(image_path) else None,
                            "metadata": {
                                "subset": "unsafes",
                                "original_id": item["id"],
                                "category": item.get("harmful_category", ""),
                                "subcategory": item.get("harmful_subcategory", ""),
                                "reference": ir.get("response", ""),
                            },
                        })
                        sample_idx += 1

                # safe image samples
                if is_safe_image:
                    # safe_unsafes: safe image + unsafe instruction
                    if "unsafe_instruction" in ir and "safe_unsafes" in subsets:
                        samples.append({
                            "id": f"vlguard_{sample_idx}",
                            "question": ir["unsafe_instruction"],
                            "image_path": image_path if os.path.exists(image_path) else None,
                            "metadata": {
                                "subset": "safe_unsafes",
                                "original_id": item["id"],
                                "category": "",
                                "reference": ir.get("response", ""),
                            },
                        })
                        sample_idx += 1

                    # safe_safes: safe image + safe instruction
                    if "safe_instruction" in ir and "safe_safes" in subsets:
                        samples.append({
                            "id": f"vlguard_{sample_idx}",
                            "question": ir["safe_instruction"],
                            "image_path": image_path if os.path.exists(image_path) else None,
                            "metadata": {
                                "subset": "safe_safes",
                                "original_id": item["id"],
                                "category": "",
                                "reference": ir.get("response", ""),
                            },
                        })
                        sample_idx += 1

        return samples
