"""
BenchmarkAdapter 基类
定义统一的数据加载和 prompt 生成接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


# 全局配置：constitution 模式
# mode: "none" | "prefix" | "system"
#   - "none":   不嵌入 policy（默认）
#   - "prefix": 在 user message 文本前拼接 policy
#   - "system": 将 policy 作为独立的 system message 插入
CONSTITUTION_MODE = "none"
CONSTITUTION_POLICY = ""


def set_constitution_mode(mode: str = "none", policy_text: str = ""):
    """
    设置 constitution 模式。

    Args:
        mode: constitution 嵌入模式，可选值:
            - "none":   不嵌入 policy（默认）
            - "prefix": 在 user message 文本前拼接 policy
            - "system": 将 policy 作为独立的 system message 插入
        policy_text: policy 文本内容
    """
    global CONSTITUTION_MODE, CONSTITUTION_POLICY
    if mode not in ("none", "prefix", "system"):
        raise ValueError(f"Invalid constitution mode: {mode}. Must be one of: none, prefix, system")
    CONSTITUTION_MODE = mode
    CONSTITUTION_POLICY = policy_text


class BenchmarkAdapter(ABC):
    """所有 benchmark 数据适配器的基类"""

    # 子类需要设置的属性
    benchmark_name: str = ""

    @abstractmethod
    def load(self, data_dir: str, **kwargs) -> List[Dict]:
        """
        加载数据并返回统一格式的列表。

        每个元素为:
        {
            "id": str,              # 唯一标识
            "question": str,        # 用户问题/指令
            "image_path": str|None, # 图片绝对路径，纯文本任务为 None
            "metadata": {           # benchmark 特有的元数据
                "category": str,
                "subset": str,      # 可选
                ...
            }
        }
        """
        raise NotImplementedError

    def get_gen_prompt(self, item: Dict) -> str:
        """
        返回给待测模型的 prompt。
        默认直接返回 question，子类可覆盖以添加 system prompt 等。
        """
        return item["question"]

    def get_gen_messages(self, item: Dict) -> List[Dict]:
        """
        返回给待测模型的 chat messages 格式。
        子类可覆盖以自定义消息格式。

        constitution 模式:
        - "prefix": 在 user message 的文本前拼接 policy
        - "system": 将 policy 作为独立的 system message 插入 messages 列表头部
        """
        messages = []

        # 构建消息列表
        if CONSTITUTION_MODE == "system" and CONSTITUTION_POLICY:
            # system 模式：将 policy 作为独立的 system message
            messages.append({"role": "system", "content": CONSTITUTION_POLICY})

        # 构建 user message content
        content = []
        if item.get("image_path"):
            content.append({"type": "image"})

        # 构建 prompt 文本
        prompt_text = self.get_gen_prompt(item)

        # prefix 模式：在 user message 文本前拼接 policy
        if CONSTITUTION_MODE == "prefix" and CONSTITUTION_POLICY:
            prompt_text = f"{CONSTITUTION_POLICY}\n\n{prompt_text}"

        content.append({"type": "text", "text": prompt_text})

        messages.append({"role": "user", "content": content})

        return messages
