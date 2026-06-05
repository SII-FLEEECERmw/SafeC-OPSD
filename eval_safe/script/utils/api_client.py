"""
Qwen VL API 调用封装
"""

import os
import io
import json
import time
import base64
import random
import logging
import requests
from typing import Union, Optional
from PIL import Image


# --- API 配置 ---


# 默认 judge 模型
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen3-VL-235B-A22B-Instruct")

# 默认生成模型
DEFAULT_GEN_MODEL = os.environ.get("GEN_MODEL", "Qwen2.5-VL-7B-Instruct")
API_KEY = os.environ.get("JUDGE_API_KEY", "")
API_KEY_QWEN = os.environ.get("JUDGE_API_KEY_QWEN", "")
API_KEY_GEN = os.environ.get("GEN_API_KEY", "")  # 生成用的 API Key
# API 端点
API_URL_JUDGE = os.environ.get("JUDGE_API_URL", "")
API_URL_GEN = os.environ.get("GEN_API_URL", "")  # 生成 API 端点


def _encode_image_to_base64(image: Union[str, Image.Image]):
    """将图片编码为 Base64 字符串"""
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                logging.error(f"Image file not found: {image}")
                return None, None

            ext = os.path.splitext(image)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp",
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            with open(image, "rb") as f:
                image_bytes = f.read()

        elif isinstance(image, Image.Image):
            mime_type = "image/jpeg"
            buffer = io.BytesIO()
            img = image.convert("RGB") if image.mode != "RGB" else image
            img.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            logging.error(f"Unsupported image type: {type(image)}")
            return None, None

        return base64.b64encode(image_bytes).decode("utf-8"), mime_type

    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None, None


def _safe_post(session, url, **kwargs):
    """带指数退避重试的 POST 请求"""
    for attempt in range(5):
        try:
            resp = session.post(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.random()
            logging.warning(f"Request failed, retrying in {wait:.1f}s... ({e})")
            time.sleep(wait)
    raise Exception("Request failed after 5 retries")


def _extract_after_think_marker(text: str) -> str:
    """从模型输出中提取 </think> 标识符之后的内容"""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker):].strip()


def call_judge_api(
    prompt: str,
    image: Optional[Union[str, Image.Image]] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    调用 Judge API（支持多模态和纯文本）。

    Args:
        prompt: 文本 prompt
        image: 图片路径或 PIL Image 对象，None 表示纯文本
        model: 模型名称，默认使用 DEFAULT_JUDGE_MODEL
        max_retries: 最大重试次数

    Returns:
        模型输出文本，失败返回 None
    """
    model = model or DEFAULT_JUDGE_MODEL

    # 构建消息内容
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        b64, mime = _encode_image_to_base64(image)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

    # 选择 API 端点
    url = API_URL_JUDGE
    auth_key = API_KEY_QWEN or API_KEY

    if not url:
        raise ValueError("JUDGE_API_URL environment variable is required. Set it to your API endpoint.")
    if not auth_key:
        raise ValueError("JUDGE_API_KEY_QWEN or JUDGE_API_KEY environment variable is required.")


    data = {
        "stream": False,
        "model": model,
        "enable_sec_check": False,
        "messages": [{"role": "user", "content": content}],
    }

    for attempt in range(max_retries):
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Authorization": auth_key,
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=20, max_retries=3
        )
        session.mount("https://", adapter)

        try:
            resp = _safe_post(session, url, json=data, timeout=(10, 180))
            result = resp.json()

            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0].get("message", {}).get("content", "")
                return _extract_after_think_marker(output)
            else:
                logging.error(f"API returned unexpected response: {json.dumps(result, ensure_ascii=False)[:200]}")

        except Exception as e:
            logging.error(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        finally:
            session.close()

    return None


def call_gen_api(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    image: Optional[Union[str, Image.Image]] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    调用生成 API（支持多模态和纯文本）。

    Args:
        messages: 对话消息列表，格式如 [{"role": "user", "content": "text"}]
                 content 可以是字符串或列表
        model: 模型名称，默认使用 DEFAULT_GEN_MODEL
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        image: 图片路径或 PIL Image 对象
        max_retries: 最大重试次数

    Returns:
        模型输出文本，失败返回 None
    """
    model = model or DEFAULT_GEN_MODEL

    # 构建消息内容（与 call_judge_api 保持一致）
    processed_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        # 处理 content：可能是字符串或列表
        if isinstance(raw_content, str):
            # 纯文本：直接作为 text
            msg_content = [{"type": "text", "text": raw_content}]
        elif isinstance(raw_content, list):
            # 列表格式：需要检查是否有 {"type": "image"} 需要转换
            msg_content = []
            for item in raw_content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        # 这是 base.py 返回的格式，需要转换
                        if image is not None:
                            b64, mime = _encode_image_to_base64(image)
                            if b64:
                                msg_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                })
                    else:
                        msg_content.append(item)
                else:
                    msg_content.append({"type": "text", "text": str(item)})
        else:
            msg_content = [{"type": "text", "text": str(raw_content)}]

        # 如果外部传入了 image 参数且还没有处理过，添加到开头
        if image is not None and role == "user":
            # 检查是否已经有 image_url
            has_image = any(
                isinstance(c, dict) and c.get("type") == "image_url"
                for c in msg_content
            )
            if not has_image:
                b64, mime = _encode_image_to_base64(image)
                if b64:
                    # 图片添加到开头（符合视觉模型习惯）
                    msg_content.insert(0, {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })

        processed_messages.append({
            "role": role,
            "content": msg_content
        })

    # 选择 API 端点（与 call_judge_api 一致）
    url = API_URL_GEN or API_URL_JUDGE
    # 优先使用 GEN_API_KEY，如果没有则回退到 JUDGE_API_KEY_QWEN
    auth_key = API_KEY_GEN or API_KEY_QWEN or API_KEY
    if not url:
        raise ValueError("GEN_API_URL or JUDGE_API_URL environment variable is required.")
    if not auth_key:
        raise ValueError("GEN_API_KEY or JUDGE_API_KEY_QWEN environment variable is required.")

    # 构建请求数据（与 call_judge_api 保持一致）
    data = {
        "stream": False,
        "model": model,
        "enable_sec_check": False,
        "messages": processed_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 打印调试信息：请求体（截断长文本）
    debug_data = json.dumps(data, ensure_ascii=False)
    if len(debug_data) > 500:
        debug_data = debug_data[:500] + "..."
    logging.info(f"Request data: {debug_data}")

    for attempt in range(max_retries):
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Authorization": auth_key,
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=20, max_retries=3
        )
        session.mount("https://", adapter)

        try:
            resp = _safe_post(session, url, json=data, timeout=(10, 180))
            result = resp.json()

            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0].get("message", {}).get("content", "")
                return output.strip()
            else:
                logging.error(f"API returned unexpected response: {json.dumps(result, ensure_ascii=False)[:500]}")

        except requests.exceptions.HTTPError as e:
            # 打印详细的错误信息
            try:
                error_body = e.response.text if e.response else "No response body"
                logging.error(f"HTTP Error {e.response.status_code if e.response else 'N/A'}: {error_body}")
            except:
                logging.error(f"HTTP Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        finally:
            session.close()

    return None


def call_gen_api_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    image: Optional[Union[str, Image.Image]] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    简化的生成 API 调用（单轮对话）。

    Args:
        prompt: 用户 prompt
        model: 模型名称，默认使用 DEFAULT_GEN_MODEL
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        image: 图片路径或 PIL Image 对象
        max_retries: 最大重试次数

    Returns:
        模型输出文本，失败返回 None
    """
    messages = [{"role": "user", "content": prompt}]
    return call_gen_api(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        image=image,
        max_retries=max_retries,
    )
