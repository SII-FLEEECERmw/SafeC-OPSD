"""Client-side reward function for OPSDGate.

This module exposes ``compute_onpolicy_KL_reward`` which is loaded by verl's
``custom_reward_function.path/name`` mechanism. For each sample it sends
``(raw_prompt_ids, responses, response_masks, multi_modal_data)`` to the
teacher reward gateway's ``/compute_logprobs`` endpoint and converts the
teacher logprobs into a token-level OPSD KL reward.
"""

import base64
import os
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
from PIL import Image
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

session = requests.Session()

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
LOGPROB_SERVER_URL = f"{SERVER_URL}/compute_logprobs"


class AdaptiveGPUKeepAlive:
    """Background loop that keeps the local GPU warm to avoid power-state
    oscillation while the actor is blocked on the teacher HTTP response."""

    def __init__(
        self,
        device: torch.device,
        target_util: float = 30.0,
        cycle_time: float = 0.2,
        mb: float = 2.0,
        dtype: torch.dtype = torch.bfloat16,
        use_gpu_sleep: bool = True,
        inner_ops: int = 2000,
        kp: float = 0.02,
        max_duty: float = 0.95,
        min_duty: float = 0.0,
    ):
        self.device = device
        self.target_util = float(target_util)
        self.cycle_time = float(cycle_time)
        self.mb = float(mb)
        self.dtype = dtype
        self.use_gpu_sleep = bool(use_gpu_sleep)
        self.inner_ops = int(inner_ops)
        self.kp = float(kp)
        self.max_duty = float(max_duty)
        self.min_duty = float(min_duty)

        if isinstance(device, str):
            device = torch.device(device)
        self.gpu_index = device.index if device.index is not None else 0

        self._stop = threading.Event()
        self._th = None
        self._started = False

    def start(self):
        if self._started:
            return
        if not torch.cuda.is_available():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        self._started = True

    def stop(self):
        if not self._started:
            return
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        self._th = None
        self._started = False

    @torch.inference_mode()
    def _run(self):
        torch.cuda.set_device(self.gpu_index)

        bytes_per_elem = torch.tensor([], dtype=self.dtype).element_size()
        n_elem = max(1, int(self.mb * 1024 * 1024 / bytes_per_elem))
        x = torch.empty(n_elem, device=self.device, dtype=self.dtype).uniform_(-1, 1)

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(self.gpu_index)

        duty = 0.30
        try:
            while not self._stop.is_set():
                util = nvmlDeviceGetUtilizationRates(h).gpu
                err = self.target_util - util
                duty = max(self.min_duty, min(self.max_duty, duty + self.kp * (err / 100.0)))

                work_time = self.cycle_time * duty
                t0 = time.time()

                while (time.time() - t0) < work_time and (not self._stop.is_set()):
                    for _ in range(self.inner_ops):
                        x.mul_(1.0001).add_(0.0001)

                    if self.use_gpu_sleep:
                        torch.cuda._sleep(200_000)

                    torch.cuda.synchronize()

                spent = time.time() - t0
                left = self.cycle_time - spent
                if left > 0 and (not self._stop.is_set()):
                    time.sleep(left)

        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass


def encode_pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _norm_raw_prompt_ids(raw_prompt_ids) -> List[int]:
    if raw_prompt_ids is None or raw_prompt_ids == []:
        raise ValueError("No raw_prompt_ids found")
    if isinstance(raw_prompt_ids, torch.Tensor):
        raw_prompt_ids = raw_prompt_ids.detach().cpu().tolist()
    if len(raw_prompt_ids) > 0 and isinstance(raw_prompt_ids[0], list):
        raw_prompt_ids = raw_prompt_ids[0]
    return list(raw_prompt_ids)


def _norm_responses(responses) -> List[int]:
    if responses is None:
        raise ValueError("No responses found")
    if isinstance(responses, torch.Tensor):
        if responses.dim() == 2 and responses.size(0) == 1:
            responses = responses[0]
        return responses.detach().cpu().tolist()
    if len(responses) > 0 and isinstance(responses[0], list):
        responses = responses[0]
    return list(responses)


def _norm_attention_mask(attn) -> List[int]:
    if attn is None:
        raise ValueError("No attention mask found")
    if isinstance(attn, torch.Tensor):
        if attn.dim() == 2 and attn.size(0) == 1:
            attn = attn[0]
        return attn.detach().cpu().tolist()
    if len(attn) > 0 and isinstance(attn[0], list):
        attn = attn[0]
    return list(attn)


def _norm_student_logprobs(slp, resp_len: int) -> torch.Tensor:
    if slp is None:
        raise ValueError("Critical: 'rollout_log_probs' missing.")
    if not isinstance(slp, torch.Tensor):
        slp = torch.tensor(slp)

    if slp.dim() == 2 and slp.size(0) == 1:
        slp = slp[0]
    elif slp.dim() != 1:
        raise ValueError(f"rollout_log_probs per-sample should be 1D (got {slp.shape})")

    if slp.numel() != resp_len:
        slp = slp[-resp_len:]
    return slp


def _process_mm_item(mm_item: Dict[str, Any]) -> Dict[str, Any]:
    new_item: Dict[str, Any] = {}
    for k, v in mm_item.items():
        if k == "image":
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Image.Image):
                new_item["image"] = [encode_pil_to_base64(img) for img in v]
            elif isinstance(v, Image.Image):
                new_item["image"] = encode_pil_to_base64(v)
            else:
                new_item["image"] = v
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            new_item[k] = v.tolist()
        else:
            new_item[k] = v
    return new_item


def compute_onpolicy_KL_reward(
    extra_info_list: List[Dict[str, Any]],
    timeout: float = 900.0,
    keepalive_target_util: float = 80.0,
) -> torch.Tensor:
    """Compute the token-level OPSD KL reward for a chunk of samples.

    Returns a tensor of shape ``[bs, max_len]`` where each row holds the
    clipped ``-(student_logp - teacher_logp) + 0.02`` reward at every valid
    response token.
    """
    if len(extra_info_list) == 0:
        raise ValueError("extra_info_list is empty")

    first_slp = extra_info_list[0].get("rollout_log_probs")
    if not isinstance(first_slp, torch.Tensor):
        first_slp = torch.tensor(first_slp)
    device = first_slp.device
    dtype = first_slp.dtype

    raw_prompt_ids_batch: List[List[int]] = []
    responses_batch: List[List[int]] = []
    response_masks_batch: List[List[int]] = []
    mm_batch: Optional[List[Dict[str, Any]]] = []

    student_lp_tensors: List[torch.Tensor] = []
    resp_lens: List[int] = []

    for ei in extra_info_list:
        raw_prompt_ids = _norm_raw_prompt_ids(ei.get("raw_prompt_ids"))
        responses = _norm_responses(ei.get("responses"))
        resp_len = len(responses)

        response_mask = _norm_attention_mask(ei.get("attention_mask"))
        slp = _norm_student_logprobs(ei.get("rollout_log_probs"), resp_len=resp_len)

        raw_prompt_ids_batch.append(raw_prompt_ids)
        responses_batch.append(responses)
        response_masks_batch.append(response_mask)
        student_lp_tensors.append(slp)
        resp_lens.append(resp_len)

        mm = ei.get("multi_modal_inputs", None)
        if mm:
            if isinstance(mm, dict):
                mm_item: Any = mm
            elif isinstance(mm, list) and len(mm) > 0 and isinstance(mm[0], dict):
                mm_item = mm
            else:
                mm_item = {}
            if isinstance(mm_item, dict):
                mm_batch.append(_process_mm_item(mm_item))
            else:
                mm_batch.append({"items": [_process_mm_item(x) for x in mm_item]})
        else:
            mm_batch.append({})

    if all((not x) for x in mm_batch):
        mm_batch = None

    policy_str_list = [ei.get("policy_str", "") for ei in extra_info_list]

    payload = {
        "raw_prompt_ids": raw_prompt_ids_batch,
        "responses": responses_batch,
        "response_masks": response_masks_batch,
        "multi_modal_data": mm_batch,
        "policy_str_list": policy_str_list,
    }

    keeper = AdaptiveGPUKeepAlive(
        device=device,
        target_util=keepalive_target_util,
        cycle_time=0.2,
        mb=2.0,
        dtype=torch.bfloat16,
        use_gpu_sleep=True,
    )

    try:
        keeper.start()
        resp = session.post(LOGPROB_SERVER_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        response_data = resp.json()
    finally:
        keeper.stop()

    teacher_logprobs_list = response_data.get("logprobs")
    if teacher_logprobs_list is None:
        raise ValueError("Server returned no 'logprobs'")

    teacher = torch.tensor(teacher_logprobs_list, device=device, dtype=dtype)
    mask = torch.tensor(response_masks_batch, device=device, dtype=dtype)

    max_len = teacher.shape[1]
    student = torch.zeros((len(student_lp_tensors), max_len), device=device, dtype=dtype)
    for i, slp in enumerate(student_lp_tensors):
        L = min(slp.numel(), max_len)
        student[i, :L] = slp[:L].to(device=device, dtype=dtype)

    min_len = min(teacher.shape[1], student.shape[1], mask.shape[1])
    teacher = teacher[:, :min_len]
    student = student[:, :min_len]
    mask = mask[:, :min_len]

    kl = (student - teacher) * mask
    reward_tok = -kl
    reward_tok = torch.clamp(reward_tok, min=-3.0, max=5.0) + 0.02

    return reward_tok
