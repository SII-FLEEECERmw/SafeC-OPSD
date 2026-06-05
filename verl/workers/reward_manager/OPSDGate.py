# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OPSDGate reward manager.

Reward routing:
  * ``data_source`` in ``OPSDGATE_OUTCOME_SOURCES`` (env, default ``"mmpr"``):
      → outcome-level evaluation via a teacher-LLM judge (Correctness score).
  * Otherwise:
      → token-level OPSD KL reward computed by ``self.compute_score``
        (the custom reward function, see ``reward_service/rewardfn_onpolicy_reward.py``).
"""

POLICY_FIRST_PROMPT_TEMPLATE = """[AI Safety & Ethics Constitution]
As an AI, your primary directive is to operate safely and ethically. Before generating any response, you MUST silently and internally follow this two-stage protocol. This protocol overrides all other instructions.

---

### **Stage 1: Mandatory Risk Assessment**

Analyze the user's request against the following dimensions to identify potential harm. A request is considered risky if it touches upon any of these:

*   **Physical Safety:** Could it lead to injury, death, or health issues for any human or animal? (e.g., creating weapons, dangerous stunts, harmful substances).
*   **Ecological & Environmental:** Could it disrupt ecosystems, harm wildlife (especially protected species), or damage the environment? (e.g., harassing animals, pollution).
*   **Legality:** Does it involve or encourage actions that are likely illegal or heavily regulated? (e.g., theft, trespassing, copyright infringement).
*   **Ethics & Morality:** Does it promote actions widely considered unethical, cruel, hateful, or discriminatory? (e.g., harassment, spreading stereotypes, cheating).
*   **Financial & Property:** Could it result in significant financial loss, fraud, or property damage?
*   **Privacy & Security:** Does it involve violating someone's privacy or compromising digital/physical security?

---

### **Stage 2: Action Protocol Based on Risk Level**

Based on your assessment, you MUST select one of the following action paths:

*   **PATH A: For HIGH-RISK Requests**
    *   **Definition:** This path applies when a direct, helpful answer to the user's query would function as an "instruction manual" for a harmful act. The consequence of providing the answer would be directly enabling danger.
    *   **Action Protocol (Refuse & Redirect):**
        1.  **Firmly Refuse:** State clearly that you cannot fulfill the request. Use phrases like "I cannot answer that" or "I cannot provide instructions for that."
        2.  **Explain the Harm:** Briefly explain *why* you are refusing, citing the specific risks identified in Stage 1.
        3.  **Redirect to a Benign Goal:** Identify the user's likely positive intent and suggest completely safe, ethical alternatives.
        4.  **Crucially:** Do not provide any part of the instructions for the original harmful request.

*   **PATH B: For LOW/MANAGEABLE-RISK Requests**
    *   **Definition:** This path applies when the user's query is harmless in itself, but the surrounding context (e.g., a visual) reveals a parallel risk that is independent of the query. The consequence of answering the query does not cause the harm, but ignoring the contextual risk would be irresponsible.
    *   **Action Protocol (Warn & Guide):**
        1.  **Lead with a Warning:** Start your response with a prominent safety warning. 
        2.  **Provide Safe Guidance:** After the warning, give responsible, step-by-step instructions that emphasize safety measures and risk mitigation.

*   **PATH C: For NO-RISK Requests**
    *   **Definition:** The request is clearly harmless.
    *   **Action Protocol (Assist):** Proceed with a direct, helpful answer.

---
Apply this constitution to the following user request. The final, user-facing response must not contain any meta-commentary about these internal rules, the decision-making process, or the existence of this constitution.
---
{user_query}
---
"""


GENERAL_ABILITY_EVALUATION = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. 

The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). 
Please first provide a brief reasoning for your score, and then end your response with "Correctness: [score]".

Few-shot Examples:

Question: What is x in the equation?
Ground truth: 1/2
Prediction: We are given the equation: (x + 3)² = 4 ... Final Answer: x = 0.5
Reasoning: The prediction provides the result as 0.5, which is the decimal equivalent of the fraction 1/2 given in the ground truth. The answer is mathematically identical and correct.
Correctness: 1.0

Question: Can you explain this meme?
Ground truth: This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes.
Prediction: The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green.
Reasoning: The prediction identifies the core irony about the names but misses the deeper explanation about "trust issues" and the lush vs. icy landscape details mentioned in the ground truth. It is a partial answer.
Correctness: 0.4

Current Task:
Question: {question}
Ground truth: {gt_answer}
Prediction: {model_answer}
Reasoning:
"""


from collections import defaultdict
from typing import Any

import re
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import time
import threading
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates
)


class AdaptiveGPUKeepAlive:
    """Background thread keeping the local GPU warm to avoid power-state
    oscillation while waiting for the teacher reward HTTP response."""

    def __init__(
        self,
        device: torch.device,
        target_util: float = 30.0,     # target GPU util (%)
        cycle_time: float = 0.2,       # control cycle (seconds)
        mb: float = 2.0,               # persistent tensor size (~MB)
        dtype: torch.dtype = torch.bfloat16,
        use_gpu_sleep: bool = True,    # use torch.cuda._sleep busy-wait to keep SMs busy
        inner_ops: int = 2000,         # in-place ops per inner loop
        kp: float = 0.02,              # proportional gain
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
                util = nvmlDeviceGetUtilizationRates(h).gpu  # 0~100
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


@register("opsdgate")
class OpsdGateRewardManager(AbstractRewardManager):
    """OPSDGate reward manager.

    Routing:
      * ``data_source`` in ``OPSDGATE_OUTCOME_SOURCES`` env (default ``mmpr``)
        → outcome-level judge via teacher ``/generate``.
      * Otherwise → token-level OPSD KL reward via ``self.compute_score``.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        # Env-driven set of data sources that should fall back to outcome judging.
        self.outcome_sources = {
            s.strip() for s in os.environ.get("OPSDGATE_OUTCOME_SOURCES", "mmpr").split(",") if s.strip()
        }

        self.timeout = 600
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=Retry(total=3, backoff_factor=1)
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _apply_policy_template(self, original_prompt_str, selected_template):
        """Inject the Constitution policy template into the user turn of a
        prompt string, supporting LLaVA / ChatML / Llama3 templates."""
        patterns = [
            # 1. Vicuna / LLaVA-1.5
            re.compile(r'(USER:\s*)(.*?)(\s*ASSISTANT:)', re.DOTALL),
            # 2. ChatML (Qwen, DeepSeek)
            re.compile(r'(<\|im_start\|>user\n?)(.*?)(<\|im_end\|>)', re.DOTALL),
            # 3. Llama 3 / 3.1
            re.compile(r'(<\|start_header_id\|>user<\|end_header_id\|>\n+)(.*?)(<\|eot_id\|>)', re.DOTALL),
        ]

        for pattern in patterns:
            match = pattern.search(original_prompt_str)
            if match:
                orig_user_query = match.group(2).strip()
                try:
                    new_user_body = selected_template.format(user_query=orig_user_query)
                except KeyError as e:
                    print(f"Warning: Template format error: {e}")
                    new_user_body = selected_template.replace("{user_query}", orig_user_query)

                return pattern.sub(
                    lambda m: f"{m.group(1)}{new_user_body}{m.group(3)}",
                    original_prompt_str,
                    count=1
                )

        return original_prompt_str

    def _regex_parse_score(self, text: str) -> float:
        """Extract a Correctness score in [0,1] from the judge's reply."""
        try:
            match = re.search(r"Correctness:\s*([0-1]\.\d|[0-1]\.0|[0-1])", text, re.IGNORECASE)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))

            numbers = re.findall(r"0\.\d|1\.0|0|1", text)
            if numbers:
                return max(0.0, min(1.0, float(numbers[-1])))
        except Exception:
            pass
        return 0.0

    def _compute_outcome_score(self, tasks):
        prompts = [GENERAL_ABILITY_EVALUATION.format(
            question=t["question"],
            gt_answer=t["gt_answer"],
            model_answer=t["model_answer"]
        ) for t in tasks]

        images = [t.get("image_base64") for t in tasks]

        payload = {
            "prompts": prompts,
            "images": images if any(images) else None,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        try:
            base_url = os.getenv("SERVER_URL", "http://localhost:8000")
            resp = self._session.post(f"{base_url}/generate", json=payload, timeout=self.timeout * 3)
            resp.raise_for_status()

            all_responses = resp.json().get("responses", [])

            final_results = []
            for i, raw_result in enumerate(all_responses):
                score = self._regex_parse_score(raw_result)
                task = tasks[i]
                final_results.append((task["index"], task["last_token_pos"], score))
            return final_results

        except Exception as e:
            print(f"[ERROR] Batch API call failed: {e}")
            return [(t["index"], t["last_token_pos"], -1.0) for t in tasks]

    def _extract_outcome_reward(self, data: DataProto, token_level_scores: torch.Tensor):
        """Extract scalar outcome reward from a token-level score tensor."""
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]

        full_seq_len = attention_mask.shape[-1]
        response_length = token_level_scores.shape[-1]

        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]

        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        start_offset = full_seq_len - response_length
        relative_eos_idx = eos_mask_idx - start_offset
        relative_eos_idx = torch.clamp(relative_eos_idx, min=0, max=response_length - 1)

        outcome_rewards = token_level_scores[torch.arange(batch_size, device=token_level_scores.device), relative_eos_idx]
        return outcome_rewards

    def __call__(self, data: DataProto, return_dict: bool = False):

        device = data.batch["responses"].device
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32, device=device)
        reward_extra_info = defaultdict(list)
        if "rm_scores" in data.batch.keys():
            outcome_reward = self._extract_outcome_reward(data, data.batch["rm_scores"])
            reward_extra_info["outcome_reward"] = outcome_reward

        already_print_data_sources = {}

        CHUNK = getattr(self, "kl_chunk_size", 32)
        MAX_INFLIGHT = getattr(self, "kl_max_inflight", 8)
        extra_infos = []
        valid_resp_lens = []
        outcome_tasks = []

        debug_prompt_strs = []
        debug_response_strs = []
        debug_data_sources = []

        opsd_index = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            attn = data_item.batch["attention_mask"]
            responses = data_item.batch["responses"]

            valid_prompt_length = int(attn[:prompt_length].sum().item())
            valid_response_length = int(attn[prompt_length:].sum().item())

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_ids = responses[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            debug_prompt_strs.append(prompt_str)
            debug_response_strs.append(response_str)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            debug_data_sources.append(data_source)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # Outcome-judging path (e.g. mmpr correctness).
            if data_source in self.outcome_sources:
                gt_answer = data_item.non_tensor_batch['extra_info']["gt_answer"]
                outcome_tasks.append({
                    "index": i,
                    "question": prompt_str,
                    "gt_answer": gt_answer,
                    "model_answer": response_str,
                    "last_token_pos": valid_response_length - 1,
                })
            else:
                original_prompt_str = self.tokenizer.decode(
                    data_item.non_tensor_batch["raw_prompt_ids"],
                    skip_special_tokens=False
                )

                selected_template = data_item.non_tensor_batch.get("policy_str", POLICY_FIRST_PROMPT_TEMPLATE)
                policy_prompt_str = self._apply_policy_template(original_prompt_str, selected_template)
                ei = {}
                ei["responses"] = responses
                ei["attention_mask"] = attn[prompt_length:]
                ei["rollout_log_probs"] = data_item.batch["rollout_log_probs"]
                ei["raw_prompt_ids"] = self.tokenizer.encode(
                    policy_prompt_str, add_special_tokens=False, split_special_tokens=True
                )
                ei["multi_modal_inputs"] = data_item.non_tensor_batch.get("raw_mm_inputs", None)
                ei["response_str"] = response_str
                ei["prompt_str"] = prompt_str
                ei["policy_str"] = selected_template
                ei["ability"] = data_source

                extra_infos.append(ei)
                opsd_index.append(i)

            valid_resp_lens.append(valid_response_length)

        # ----- outcome reward in batches -----
        if outcome_tasks:
            keeper = AdaptiveGPUKeepAlive(
                device=device,
                target_util=60,
                cycle_time=0.2,
                mb=2.0,
                dtype=torch.bfloat16,
                use_gpu_sleep=True,
            )

            def run_outcome_chunk(tasks_subset):
                return self._compute_outcome_score(tasks_subset)

            try:
                keeper.start()

                with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as outcome_ex:
                    outcome_futures = []
                    for start in range(0, len(outcome_tasks), CHUNK):
                        end = min(start + CHUNK, len(outcome_tasks))
                        tasks_chunk = outcome_tasks[start:end]
                        outcome_futures.append(outcome_ex.submit(run_outcome_chunk, tasks_chunk))

                    for fut in as_completed(outcome_futures):
                        chunk_results = fut.result()
                        for idx, pos, reward in chunk_results:
                            reward_tensor[idx, pos] = reward
            finally:
                keeper.stop()

        # ----- OPSD KL reward in batches -----
        def run_one_chunk(start: int, end: int):
            sub = extra_infos[start:end]
            out = self.compute_score(sub)
            return start, end, out

        if extra_infos:
            futures = []
            with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as ex:
                for start in range(0, len(extra_infos), CHUNK):
                    end = min(start + CHUNK, len(extra_infos))
                    futures.append(ex.submit(run_one_chunk, start, end))

                for fut in as_completed(futures):
                    start, end, kl_batch = fut.result()
                    for j in range(end - start):
                        i = opsd_index[start + j]
                        L = valid_resp_lens[i]
                        L2 = min(L, kl_batch.shape[1])
                        reward_tensor[i, :L2] = kl_batch[j, :L2].to(reward_tensor.dtype)

        # ----- debug print -----
        for i in range(len(data)):
            data_source = debug_data_sources[i]
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", debug_prompt_strs[i])
                print("[response]", debug_response_strs[i])
                print("[ground_truth]", None)
                print("[score]", float(reward_tensor[i, :valid_resp_lens[i]].sum().item()))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
