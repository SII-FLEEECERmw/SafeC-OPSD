"""vLLM worker behind the OPSDGate reward gateway.

Exposes two endpoints used by the OPSDGate training pipeline:
  * ``POST /compute_logprobs`` — teacher logprob computation for OPSD KL reward.
  * ``POST /generate`` — judge generation for the outcome-level path.
"""

import base64
import gc
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from vllm import LLM, SamplingParams

LOG_PATH = os.environ.get("REWARD_WORKER_LOG", os.path.join(os.getcwd(), "reward_server.log"))
logger.add(LOG_PATH, rotation="500 MB", encoding="utf-8")

app = FastAPI(title="OPSDGate Reward vLLM Worker")


class RewardRequest(BaseModel):
    raw_prompt_ids: List[List[int]]      # [bsz, dy_len]
    responses: List[List[int]]           # [bsz, response_len]
    response_masks: List[List[int]]      # [bsz, response_len]
    multi_modal_data: Optional[List[Dict[str, Any]]] = None  # [{"image": [<PIL or base64>]}]


class GenerateRequest(BaseModel):
    prompts: List[str]
    images: Optional[List[Optional[str]]] = None  # base64-encoded per prompt (or None)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


MODEL_PATH = os.environ.get("REWARD_MODEL_PATH")
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_LEN = int(os.environ.get("MAX_MODEL_LEN", 3072))
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", 1))
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", 0.65))

if MODEL_PATH is None:
    raise ValueError("REWARD_MODEL_PATH env var is required.")

engine: Optional[LLM] = None
processor = None

print(f"[RewardServer] Configuration loaded for model: {MODEL_PATH}")


def decode_base64_to_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError("Invalid base64 image data")


@app.post("/compute_logprobs")
async def compute_logprobs(request: RewardRequest):
    total_batch_size = len(request.raw_prompt_ids)
    print(f"{total_batch_size} trajectories received, start computing logprobability")
    final_response_logprobs: List[List[float]] = []

    for start_idx in range(0, total_batch_size, INFERENCE_BATCH_SIZE):
        end_idx = min(start_idx + INFERENCE_BATCH_SIZE, total_batch_size)

        sub_raw_prompts = request.raw_prompt_ids[start_idx:end_idx]
        sub_responses = request.responses[start_idx:end_idx]
        sub_masks = request.response_masks[start_idx:end_idx]

        sub_mm_data = None
        if request.multi_modal_data:
            sub_mm_data = request.multi_modal_data[start_idx:end_idx]

        vllm_inputs = []
        response_slices = []

        current_batch_size = len(sub_raw_prompts)

        for i in range(current_batch_size):
            raw_prompt = sub_raw_prompts[i]
            response_seq = sub_responses[i]
            response_mask = sub_masks[i]

            valid_response_tokens = [
                token for token, mask in zip(response_seq, response_mask) if mask == 1
            ]

            full_token_ids = raw_prompt + valid_response_tokens

            r_len = len(valid_response_tokens)
            if r_len > 0:
                response_slices.append(slice(-r_len, None))
            else:
                response_slices.append(slice(0, 0))

            item_input = {"prompt_token_ids": full_token_ids}

            if sub_mm_data and i < len(sub_mm_data):
                mm_data = sub_mm_data[i]
                processed_mm: Dict[str, Any] = {}

                for k, v in mm_data.items():
                    if k == "image" and isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                        processed_mm[k] = [decode_base64_to_image(img) for img in v]
                    elif k == "image" and isinstance(v, str):
                        processed_mm[k] = decode_base64_to_image(v)
                    elif isinstance(v, list):
                        processed_mm[k] = torch.tensor(v)
                    else:
                        processed_mm[k] = v

                if processed_mm:
                    item_input["multi_modal_data"] = processed_mm

            vllm_inputs.append(item_input)

        sampling_params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            detokenize=False,
        )

        try:
            outputs = engine.generate(vllm_inputs, sampling_params, use_tqdm=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("CUDA OOM detected inside mini-batch! reducing batch size recommended.")
                torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail="CUDA OOM on Reward Server")
            raise e

        for i, output in enumerate(outputs):
            full_logprobs = output.prompt_logprobs
            target_slice = response_slices[i]
            response_logp_dicts = full_logprobs[target_slice]

            current_valid_tokens = [
                t for t, m in zip(sub_responses[i], sub_masks[i]) if m == 1
            ]

            current_seq_logprobs = []
            for token_id, logp_dict in zip(current_valid_tokens, response_logp_dicts):
                if logp_dict is None:
                    current_seq_logprobs.append(0.0)
                    continue

                if token_id in logp_dict:
                    current_seq_logprobs.append(logp_dict[token_id].logprob)
                else:
                    current_seq_logprobs.append(-100.0)

            final_logprobs = []
            valid_idx = 0
            original_mask = sub_masks[i]

            for mask_val in original_mask:
                if mask_val == 1:
                    if valid_idx < len(current_seq_logprobs):
                        final_logprobs.append(current_seq_logprobs[valid_idx])
                        valid_idx += 1
                    else:
                        final_logprobs.append(0.0)
                else:
                    final_logprobs.append(0.0)

            final_response_logprobs.append(final_logprobs)

        del vllm_inputs
        del outputs
        gc.collect()

    return {"logprobs": final_response_logprobs}


@app.post("/generate")
async def generate(request: GenerateRequest):
    total_batch_size = len(request.prompts)
    print(f"{total_batch_size} prompts received, start generating responses")
    all_responses: List[str] = []

    for start_idx in range(0, total_batch_size, INFERENCE_BATCH_SIZE):
        end_idx = min(start_idx + INFERENCE_BATCH_SIZE, total_batch_size)

        sub_prompts = request.prompts[start_idx:end_idx]
        sub_images = None
        if request.images:
            sub_images = request.images[start_idx:end_idx]

        vllm_inputs = []
        current_batch_size = len(sub_prompts)

        for i in range(current_batch_size):
            prompt_text = sub_prompts[i]
            has_image = sub_images is not None and i < len(sub_images) and sub_images[i] is not None

            if has_image:
                content = [
                    {"type": "image", "image": f"data:image/jpeg;base64,{sub_images[i]}"},
                    {"type": "text", "text": prompt_text},
                ]
            else:
                content = prompt_text

            messages = [{"role": "user", "content": content}]

            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            item_input = {"prompt": formatted_prompt}

            if has_image:
                pil_image = decode_base64_to_image(sub_images[i])
                item_input["multi_modal_data"] = {"image": [pil_image]}

            vllm_inputs.append(item_input)

        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=max(request.temperature, 1e-6),  # vLLM disallows temperature=0
            top_p=request.top_p,
        )

        try:
            outputs = engine.generate(vllm_inputs, sampling_params, use_tqdm=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("CUDA OOM detected in /generate mini-batch! reducing batch size recommended.")
                torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail="CUDA OOM on Reward Server (generate)")
            raise e

        for output in outputs:
            generated_text = output.outputs[0].text
            all_responses.append(generated_text)

        del vllm_inputs
        del outputs
        gc.collect()

    return {"responses": all_responses}


if __name__ == "__main__":

    engine = LLM(
        model=MODEL_PATH,
        dtype=DTYPE,
        trust_remote_code=True,
        tensor_parallel_size=TP_SIZE,
        distributed_executor_backend=None,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enable_sleep_mode=False,
        enforce_eager=True,
        max_model_len=MAX_LEN,
        gpu_memory_utilization=GPU_UTIL,
        max_num_batched_tokens=16384,
        max_num_seqs=6,
    )

    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"[RewardServer] Processor loaded from {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"AutoProcessor failed, falling back to AutoTokenizer: {e}")
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"[RewardServer] Tokenizer loaded from {MODEL_PATH}")

    port = int(os.environ.get("PORT", 8000))
    print(f"[RewardServer] Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
