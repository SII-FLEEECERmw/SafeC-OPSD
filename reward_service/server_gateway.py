"""FastAPI gateway for the OPSDGate teacher reward service.

Spawns ``NUM_GPUS // TP_SIZE`` vLLM workers and load-balances incoming
requests round-robin. All configuration is passed via environment variables
(see ``.env.example`` and ``docs/reward_service.md``).
"""

import asyncio
import os
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# --- Config (env-driven, all optional except TEACHER_MODEL_PATH) ---
MODEL_PATH = os.environ.get("TEACHER_MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("TEACHER_MODEL_PATH env var is required for the gateway.")

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
START_PORT = int(os.environ.get("START_PORT", "8001"))
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))
GPU_UTIL = os.environ.get("GPU_UTIL", "0.60")
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
INFERENCE_BATCH_SIZE = os.environ.get("INFERENCE_BATCH_SIZE", "6")

DEFAULT_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_onpolicy_reward.py")
WORKER_PATH = os.environ.get("REWARD_WORKER_PATH", DEFAULT_WORKER_PATH)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = os.path.basename(os.path.normpath(MODEL_PATH))
LOGDIR_DEFAULT = os.path.join(os.getcwd(), "reward_worker_logs")
WORKER_LOGDIR_PATH = os.path.join(
    os.environ.get("REWARD_WORKER_LOGDIR", LOGDIR_DEFAULT),
    f"{model_name}_{timestamp}",
)
os.makedirs(WORKER_LOGDIR_PATH, exist_ok=True)

assert NUM_GPUS % TP_SIZE == 0, "NUM_GPUS must be divisible by TP_SIZE"
NUM_WORKERS = NUM_GPUS // TP_SIZE

worker_processes = []
worker_log_files = []
worker_urls = [f"http://127.0.0.1:{START_PORT + i}" for i in range(NUM_WORKERS)]
request_counter = 0
client = httpx.AsyncClient(timeout=1200.0)  # 20 min, enough for batched rewards

# ---------------- Metrics ----------------
METRICS_WINDOW_SEC = 60
metrics_lock = asyncio.Lock()
events: deque = deque()  # entries of (ts, ok(1/0), latency_sec, tokens)

total_ok = 0
total_fail = 0


def _print_config():
    print("[Gateway] Effective config:")
    for k, v in [
        ("TEACHER_MODEL_PATH", MODEL_PATH),
        ("NUM_GPUS", NUM_GPUS),
        ("TP_SIZE", TP_SIZE),
        ("NUM_WORKERS", NUM_WORKERS),
        ("START_PORT", START_PORT),
        ("GATEWAY_PORT", GATEWAY_PORT),
        ("GPU_UTIL", GPU_UTIL),
        ("DTYPE", DTYPE),
        ("MAX_MODEL_LEN", MAX_LEN),
        ("INFERENCE_BATCH_SIZE", INFERENCE_BATCH_SIZE),
        ("REWARD_WORKER_PATH", WORKER_PATH),
        ("REWARD_WORKER_LOGDIR", WORKER_LOGDIR_PATH),
    ]:
        print(f"  {k}={v}")


def _prune_events(now: float):
    cutoff = now - METRICS_WINDOW_SEC
    while events and events[0][0] < cutoff:
        events.popleft()


def _count_tokens_from_result(result_json: dict) -> int:
    """Best-effort token count for /compute_logprobs (length of logprobs lists)
    or /generate (rough char/4 estimate)."""
    try:
        lp = result_json.get("logprobs", None)
        if lp is not None:
            return sum(len(x) for x in lp if isinstance(x, list))

        responses = result_json.get("responses", None)
        if responses is not None:
            return sum(max(len(r) // 4, 1) for r in responses if isinstance(r, str))

        return 0
    except Exception:
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_config()
    print(f"[Gateway] Launching {NUM_WORKERS} workers, each with TP_SIZE={TP_SIZE} GPUs...")

    for w in range(NUM_WORKERS):
        port = START_PORT + w
        gpu_ids = list(range(w * TP_SIZE, (w + 1) * TP_SIZE))
        cuda_visible = ",".join(map(str, gpu_ids))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["PORT"] = str(port)
        env["REWARD_MODEL_PATH"] = MODEL_PATH  # worker still reads this name
        env["GPU_UTIL"] = str(GPU_UTIL)
        env["DTYPE"] = str(DTYPE)
        env["MAX_MODEL_LEN"] = str(MAX_LEN)
        env["INFERENCE_BATCH_SIZE"] = str(INFERENCE_BATCH_SIZE)
        env["TP_SIZE"] = str(TP_SIZE)

        log_filename = os.path.join(
            WORKER_LOGDIR_PATH,
            f"worker_{w}_gpus_{cuda_visible.replace(',', '_')}.log",
        )
        log_file = open(log_filename, "w", encoding="utf-8")
        worker_log_files.append(log_file)

        p = subprocess.Popen(
            ["python", WORKER_PATH],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        worker_processes.append(p)

        print(f"[Gateway] Worker {w} started on GPUs [{cuda_visible}], Port {port}, log={log_filename}")

    print("[Gateway] Waiting 30s for workers to load models...")
    time.sleep(30)

    yield

    print("[Gateway] Shutting down workers...")
    for p in worker_processes:
        p.terminate()

    print("[Gateway] Closing log files...")
    for f in worker_log_files:
        f.close()

    await client.aclose()


app = FastAPI(title="OPSDGate Reward Gateway", lifespan=lifespan)


async def _proxy(target_path: str, request: Request):
    global request_counter, total_ok, total_fail

    worker_index = request_counter % NUM_WORKERS
    request_counter += 1
    target_url = f"{worker_urls[worker_index]}{target_path}"

    body = await request.json()

    t0 = time.perf_counter()
    now = time.time()

    try:
        response = await client.post(target_url, json=body)
        latency = time.perf_counter() - t0

        if response.status_code != 200:
            async with metrics_lock:
                total_fail += 1
                events.append((now, 0, latency, 0))
                _prune_events(now)
            return JSONResponse(status_code=response.status_code, content=response.json())

        result = response.json()
        tokens = _count_tokens_from_result(result)

        async with metrics_lock:
            total_ok += 1
            events.append((now, 1, latency, tokens))
            _prune_events(now)

        return result

    except httpx.RequestError as exc:
        latency = time.perf_counter() - t0
        async with metrics_lock:
            total_fail += 1
            events.append((now, 0, latency, 0))
            _prune_events(now)
        raise HTTPException(status_code=502, detail=f"Worker {worker_index} unavailable: {exc}")


@app.post("/compute_logprobs")
async def proxy_compute_logprobs(request: Request):
    return await _proxy("/compute_logprobs", request)


@app.post("/generate")
async def proxy_generate(request: Request):
    return await _proxy("/generate", request)


@app.get("/metrics")
async def metrics():
    now = time.time()
    async with metrics_lock:
        _prune_events(now)
        window = list(events)

        ok = sum(e[1] for e in window)
        fail = len(window) - ok
        latencies = [e[2] for e in window if e[1] == 1]
        tokens = sum(e[3] for e in window if e[1] == 1)

        window_sec = METRICS_WINDOW_SEC
        rps = (len(window) / window_sec) if window_sec > 0 else 0.0
        ok_rps = (ok / window_sec) if window_sec > 0 else 0.0
        tps = (tokens / window_sec) if window_sec > 0 else 0.0
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        return {
            "window_sec": window_sec,
            "rps": rps,
            "ok_rps": ok_rps,
            "tokens_per_sec": tps,
            "avg_latency_sec": avg_latency,
            "in_window": {"requests": len(window), "ok": ok, "fail": fail},
            "total": {"ok": total_ok, "fail": total_fail},
            "workers": NUM_WORKERS,
            "tp_size": TP_SIZE,
        }


if __name__ == "__main__":
    import uvicorn

    print(f"[Gateway] Serving on port {GATEWAY_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
