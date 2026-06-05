#!/bin/bash
# ============================================================
# VL 安全 Benchmark 一键测评入口
#
# 用法:
#   bash script/run_all.sh \
#       --models "/path/to/model_a,/path/to/model_b" \
#       --benchmarks "all" \
#       --gpus 8 \
#       --tp 2 \
#       --output_dir ./output
#
# 环境变量（运行前需设置）:
#   JUDGE_API_KEY / JUDGE_API_KEY_QWEN  Judge 模型的 API Key
#   JUDGE_API_URL                       Judge 模型的 API 端点
#   GEN_API_KEY    (可选)               API 生成模式的 API Key（默认复用 JUDGE key）
#   GEN_API_URL    (可选)               API 生成模式的端点（默认复用 JUDGE_API_URL）
#   JUDGE_MODEL    (可选)               Judge 模型名称（默认 Qwen3-VL-235B-A22B-Instruct）
#   GEN_MODEL      (可选)               生成模型名称（默认 Qwen2.5-VL-7B-Instruct）
#
# 参数说明:
#   --models       逗号分隔的模型路径列表
#   --benchmarks   逗号分隔的 benchmark 列表，或 "all" 表示全部
#   --gpus         总 GPU 数量 (默认 8)
#   --tp           Tensor Parallel 大小 (默认 1)
#   --output_dir   输出根目录 (默认 ./output)
#   --judge_threads  Judge API 并发线程数 (默认 32)
#   --skip_gen     跳过生成步骤（仅做 judge + metrics）
#   --skip_judge   跳过 judge 步骤（仅做 metrics）
#   --baseline_beavertails  覆盖 beavertails 的 baseline gen_results 路径
#   --baseline_spavl        覆盖 spavl 的 baseline gen_results 路径
# ============================================================
# --- 获取项目根目录 ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/script"
CONFIG_FILE="$SCRIPT_DIR/config/benchmarks.yaml"

# --- 所有支持的 benchmarks ---
ALL_BENCHMARKS="mssbench,mssembodied,siuo,beavertails,spavl,vlguard,vlsbench"

# --- benchmark 对应的数据目录 ---
declare -A DATA_DIRS
DATA_DIRS=(
    ["mssbench"]="$PROJECT_ROOT/data/mssbench"
    ["mssembodied"]="$PROJECT_ROOT/data/mssembodied/kzhou35__mssbench"
    ["siuo"]="$PROJECT_ROOT/data/siuo"
    ["beavertails"]="$PROJECT_ROOT/data/beavertails"
    ["spavl"]="$PROJECT_ROOT/data/spavl"
    ["vlguard"]="$PROJECT_ROOT/data/vlguard"
    ["vlsbench"]="$PROJECT_ROOT/data/vlsbench"
)



# --- 默认参数 ---
MODELS=""
GPUS=8
TP=1
OUTPUT_DIR="$PROJECT_ROOT/output"
JUDGE_THREADS=32
SKIP_GEN=false
SKIP_JUDGE=false
# baseline 路径（可通过命令行覆盖，默认使用 yaml 中的配置）
BASELINE_BEAVERTAILS=""
BASELINE_SPAVL=""
# vLLM 额外参数
MAX_MODEL_LEN=4096
MAX_TOKENS=512
TEMPERATURE=0.0
BATCH_SIZE=64
GPU_MEMORY_UTILIZATION=0.9
LIMIT_MM_PER_PROMPT="image:1"
MIN_PIXELS="3136"
MAX_PIXELS="200704"
CONSTITUTION_MODE="system"

# ============================================================
# API 生成模式参数（使用 --use_api 启用）
# ============================================================
USE_API=false                   # 是否使用 API 生成（替代 vLLM）
API_MODEL="${API_MODEL:-}"                         # API 模型名称（如 Qwen2.5-VL-7B-Instruct）
API_KEY="${API_KEY:-}"                             # API Key
API_URL="${API_URL:-}"                             # API 端点
API_DELAY=0.5                      # API 调用间隔（秒）
API_SHARD_NUM=8                    # API 模式总分片数

# --- 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --judge_threads) JUDGE_THREADS="$2"; shift 2 ;;
        --skip_gen) SKIP_GEN=true; shift ;;
        --skip_judge) SKIP_JUDGE=true; shift ;;
        --baseline_beavertails) BASELINE_BEAVERTAILS="$2"; shift 2 ;;
        --baseline_spavl) BASELINE_SPAVL="$2"; shift 2 ;;
        # vLLM 额外参数
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --limit_mm_per_prompt) LIMIT_MM_PER_PROMPT="$2"; shift 2 ;;
        --min_pixels) MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels) MAX_PIXELS="$2"; shift 2 ;;
        # API 生成模式参数
        --use_api) USE_API=true; shift ;;
        --constitution_mode) CONSTITUTION_MODE="$2"; shift 2 ;;
        --api_model) API_MODEL="$2"; shift 2 ;;
        --api_key) API_KEY="$2"; shift 2 ;;
        --api_url) API_URL="$2"; shift 2 ;;
        --api_delay) API_DELAY="$2"; shift 2 ;;
        --api_shard_num) API_SHARD_NUM="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- 参数校验 ---
if [[ -z "$MODELS" ]]; then
    echo "Error: --models is required."
    echo "Usage: bash script/run_all.sh --models '/path/to/model_a,/path/to/model_b' [options]"
    exit 1
fi

# --- 展开 benchmarks ---
if [[ "$BENCHMARKS" == "all" ]]; then
    BENCHMARKS="$ALL_BENCHMARKS"
fi

# --- 设置 API 环境变量 ---
if [[ "$USE_API" == true ]]; then
    if [[ -n "$API_KEY" ]]; then
        export GEN_API_KEY="$API_KEY"
    fi
    if [[ -n "$API_MODEL" ]]; then
        export GEN_MODEL="$API_MODEL"
    fi
    if [[ -n "$API_URL" ]]; then
        export GEN_API_URL="$API_URL"
    fi
    # API 模式下不使用本地模型路径，生成时使用 API_MODEL
    model_name="api_${API_MODEL:-unknown}"
fi

# --- 打印配置 ---
echo "============================================================"
echo "  VL Safety Benchmark Evaluation Pipeline"
echo "============================================================"
echo "  Models:       $MODELS"
echo "  Benchmarks:   $BENCHMARKS"
echo "  GPUs:         $GPUS"
echo "  TP Size:      $TP"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Judge Threads: $JUDGE_THREADS"
echo "  Skip Gen:     $SKIP_GEN"
echo "  Skip Judge:   $SKIP_JUDGE"
if [[ "$USE_API" == true ]]; then
    echo "  ---- API Generation Mode ----"
    echo "  Use API:      YES"
    echo "  API Model:    $API_MODEL"
    echo "  API URL:      ${API_URL:-default}"
    echo "  API Delay:    ${API_DELAY}s"
    echo "  API Shards:   $API_SHARD_NUM"
else
    echo "  ---- vLLM Parameters ----"
    echo "  Max Model Len: $MAX_MODEL_LEN"
    echo "  Max Tokens:    $MAX_TOKENS"
    echo "  Temperature:   $TEMPERATURE"
    echo "  Batch Size:    $BATCH_SIZE"
    echo "  GPU Memory Util: $GPU_MEMORY_UTILIZATION"
    echo "  Limit MM/Prompt: $LIMIT_MM_PER_PROMPT"
    echo "  Min Pixels:    ${MIN_PIXELS:-auto}"
    echo "  Max Pixels:    ${MAX_PIXELS:-auto}"
    echo "  Constitution Mode: $CONSTITUTION_MODE"
fi
echo "============================================================"

# ============================================================
# 分支 1: API 生成模式（不需要 model_path）
# ============================================================
if [[ "$USE_API" == true ]]; then
    IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

    model_name="api_${API_MODEL}"
    TOTAL_TASKS=${#BENCH_LIST[@]}
    CURRENT_TASK=0

    for benchmark in "${BENCH_LIST[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        data_dir="${DATA_DIRS[$benchmark]:-}"

        if [[ -z "$data_dir" ]]; then
            echo "WARNING: Unknown benchmark '$benchmark', skipping."
            continue
        fi

        if [[ ! -d "$data_dir" ]]; then
            echo "WARNING: Data dir not found for '$benchmark': $data_dir, skipping."
            continue
        fi

        bench_output_dir="$OUTPUT_DIR/$model_name/$benchmark"
        mkdir -p "$bench_output_dir"

        echo ""
        echo "============================================================"
        echo "  [API Mode $CURRENT_TASK/$TOTAL_TASKS] $model_name x $benchmark"
        echo "============================================================"

        # --- Step 1: API 生成 ---
        if [[ "$SKIP_GEN" == false ]]; then
            echo "[Step 1/3] Generating responses via API..."
            if [[ -f "$bench_output_dir/gen_results.jsonl" ]]; then
                echo "  gen_results.jsonl already exists, will resume."
            fi

            # 使用 api_gen.py 进行分片生成
            num_shards=${API_SHARD_NUM:-1}
            echo "  Using API generation (model: $API_MODEL, shards: $num_shards)"

            for shard_id in $(seq 0 $((num_shards - 1))); do
                echo "  Processing shard $shard_id/$((num_shards - 1))..."
                python "$SCRIPT_DIR/gen/api_gen.py" \
                    --benchmark "$benchmark" \
                    --data_dir "$data_dir" \
                    --output_dir "$bench_output_dir" \
                    --model "$API_MODEL" \
                    --shard_id "$shard_id" \
                    --num_shards "$num_shards" \
                    --max_tokens "$MAX_TOKENS" \
                    --temperature "$TEMPERATURE" \
                    --delay "$API_DELAY" \
                    ${CONSTITUTION_MODE:+--constitution_mode "$CONSTITUTION_MODE"} \
                    --resume
            done

            # 合并所有分片结果
            echo "  Merging shard results..."
            python -c "
import os
import json

output_dir = '$bench_output_dir'
merged_file = os.path.join(output_dir, 'gen_results.jsonl')

results = []
for f in sorted(os.listdir(output_dir)):
    if f.startswith('gen_shard_') and f.endswith('.jsonl'):
        shard_file = os.path.join(output_dir, f)
        with open(shard_file, 'r', encoding='utf-8') as sf:
            for line in sf:
                if line.strip():
                    results.append(json.loads(line))

# 按 id 去重（保留第一个）
seen = set()
unique_results = []
for r in results:
    if r['id'] not in seen:
        seen.add(r['id'])
        unique_results.append(r)

# 写入合并文件
with open(merged_file, 'w', encoding='utf-8') as f:
    for r in unique_results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Merged {len(unique_results)} results to {merged_file}')
"
        else
            echo "[Step 1/3] Skipping generation (--skip_gen)"
        fi

        # --- Step 2: Judge ---
        if [[ "$SKIP_JUDGE" == false ]]; then
            echo "[Step 2/3] Running judge..."
            if [[ ! -f "$bench_output_dir/gen_results.jsonl" ]]; then
                echo "  WARNING: gen_results.jsonl not found, skipping judge."
                continue
            fi

            python "$SCRIPT_DIR/judge/judge_dispatch.py" \
                --benchmark "$benchmark" \
                --input "$bench_output_dir/gen_results.jsonl" \
                --output "$bench_output_dir/judge_results.jsonl" \
                --config "$CONFIG_FILE" \
                --threads "$JUDGE_THREADS" \
                --resume
        else
            echo "[Step 2/3] Skipping judge (--skip_judge)"
        fi

        # --- Step 3: Metrics ---
        echo "[Step 3/3] Computing metrics..."
        JUDGE_FILE="$bench_output_dir/judge_results.jsonl"
        if [[ ! -f "$JUDGE_FILE" ]]; then
            echo "  WARNING: judge_results.jsonl not found, skipping metrics."
            continue
        fi

        python "$SCRIPT_DIR/metrics/compute_metrics.py" \
            --benchmark "$benchmark" \
            --input "$JUDGE_FILE" \
            --output "$bench_output_dir/metrics.json"

        echo "  Done: $bench_output_dir/metrics.json"
    done

# ============================================================
# 分支 2: 本地 vLLM 生成模式
# ============================================================
else
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

    TOTAL_TASKS=$(( ${#MODEL_LIST[@]} * ${#BENCH_LIST[@]} ))
    CURRENT_TASK=0

    for model_path in "${MODEL_LIST[@]}"; do
        model_name=$(basename "$model_path")

        for benchmark in "${BENCH_LIST[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))
            data_dir="${DATA_DIRS[$benchmark]:-}"

            if [[ -z "$data_dir" ]]; then
                echo "WARNING: Unknown benchmark '$benchmark', skipping."
                continue
            fi

            if [[ ! -d "$data_dir" ]]; then
                echo "WARNING: Data dir not found for '$benchmark': $data_dir, skipping."
                continue
            fi

            bench_output_dir="$OUTPUT_DIR/$model_name/$benchmark"
            mkdir -p "$bench_output_dir"

            echo ""
            echo "============================================================"
            echo "  [vLLM $CURRENT_TASK/$TOTAL_TASKS] $model_name x $benchmark"
            echo "============================================================"

            # --- Step 1: 生成 ---
            if [[ "$SKIP_GEN" == false ]]; then
                echo "[Step 1/3] Generating responses..."
                if [[ -f "$bench_output_dir/gen_results.jsonl" ]]; then
                    echo "  gen_results.jsonl already exists, will resume."
                fi

                bash "$SCRIPT_DIR/gen/dp_launcher.sh" \
                    --model_path "$model_path" \
                    --benchmark "$benchmark" \
                    --data_dir "$data_dir" \
                    --output_dir "$bench_output_dir" \
                    --total_gpus "$GPUS" \
                    --tp_size "$TP" \
                    --max_model_len "$MAX_MODEL_LEN" \
                    --max_tokens "$MAX_TOKENS" \
                    --temperature "$TEMPERATURE" \
                    --batch_size "$BATCH_SIZE" \
                    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
                    --limit_mm_per_prompt "$LIMIT_MM_PER_PROMPT" \
                    ${MIN_PIXELS:+--min_pixels "$MIN_PIXELS"} \
                    ${MAX_PIXELS:+--max_pixels "$MAX_PIXELS"} \
                    ${CONSTITUTION_MODE:+--constitution_mode "$CONSTITUTION_MODE"}
            else
                echo "[Step 1/3] Skipping generation (--skip_gen)"
            fi

            # --- Step 2: Judge ---
            if [[ "$SKIP_JUDGE" == false ]]; then
                echo "[Step 2/3] Running judge..."
                if [[ ! -f "$bench_output_dir/gen_results.jsonl" ]]; then
                    echo "  WARNING: gen_results.jsonl not found, skipping judge."
                    continue
                fi

                # 构建额外参数（胜和率任务需要 baseline）
                EXTRA_JUDGE_ARGS=""
                if [[ "$benchmark" == "beavertails" && -n "$BASELINE_BEAVERTAILS" ]]; then
                    EXTRA_JUDGE_ARGS="--baseline_gen_results $BASELINE_BEAVERTAILS"
                elif [[ "$benchmark" == "spavl" && -n "$BASELINE_SPAVL" ]]; then
                    EXTRA_JUDGE_ARGS="--baseline_gen_results $BASELINE_SPAVL"
                fi

                python "$SCRIPT_DIR/judge/judge_dispatch.py" \
                    --benchmark "$benchmark" \
                    --input "$bench_output_dir/gen_results.jsonl" \
                    --output "$bench_output_dir/judge_results.jsonl" \
                    --config "$CONFIG_FILE" \
                    --threads "$JUDGE_THREADS" \
                    --resume \
                    $EXTRA_JUDGE_ARGS
            else
                echo "[Step 2/3] Skipping judge (--skip_judge)"
            fi

            # --- Step 3: Metrics ---
            echo "[Step 3/3] Computing metrics..."
            JUDGE_FILE="$bench_output_dir/judge_results.jsonl"
            if [[ ! -f "$JUDGE_FILE" ]]; then
                echo "  WARNING: judge_results.jsonl not found, skipping metrics."
                continue
            fi

            python "$SCRIPT_DIR/metrics/compute_metrics.py" \
                --benchmark "$benchmark" \
                --input "$JUDGE_FILE" \
                --output "$bench_output_dir/metrics.json"

            echo "  Done: $bench_output_dir/metrics.json"
        done
    done
fi

# --- Step 4: 汇总所有结果 ---
echo ""
echo "============================================================"
echo "  Generating Summary..."
echo "============================================================"
python "$SCRIPT_DIR/metrics/compute_metrics.py" \
    --summarize \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  All evaluations complete!"
echo "  Results: $OUTPUT_DIR/"
echo "  Summary: $OUTPUT_DIR/summary.csv"
echo "============================================================"
