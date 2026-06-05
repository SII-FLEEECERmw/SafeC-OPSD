#!/bin/bash
# ============================================================
# DP 启动器：自动计算 tp/dp，启动多个 vllm 进程并行推理
#
# 用法:
#   bash script/gen/dp_launcher.sh \
#       --model_path /path/to/model \
#       --benchmark mssbench \
#       --data_dir data/mssbench \
#       --output_dir output/model_name/mssbench \
#       --total_gpus 8 \
#       --tp_size 2
# ============================================================

set -euo pipefail

# --- 默认参数 ---
MODEL_PATH=""
BENCHMARK=""
DATA_DIR=""
OUTPUT_DIR=""
TOTAL_GPUS=8
TP_SIZE=1
MAX_MODEL_LEN=4096
MAX_TOKENS=512
TEMPERATURE=0.0
BATCH_SIZE=64
GPU_MEMORY_UTILIZATION=0.9
LIMIT_MM_PER_PROMPT="image:1"
MIN_PIXELS="3136"
MAX_PIXELS="200704"
TRUST_REMOTE_CODE=true
CONSTITUTION_MODE="none"

# --- 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --total_gpus) TOTAL_GPUS="$2"; shift 2 ;;
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --limit_mm_per_prompt) LIMIT_MM_PER_PROMPT="$2"; shift 2 ;;
        --min_pixels) MIN_PIXELS="$2"; shift 2 ;;
        --max_pixels) MAX_PIXELS="$2"; shift 2 ;;
        --trust_remote_code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
        --constitution_mode) CONSTITUTION_MODE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- 参数校验 ---
if [[ -z "$MODEL_PATH" || -z "$BENCHMARK" || -z "$DATA_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --model_path, --benchmark, --data_dir, --output_dir are required."
    exit 1
fi

# --- 计算 DP 大小 ---
DP_SIZE=$((TOTAL_GPUS / TP_SIZE))
echo "============================================================"
echo "DP Launcher Configuration:"
echo "  Model:       $MODEL_PATH"
echo "  Benchmark:   $BENCHMARK"
echo "  Data Dir:    $DATA_DIR"
echo "  Output Dir:  $OUTPUT_DIR"
echo "  Total GPUs:  $TOTAL_GPUS"
echo "  TP Size:     $TP_SIZE"
echo "  DP Size:     $DP_SIZE"
echo "  Constitution Mode: $CONSTITUTION_MODE"
echo "============================================================"

# --- 获取脚本所在目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/vllm_gen.py"

# --- 获取数据总条数 ---
TOTAL_SAMPLES=$(python "$GEN_SCRIPT" --count_only --benchmark "$BENCHMARK" --data_dir "$DATA_DIR")
echo "Total samples: $TOTAL_SAMPLES"

if [[ "$TOTAL_SAMPLES" -eq 0 ]]; then
    echo "No samples found, exiting."
    exit 0
fi

# --- 创建输出目录 ---
mkdir -p "$OUTPUT_DIR"

# --- 启动 DP 个后台进程 ---
PIDS=()
for ((shard_id=0; shard_id<DP_SIZE; shard_id++)); do
    # 计算 CUDA_VISIBLE_DEVICES
    START_GPU=$((shard_id * TP_SIZE))
    GPU_LIST=""
    for ((g=0; g<TP_SIZE; g++)); do
        if [[ -n "$GPU_LIST" ]]; then
            GPU_LIST="${GPU_LIST},"
        fi
        GPU_LIST="${GPU_LIST}$((START_GPU + g))"
    done

    echo "Starting shard $shard_id on GPUs: $GPU_LIST"

    CUDA_VISIBLE_DEVICES="$GPU_LIST" python "$GEN_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --benchmark "$BENCHMARK" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --shard_id "$shard_id" \
        --num_shards "$DP_SIZE" \
        --tp_size "$TP_SIZE" \
        --max_model_len "$MAX_MODEL_LEN" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --batch_size "$BATCH_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --limit_mm_per_prompt "$LIMIT_MM_PER_PROMPT" \
        ${MIN_PIXELS:+--min_pixels "$MIN_PIXELS"} \
        ${MAX_PIXELS:+--max_pixels "$MAX_PIXELS"} \
        ${TRUST_REMOTE_CODE:+--trust_remote_code} \
        ${CONSTITUTION_MODE:+--constitution_mode "$CONSTITUTION_MODE"} \
        --resume \
        > "$OUTPUT_DIR/shard_${shard_id}.log" 2>&1 &

    PIDS+=($!)
done

echo "All $DP_SIZE shards launched. Waiting for completion..."

# --- 等待所有进程完成 ---
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: Process $pid failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED shard(s) failed. Check logs in $OUTPUT_DIR/"
fi

# --- 合并所有分片输出 ---
echo "Merging shard outputs..."
MERGED_OUTPUT="$OUTPUT_DIR/gen_results.jsonl"

# 清空或创建合并文件
> "$MERGED_OUTPUT"

for ((shard_id=0; shard_id<DP_SIZE; shard_id++)); do
    SHARD_FILE="$OUTPUT_DIR/gen_shard_${shard_id}.jsonl"
    if [[ -f "$SHARD_FILE" ]]; then
        cat "$SHARD_FILE" >> "$MERGED_OUTPUT"
    else
        echo "WARNING: Shard file not found: $SHARD_FILE"
    fi
done

# 统计合并后的总条数
MERGED_COUNT=$(wc -l < "$MERGED_OUTPUT")
echo "============================================================"
echo "Generation complete!"
echo "  Total generated: $MERGED_COUNT / $TOTAL_SAMPLES"
echo "  Output: $MERGED_OUTPUT"
echo "============================================================"
