#!/usr/bin/env bash
# Launch verl GRPO-OPSDGate training.
#
# Required env:
#   MODEL_PATH    Local path to the student model
#   TRAIN_FILES   Path (or comma-separated list) of training parquet files
#   VAL_FILES     Path (or comma-separated list) of validation parquet files
#   CKPT_DIR      Local checkpoint output directory
#   SERVER_URL    Teacher reward gateway URL (e.g. http://127.0.0.1:8000)
#
# Optional env (defaults shown):
#   PROJECT_NAME=copsd
#   EXP_NAME=grpo_opsdgate
#   NNODES=1
#   N_GPUS_PER_NODE=8
#   ROLLOUT_N=16
#   TOTAL_EPOCHS=3
#   TRAIN_BATCH_SIZE=128
#   PPO_MINI_BATCH_SIZE=32
#   PPO_MICRO_BATCH_SIZE_PER_GPU=8
#   ROLLOUT_TP_SIZE=2
#   MAX_PROMPT_LENGTH=2048
#   MAX_RESPONSE_LENGTH=2048
#   OPSDGATE_OUTCOME_SOURCES=mmpr   (comma-separated)
set -euo pipefail

: "${MODEL_PATH:?need MODEL_PATH (student model)}"
: "${TRAIN_FILES:?need TRAIN_FILES}"
: "${VAL_FILES:?need VAL_FILES}"
: "${CKPT_DIR:?need CKPT_DIR}"
: "${SERVER_URL:?need SERVER_URL (teacher reward gateway)}"

PROJECT_NAME="${PROJECT_NAME:-copsd}"
EXP_NAME="${EXP_NAME:-grpo_opsdgate}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ROLLOUT_N="${ROLLOUT_N:-16}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export OPSDGATE_OUTCOME_SOURCES="${OPSDGATE_OUTCOME_SOURCES:-mmpr}"
export SERVER_URL

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_DIR="$REPO_ROOT/verl"
REWARD_FN_PATH="$REPO_ROOT/reward_service/rewardfn_onpolicy_reward.py"

# Treat verl/ as a top-level package without `pip install`.
export PYTHONPATH="$VERL_DIR/..:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

exec python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_opsdgate \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.do_sample=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=opsdgate \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=compute_onpolicy_KL_reward \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=25 \
    trainer.test_freq=0 \
    trainer.total_epochs="$TOTAL_EPOCHS"
