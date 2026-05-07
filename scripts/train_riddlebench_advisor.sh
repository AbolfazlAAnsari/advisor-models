#!/bin/bash

# Training script for RiddleBench advisor domain.
#
# Counterfactual advantage estimation: each training question appears as two rows
# (null_advice=False and null_advice=True) with advantage_batch_normalize=true,
# so the null row acts as a per-question baseline for the GRPO advantage.
#
# Reward: R_total = alpha*R_outcome + gate*(beta*R_diag + gamma*R_adh)
# gate = 1[R_outcome > 0]
#
# Usage:
#   export OPENAI_API_KEY=...
#   export API_BASE=...          # Azure / vLLM endpoint base URL
#   export STUDENT_MODEL=gpt-4o-mini
#   export JUDGE_MODEL=gpt-4o-mini
#   bash scripts/train_riddlebench_advisor.sh

set -e

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="/workspace/advisor-models/SkyRL/skyrl-train:$PYTHONPATH"

export ADVISOR_MODELS_MODE="${ADVISOR_MODELS_MODE:-advisor}"
export STUDENT_MODEL="${STUDENT_MODEL:-gpt-4o-mini}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

RB_DATA_DIR="${RB_DATA_DIR:-/workspace/advisor-models/data/riddlebench}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-console}"

/workspace/advisor-models/SkyRL/skyrl-train/.venv/bin/python \
  -m advisor_models.riddlebench.main_riddlebench \
  data.train_data="['$RB_DATA_DIR/train.parquet']" \
  data.val_data="['$RB_DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.advantage_batch_normalize=true \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=10 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  environment.env_class=riddlebench \
  ++environment.skyrl_gym.riddlebench.outcome_weight=1.0 \
  ++environment.skyrl_gym.riddlebench.diag_weight=0.1 \
  ++environment.skyrl_gym.riddlebench.adh_weight=0.1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="advisor_models" \
  trainer.run_name="riddlebench_advisor" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/riddlebench"
