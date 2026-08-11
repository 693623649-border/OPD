#!/usr/bin/env bash
# Formal Table 1 launcher for training the Qwen3-4B-Base-GRPO teacher.
#
# PRESET=paper preserves the disclosed scientific hyperparameters.  The paper
# used 8xA800-80G; this launcher deliberately uses 2xA100-80G and records that
# hardware change as an adaptation instead of presenting it as exact hardware
# reproduction.  smoke/calibration are engineering checks only.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRESET="${PRESET:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
N_GPUS="${N_GPUS:-2}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/artifacts/upstream/manual-grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/grpo}"
METRICS_FILE="${METRICS_FILE:-${RUN_DIR}/metrics.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/datasets/dapo-math-17k-processed.parquet}"

# Immutable paper identities.  A different model/revision is a new experiment,
# not this Table 1 reproduction entry point.
MODEL_ID="Qwen/Qwen3-4B-Base"
MODEL_REVISION="906bfd4b4dc7f14ee4320094d8b41684abff8539"
EXPECTED_DATA_ROWS="${EXPECTED_DATA_ROWS:-17917}"
EXPECTED_DATA_SHA256="${EXPECTED_DATA_SHA256:-500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39}"

LEARNING_RATE="1e-6"
MAX_PROMPT_LENGTH="1024"
VALIDATION_MAX_RESPONSE_LENGTH="31744"
TEMPERATURE="1.0"
TOP_P="1.0"
REPETITION_PENALTY="1.0"
KL_COEFFICIENT="0.0"
LOSS_AGGREGATION="token-mean"

case "${PRESET}" in
  smoke)
    FIDELITY_LABEL="engineering-smoke; 2xA100 adaptation; not a paper result"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
    # GRPO needs at least two samples in a prompt group; n=1 would produce a
    # zero group-relative advantage and is not a meaningful integration test.
    N_RESPONSES="${N_RESPONSES:-2}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
    SAVE_FREQ="${SAVE_FREQ:-2}"
    ;;
  calibration)
    FIDELITY_LABEL="engineering-calibration; 2xA100 adaptation; not a paper result"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
    N_RESPONSES="${N_RESPONSES:-4}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-40}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
    SAVE_FREQ="${SAVE_FREQ:-10}"
    ;;
  paper)
    FIDELITY_LABEL="Table 1 scientific parameters; 2xA100 hardware adaptation (paper: 8xA800-80G)"
    TRAIN_BATCH_SIZE="64"
    MINI_BATCH_SIZE="64"
    N_RESPONSES="8"
    MAX_RESPONSE_LENGTH="7168"
    TRAIN_MAX_SAMPLES="-1"
    TOTAL_TRAINING_STEPS=""
    SAVE_FREQ="${SAVE_FREQ:-20}"
    ;;
  *)
    echo "PRESET must be one of: smoke, calibration, paper" >&2
    exit 2
    ;;
esac

if [[ "${N_GPUS}" != "2" ]]; then
  echo "This local formal launcher is scoped to exactly 2 GPUs; got N_GPUS=${N_GPUS}." >&2
  echo "Create a separately labelled hardware protocol for any other topology." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if (($#)); then
  echo "Hydra CLI overrides are disabled for the formal Table 1 launcher; use a separately versioned protocol." >&2
  exit 2
fi

MAX_TRAIN_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_VALIDATION_MODEL_LEN=$((MAX_PROMPT_LENGTH + VALIDATION_MAX_RESPONSE_LENGTH))
if [[ "${PRESET}" == "paper" ]]; then
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_VALIDATION_MODEL_LEN}}"
else
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_TRAIN_MODEL_LEN}}"
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_TRAIN_MODEL_LEN}}"

val_files="['${REPO_ROOT}/datasets/test_data/AIME24/test.parquet','${REPO_ROOT}/datasets/test_data/AIME25/test.parquet','${REPO_ROOT}/datasets/test_data/AMC23/test.parquet']"
hydra_args=(
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=true"
  "algorithm.grpo_outcome_weight=1.0"
  "algorithm.use_kl_in_reward=false"
  "algorithm.kl_ctrl.kl_coef=${KL_COEFFICIENT}"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${val_files}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=true"
  "data.truncation=error"
  "data.return_raw_chat=true"
  "data.shuffle=false"
  "data.seed=${SEED}"
  "+data.apply_chat_template_kwargs.enable_thinking=false"
  "actor_rollout_ref.model.path=${MODEL_ID}"
  "actor_rollout_ref.model.use_remove_padding=false"
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"
  "actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.actor.use_kl_loss=false"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEFFICIENT}"
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGGREGATION}"
  "actor_rollout_ref.actor.fsdp_config.param_offload=false"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "actor_rollout_ref.actor.fsdp_config.forward_prefetch=true"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.dtype=bfloat16"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.seed=${SEED}"
  "actor_rollout_ref.rollout.top_p=${TOP_P}"
  "actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKENS_PER_GPU}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.n=${N_RESPONSES}"
  "actor_rollout_ref.rollout.calculate_log_probs=true"
  "+actor_rollout_ref.rollout.log_prob_top_k=0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=true"
  "+actor_rollout_ref.rollout.val_kwargs.max_tokens=${VALIDATION_MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.val_kwargs.n=16"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.7"
  "actor_rollout_ref.rollout.val_kwargs.top_p=0.95"
  "reward_model.enable=false"
  "custom_reward_function.path=${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_math/__init__.py"
  "custom_reward_function.name=reward_func"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.logger=['console','file']"
  "trainer.project_name=Rethinking-OPD-Table1-Upstream"
  "trainer.experiment_name=grpo-teacher-${PRESET}-seed${SEED}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=2"
  "trainer.total_epochs=1"
  "trainer.resume_mode=disable"
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints"
  "trainer.is_plot=false"
)
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  hydra_args+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

command=("${PYTHON_BIN}" -m verl.trainer.main_ppo "${hydra_args[@]}")

printf '%s\n' \
  "GRPO teacher configuration (paper Table 1)" \
  "  fidelity           : ${FIDELITY_LABEL}" \
  "  model              : ${MODEL_ID}@${MODEL_REVISION}" \
  "  processed data     : ${TRAIN_DATA} (rows ${EXPECTED_DATA_ROWS}, sha256 ${EXPECTED_DATA_SHA256})" \
  "  data fidelity      : author-released processed file; literal prompt suffix uses \\boxed{{}}" \
  "  global/mini batch  : ${TRAIN_BATCH_SIZE} / ${MINI_BATCH_SIZE}" \
  "  rollout n          : ${N_RESPONSES}" \
  "  prompt/response    : ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}" \
  "  validation max     : ${VALIDATION_MAX_RESPONSE_LENGTH}" \
  "  lr/temp/top-p      : ${LEARNING_RATE} / ${TEMPERATURE} / ${TOP_P}" \
  "  loss/KL            : ${LOSS_AGGREGATION} / ${KL_COEFFICIENT}" \
  "  original/local HW  : 8xA800-80G / 2xA100-80G"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'MODEL_REVISION=%q TABLE1_FIDELITY=%q VERL_FILE_LOGGER_PATH=%q ' \
    "${MODEL_REVISION}" "${FIDELITY_LABEL}" "${METRICS_FILE}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Author-released processed DAPO dataset is missing: ${TRAIN_DATA}" >&2
  exit 1
fi
actual_data_sha256="$(sha256sum "${TRAIN_DATA}" | awk '{print $1}')"
if [[ "${actual_data_sha256}" != "${EXPECTED_DATA_SHA256}" ]]; then
  echo "Processed DAPO sha256 mismatch: ${actual_data_sha256}" >&2
  echo "Expected the author-released file: ${EXPECTED_DATA_SHA256}" >&2
  exit 1
fi
actual_data_rows="$("${PYTHON_BIN}" - "${TRAIN_DATA}" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"
if [[ "${actual_data_rows}" != "${EXPECTED_DATA_ROWS}" ]]; then
  echo "Processed DAPO row-count mismatch: ${actual_data_rows}; expected ${EXPECTED_DATA_ROWS}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse GRPO OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/verl/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored VERL entry point is missing." >&2
  exit 1
fi

# Resolve the immutable Hub revision before VERL sees the model path.  This
# prevents a mutable branch from silently changing a nominally identical run.
runtime_model="$("${PYTHON_BIN}" - "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
for index in "${!hydra_args[@]}"; do
  case "${hydra_args[index]}" in
    actor_rollout_ref.model.path=*) hydra_args[index]="actor_rollout_ref.model.path=${runtime_model}" ;;
  esac
done
command=("${PYTHON_BIN}" -m verl.trainer.main_ppo "${hydra_args[@]}")

mkdir -p "${OUTPUT_DIR}" "$(dirname -- "${METRICS_FILE}")"
printf '{"model_id":"%s","requested_revision":"%s","snapshot_path":"%s"}\n' \
  "${MODEL_ID}" "${MODEL_REVISION}" "${runtime_model}" >"${OUTPUT_DIR}/model_snapshot.json"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'cd %q\n' "${REPO_ROOT}"
  printf 'export VERL_FILE_LOGGER_PATH=%q\n' "${METRICS_FILE}"
  printf '%q ' "${command[@]}"
  printf '\n'
} >"${OUTPUT_DIR}/command.sh"

export VERL_FILE_LOGGER_PATH="${METRICS_FILE}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
cd "${REPO_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
touch "${OUTPUT_DIR}/_SUCCESS"
