#!/usr/bin/env bash
# Two-A100 launcher for the repository's vendored VERL 0.7 OPD implementation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRESET="${PRESET:-smoke}"
MODEL_PAIR="${MODEL_PAIR:-paper}"
N_GPUS="${N_GPUS:-2}"
SEED="${SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts/runs}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_ROOT}/datasets/dapo-math-17k.parquet}"
TOP_K="${TOP_K:-16}"
TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
SUPPORT_WEIGHT_NORMALIZATION="${SUPPORT_WEIGHT_NORMALIZATION:-author}"
REWARD_MANAGER="${REWARD_MANAGER:-batch}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
DATA_SHUFFLE="${DATA_SHUFFLE:-false}"
PIN_MODEL_SNAPSHOTS="${PIN_MODEL_SNAPSHOTS:-true}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-auto}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
POSITION_ENTROPY_LOG_FREQ="${POSITION_ENTROPY_LOG_FREQ:-0}"
POSITION_ENTROPY_START_STEP="${POSITION_ENTROPY_START_STEP:-0}"
POSITION_ENTROPY_BIN_SIZE="${POSITION_ENTROPY_BIN_SIZE:-256}"
MAX_ACTOR_CKPTS_TO_KEEP="${MAX_ACTOR_CKPTS_TO_KEEP:-}"
CHECKPOINT_SAVE_MODE="${CHECKPOINT_SAVE_MODE:-}"
RESUME_MODE="${RESUME_MODE:-disable}"
MILESTONE_STEPS="${MILESTONE_STEPS:-}"
GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-0}"
TRAINING_MATRIX_ROOT="${TRAINING_MATRIX_ROOT:-}"

case "${MODEL_PAIR}" in
  paper)
    DEFAULT_STUDENT="Qwen/Qwen3-1.7B-Base"
    DEFAULT_TEACHER="lllyx/Qwen3-4B-Base-GRPO"
    DEFAULT_STUDENT_REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    DEFAULT_TEACHER_REVISION="1f3b2966edfb75f2f98a00617588c1f748088422"
    DEFAULT_MIN_FREE_GIB="70"
    ;;
  4b)
    DEFAULT_STUDENT="Qwen/Qwen3-4B-Base"
    DEFAULT_TEACHER="lllyx/Qwen3-4B-Base-GRPO"
    DEFAULT_STUDENT_REVISION="906bfd4b4dc7f14ee4320094d8b41684abff8539"
    DEFAULT_TEACHER_REVISION="1f3b2966edfb75f2f98a00617588c1f748088422"
    DEFAULT_MIN_FREE_GIB="72"
    ;;
  mismatch)
    DEFAULT_STUDENT="Qwen/Qwen3-1.7B-Base"
    DEFAULT_TEACHER="Qwen/Qwen3-4B"
    DEFAULT_STUDENT_REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
    DEFAULT_TEACHER_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
    DEFAULT_MIN_FREE_GIB="70"
    ;;
  r1_success)
    DEFAULT_STUDENT="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEFAULT_TEACHER="hbx/JustRL-DeepSeek-1.5B"
    DEFAULT_STUDENT_REVISION="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    DEFAULT_TEACHER_REVISION="0637e4096c789c67f9eecbe8355e0bdeddede1c2"
    DEFAULT_MIN_FREE_GIB="70"
    ;;
  r1_failure)
    DEFAULT_STUDENT="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEFAULT_TEACHER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    DEFAULT_STUDENT_REVISION="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
    DEFAULT_TEACHER_REVISION="916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    DEFAULT_MIN_FREE_GIB="72"
    ;;
  *)
    echo "MODEL_PAIR must be one of: paper, 4b, mismatch, r1_success, r1_failure" >&2
    exit 2
    ;;
esac

STUDENT_MODEL="${STUDENT_MODEL:-${DEFAULT_STUDENT}}"
TEACHER_MODEL="${TEACHER_MODEL:-${DEFAULT_TEACHER}}"
MIN_FREE_GIB="${MIN_FREE_GIB:-${DEFAULT_MIN_FREE_GIB}}"
if [[ "${STUDENT_MODEL}" == "${DEFAULT_STUDENT}" ]]; then
  STUDENT_REVISION="${STUDENT_REVISION:-${DEFAULT_STUDENT_REVISION}}"
else
  STUDENT_REVISION="${STUDENT_REVISION:-auto}"
fi
if [[ "${TEACHER_MODEL}" == "${DEFAULT_TEACHER}" ]]; then
  TEACHER_REVISION="${TEACHER_REVISION:-${DEFAULT_TEACHER_REVISION}}"
else
  TEACHER_REVISION="${TEACHER_REVISION:-auto}"
fi

case "${PRESET}" in
  smoke)
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    N_RESPONSES="${N_RESPONSES:-1}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
    SAVE_FREQ="${SAVE_FREQ:-2}"
    MAX_ACTOR_CKPTS_TO_KEEP="${MAX_ACTOR_CKPTS_TO_KEEP:-2}"
    CHECKPOINT_SAVE_MODE="${CHECKPOINT_SAVE_MODE:-full}"
    ;;
  pilot)
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
    N_RESPONSES="${N_RESPONSES:-2}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1600}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
    SAVE_FREQ="${SAVE_FREQ:-50}"
    MAX_ACTOR_CKPTS_TO_KEEP="${MAX_ACTOR_CKPTS_TO_KEEP:-2}"
    CHECKPOINT_SAVE_MODE="${CHECKPOINT_SAVE_MODE:-full}"
    ;;
  paper)
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    N_RESPONSES="${N_RESPONSES:-4}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-7168}"
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
    SAVE_FREQ="${SAVE_FREQ:-20}"
    # Paper curves require every scheduled actor checkpoint.  Zero means no
    # rotation in VERL; model-only shards keep this bounded enough for merging
    # and avg@16 evaluation without retaining optimizer/RNG training state.
    MAX_ACTOR_CKPTS_TO_KEEP="${MAX_ACTOR_CKPTS_TO_KEEP:-0}"
    CHECKPOINT_SAVE_MODE="${CHECKPOINT_SAVE_MODE:-model_only}"
    ;;
  *)
    echo "PRESET must be one of: smoke, pilot, paper" >&2
    exit 2
    ;;
esac

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_MODEL_LEN}}"

case "${ENABLE_THINKING}" in
  auto|true|false) ;;
  *)
    echo "ENABLE_THINKING must be auto, true, or false" >&2
    exit 2
    ;;
esac

case "${DATA_SHUFFLE}" in
  true|false) ;;
  *) echo "DATA_SHUFFLE must be true or false" >&2; exit 2 ;;
esac

case "${PIN_MODEL_SNAPSHOTS}" in
  true) ;;
  false)
    echo "PIN_MODEL_SNAPSHOTS=false is intentionally unsupported: VERL does not pass Hub revisions to model loading." >&2
    echo "Keep snapshot pinning enabled; local model directories are supported by the same pinned path." >&2
    exit 2
    ;;
  *) echo "PIN_MODEL_SNAPSHOTS must be true" >&2; exit 2 ;;
esac

case "${ATTN_IMPLEMENTATION}" in
  sdpa|eager|flash_attention_2) ;;
  *) echo "ATTN_IMPLEMENTATION must be sdpa, eager, or flash_attention_2" >&2; exit 2 ;;
esac

case "${USE_REMOVE_PADDING}" in
  true|false) ;;
  *) echo "USE_REMOVE_PADDING must be true or false" >&2; exit 2 ;;
esac

case "${TOP_K_STRATEGY}" in
  only_stu|only_tch|intersection|union|union-intersection) ;;
  *)
    echo "Unsupported TOP_K_STRATEGY: ${TOP_K_STRATEGY}" >&2
    exit 2
    ;;
esac

case "${REWARD_WEIGHT_MODE}" in
  student_p|teacher_p|none) ;;
  *) echo "REWARD_WEIGHT_MODE must be student_p, teacher_p, or none" >&2; exit 2 ;;
esac

case "${SUPPORT_WEIGHT_NORMALIZATION}" in
  author|selected) ;;
  *) echo "SUPPORT_WEIGHT_NORMALIZATION must be author or selected" >&2; exit 2 ;;
esac

case "${CHECKPOINT_SAVE_MODE}" in
  full|model_only) ;;
  *) echo "CHECKPOINT_SAVE_MODE must be full or model_only" >&2; exit 2 ;;
esac

case "${RESUME_MODE}" in
  disable|auto) ;;
  *) echo "RESUME_MODE must be disable or auto" >&2; exit 2 ;;
esac

if [[ "${RESUME_MODE}" == "auto" && "${CHECKPOINT_SAVE_MODE}" != "full" ]]; then
  echo "RESUME_MODE=auto requires CHECKPOINT_SAVE_MODE=full so optimizer/RNG state can be restored" >&2
  exit 2
fi

if ! [[ "${GPU_TELEMETRY_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "GPU_TELEMETRY_INTERVAL must be a non-negative number of seconds" >&2
  exit 2
fi

for integer_name in POSITION_ENTROPY_LOG_FREQ POSITION_ENTROPY_START_STEP POSITION_ENTROPY_BIN_SIZE MAX_ACTOR_CKPTS_TO_KEEP; do
  integer_value="${!integer_name}"
  if ! [[ "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if [[ "${POSITION_ENTROPY_BIN_SIZE}" == "0" ]]; then
  echo "POSITION_ENTROPY_BIN_SIZE must be positive" >&2
  exit 2
fi
if ! [[ "${SAVE_FREQ}" =~ ^-?[0-9]+$ ]]; then
  echo "SAVE_FREQ must be an integer" >&2
  exit 2
fi
if (( SAVE_FREQ > 0 )) && [[ "${MAX_ACTOR_CKPTS_TO_KEEP}" == "0" && "${CHECKPOINT_SAVE_MODE}" != "model_only" ]]; then
  echo "Unlimited checkpoint retention (MAX_ACTOR_CKPTS_TO_KEEP=0) requires CHECKPOINT_SAVE_MODE=model_only" >&2
  exit 2
fi

declare -A milestone_seen=()
normalized_milestone_steps="${MILESTONE_STEPS//,/ }"
for milestone_step in ${normalized_milestone_steps}; do
  if ! [[ "${milestone_step}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MILESTONE_STEPS must contain unique positive integers" >&2
    exit 2
  fi
  if [[ -v "milestone_seen[${milestone_step}]" ]]; then
    echo "MILESTONE_STEPS repeats step ${milestone_step}" >&2
    exit 2
  fi
  milestone_seen[${milestone_step}]=1
  if (( SAVE_FREQ <= 0 )); then
    echo "MILESTONE_STEPS requires SAVE_FREQ > 0" >&2
    exit 2
  fi
  if [[ -n "${TOTAL_TRAINING_STEPS}" ]] && (( milestone_step > TOTAL_TRAINING_STEPS )); then
    echo "milestone step ${milestone_step} exceeds TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}" >&2
    exit 2
  fi
  if (( milestone_step % SAVE_FREQ != 0 )) && \
     { [[ -z "${TOTAL_TRAINING_STEPS}" ]] || (( milestone_step != TOTAL_TRAINING_STEPS )); }; then
    echo "milestone step ${milestone_step} is not produced by SAVE_FREQ=${SAVE_FREQ}" >&2
    exit 2
  fi
done
if [[ -n "${MILESTONE_STEPS}" && "${CHECKPOINT_SAVE_MODE}" != "full" ]]; then
  echo "MILESTONE_STEPS requires full recovery checkpoints; use CHECKPOINT_SAVE_MODE=full" >&2
  exit 2
fi

if ! [[ "${TOP_K}" =~ ^[0-9]+$ ]]; then
  echo "TOP_K must be a non-negative integer" >&2
  exit 2
fi
if [[ "${TOP_K}" == "0" && "${TOP_K_STRATEGY}" != "only_stu" ]]; then
  echo "TOP_K=0 is sampled-token OPD; use TOP_K_STRATEGY=only_stu" >&2
  exit 2
fi

student_name="${STUDENT_MODEL##*/}"
teacher_name="${TEACHER_MODEL##*/}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${MODEL_PAIR}-${PRESET}-k${TOP_K}-${TOP_K_STRATEGY}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd-${EXPERIMENT_TAG}-${student_name}-to-${teacher_name}-${timestamp}}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${EXPERIMENT_NAME}}"
METRICS_FILE="${METRICS_FILE:-${RUN_DIR}/metrics.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
# Permanent milestone grid lives at the cell root so finalize-training and the
# evaluation/probe contracts can validate it; per-attempt recovery checkpoints
# stay inside the attempt run dir.
MILESTONE_DIR="${MILESTONE_DIR:-$(dirname -- "${RUN_DIR}")/milestones}"

VAL_FILES="['${REPO_ROOT}/datasets/test_data/AIME24/test.parquet','${REPO_ROOT}/datasets/test_data/AIME25/test.parquet','${REPO_ROOT}/datasets/test_data/AMC23/test.parquet']"

hydra_args=(
  "algorithm.adv_estimator=token_reward_direct"
  "algorithm.grpo_outcome_weight=1.0"
  "algorithm.use_kl_in_reward=false"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=true"
  "data.truncation=error"
  "data.return_raw_chat=true"
  "data.shuffle=${DATA_SHUFFLE}"
  "data.seed=${SEED}"
  "actor_rollout_ref.model.path=${STUDENT_MODEL}"
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
  "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
  "actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=true"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.actor.use_kl_loss=false"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.fsdp_config.param_offload=false"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.forward_prefetch=true"
  "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.dtype=bfloat16"
  "actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}"
  "actor_rollout_ref.rollout.seed=${ROLLOUT_SEED}"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.repetition_penalty=1.0"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKENS_PER_GPU}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.n=${N_RESPONSES}"
  "actor_rollout_ref.rollout.calculate_log_probs=true"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}"
  "+actor_rollout_ref.rollout.log_prob_top_k=${TOP_K}"
  "+actor_rollout_ref.rollout.top_k_strategy=${TOP_K_STRATEGY}"
  "+actor_rollout_ref.rollout.reward_weight_mode=${REWARD_WEIGHT_MODE}"
  "actor_rollout_ref.rollout.support_weight_normalization=${SUPPORT_WEIGHT_NORMALIZATION}"
  "+actor_rollout_ref.rollout.teacher_temperature=${TEACHER_TEMPERATURE}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "reward_model.enable=true"
  "reward_model.reward_manager=${REWARD_MANAGER}"
  "reward_model.model.path=${TEACHER_MODEL}"
  "reward_model.model.input_tokenizer=null"
  "reward_model.model.use_remove_padding=${USE_REMOVE_PADDING}"
  "+reward_model.model.attn_implementation=${ATTN_IMPLEMENTATION}"
  "reward_model.model.fsdp_config.param_offload=false"
  "reward_model.model.fsdp_config.forward_prefetch=true"
  "+reward_model.model.dtype=bfloat16"
  "reward_model.micro_batch_size_per_gpu=1"
  "reward_model.use_dynamic_bsz=true"
  "reward_model.forward_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU}"
  "custom_reward_function.path=${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_math/__init__.py"
  "custom_reward_function.name=reward_func"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.logger=['console','file']"
  "trainer.project_name=Rethinking-OPD-2xA100"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP}"
  "trainer.total_epochs=1"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
  "trainer.is_plot=false"
  "trainer.position_entropy_log_freq=${POSITION_ENTROPY_LOG_FREQ}"
  "trainer.position_entropy_start_step=${POSITION_ENTROPY_START_STEP}"
  "trainer.position_entropy_bin_size=${POSITION_ENTROPY_BIN_SIZE}"
)

case "${CHECKPOINT_SAVE_MODE}" in
  full)
    hydra_args+=(
      "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra']"
      "actor_rollout_ref.actor.checkpoint.load_contents=['model','optimizer','extra']"
    )
    ;;
  model_only)
    hydra_args+=(
      "actor_rollout_ref.actor.checkpoint.save_contents=['model']"
      "actor_rollout_ref.actor.checkpoint.load_contents=['model']"
    )
    ;;
esac

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  hydra_args+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
if [[ "${ENABLE_THINKING}" != "auto" ]]; then
  hydra_args+=("+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}")
fi
for extra_override in "$@"; do
  normalized_override="${extra_override#++}"
  normalized_override="${normalized_override#+}"
  case "${normalized_override}" in
    actor_rollout_ref.model.path=*|reward_model.model.path=*)
      echo "Set STUDENT_MODEL/TEACHER_MODEL (and revisions) instead of a CLI model.path override" >&2
      exit 2
      ;;
    actor_rollout_ref.actor.checkpoint.save_contents=*|actor_rollout_ref.actor.checkpoint.load_contents=*|trainer.max_actor_ckpt_to_keep=*|trainer.remove_previous_ckpt_in_save=*|trainer.save_freq=*|trainer.resume_mode=*|trainer.resume_from_path=*)
      echo "Set CHECKPOINT_SAVE_MODE, MAX_ACTOR_CKPTS_TO_KEEP, SAVE_FREQ, and RESUME_MODE through the validated environment interface" >&2
      exit 2
      ;;
  esac
done
if (($#)); then
  hydra_args+=("$@")
fi

python_bin="${PYTHON_BIN:-python}"
command=("${python_bin}" -m verl.trainer.main_ppo "${hydra_args[@]}")

print_summary() {
  printf '%s\n' \
    "OPD configuration" \
    "  preset/model pair : ${PRESET} / ${MODEL_PAIR}" \
    "  student -> teacher: ${STUDENT_MODEL} -> ${TEACHER_MODEL}" \
    "  pinned revisions   : ${STUDENT_REVISION} -> ${TEACHER_REVISION}" \
    "  data               : ${TRAIN_DATA}" \
    "  data/rollout seed  : ${SEED} / ${ROLLOUT_SEED}" \
    "  batch/rollouts     : ${TRAIN_BATCH_SIZE} prompts x ${N_RESPONSES}" \
    "  prompt/response    : ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}" \
    "  top-k/strategy     : ${TOP_K} / ${TOP_K_STRATEGY} (${REWARD_WEIGHT_MODE}, ${SUPPORT_WEIGHT_NORMALIZATION})" \
    "  position entropy   : every ${POSITION_ENTROPY_LOG_FREQ} steps from ${POSITION_ENTROPY_START_STEP}, bin ${POSITION_ENTROPY_BIN_SIZE}" \
    "  checkpoints        : every ${SAVE_FREQ} steps, mode=${CHECKPOINT_SAVE_MODE}, retention=${MAX_ACTOR_CKPTS_TO_KEEP} (0=unlimited)" \
    "  resume/milestones  : ${RESUME_MODE} / ${MILESTONE_STEPS:-none}" \
    "  GPU telemetry      : every ${GPU_TELEMETRY_INTERVAL}s (0=disabled)" \
    "  actor/teacher dtype: FP32 optimizer state + BF16 FSDP forward / BF16" \
    "  attention/rmpad    : ${ATTN_IMPLEMENTATION} / ${USE_REMOVE_PADDING}" \
    "  shuffle/pin models : ${DATA_SHUFFLE} / ${PIN_MODEL_SNAPSHOTS}" \
    "  GPUs/min free      : ${N_GPUS} / ${MIN_FREE_GIB} GiB each" \
    "  run directory      : ${RUN_DIR}"
}

print_summary

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'VLLM_USE_FLASHINFER_SAMPLER=%q ' "${VLLM_USE_FLASHINFER_SAMPLER}"
  printf 'VERL_FILE_LOGGER_PATH=%q ' "${METRICS_FILE}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse an existing RUN_DIR: ${RUN_DIR}" >&2
  echo "Choose a new attempt RUN_DIR; RESUME_MODE=auto restores from the separately selected CHECKPOINT_DIR." >&2
  exit 1
fi
if [[ -e "${METRICS_FILE}" ]]; then
  echo "Refusing to overwrite an existing METRICS_FILE: ${METRICS_FILE}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "$(dirname -- "${METRICS_FILE}")"
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  "${python_bin}" "${SCRIPT_DIR}/preflight.py" \
    --repo-root "${REPO_ROOT}" \
    --student "${STUDENT_MODEL}" \
    --teacher "${TEACHER_MODEL}" \
    --student-revision "${STUDENT_REVISION}" \
    --teacher-revision "${TEACHER_REVISION}" \
    --top-k-strategy "${TOP_K_STRATEGY}" \
    --train-data "${TRAIN_DATA}" \
    --gpus "${N_GPUS}" \
    --min-free-gib "${MIN_FREE_GIB}" 2>&1 | tee "${RUN_DIR}/preflight.log"
fi

if [[ "${PIN_MODEL_SNAPSHOTS}" == "true" ]]; then
  snapshot_manifest="${RUN_DIR}/model_snapshots.json"
  "${python_bin}" "${SCRIPT_DIR}/pin_models.py" \
    --student "${STUDENT_MODEL}" \
    --student-revision "${STUDENT_REVISION}" \
    --teacher "${TEACHER_MODEL}" \
    --teacher-revision "${TEACHER_REVISION}" \
    --output "${snapshot_manifest}"
  runtime_student_model="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["student"]["snapshot_path"])' "${snapshot_manifest}")"
  runtime_teacher_model="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher"]["snapshot_path"])' "${snapshot_manifest}")"
  for index in "${!hydra_args[@]}"; do
    case "${hydra_args[index]}" in
      actor_rollout_ref.model.path=*) hydra_args[index]="actor_rollout_ref.model.path=${runtime_student_model}" ;;
      reward_model.model.path=*) hydra_args[index]="reward_model.model.path=${runtime_teacher_model}" ;;
    esac
  done
  command=("${python_bin}" -m verl.trainer.main_ppo "${hydra_args[@]}")
  printf 'Pinned student snapshot: %s\nPinned teacher snapshot: %s\n' \
    "${runtime_student_model}" "${runtime_teacher_model}"
fi

export VERL_FILE_LOGGER_PATH="${METRICS_FILE}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD:-0.95}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_BLOCKING_WAIT=1
# This host's system nvcc can be newer than its NVIDIA driver.  FlashInfer's
# JIT sampler would then compile an unusable cubin; vLLM's native sampler is
# numerically equivalent for this experiment and avoids that host dependency.
export VLLM_USE_FLASHINFER_SAMPLER

runtime_control_enabled=0
if [[ "${RESUME_MODE}" == "auto" || -n "${MILESTONE_STEPS}" || "${GPU_TELEMETRY_INTERVAL}" != "0" || -n "${TRAINING_MATRIX_ROOT}" ]]; then
  runtime_control_enabled=1
fi
if [[ "${runtime_control_enabled}" == "1" ]]; then
  runtime_prepare_args=(
    prepare-attempt
    --run-dir "${RUN_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --milestone-dir "${MILESTONE_DIR}"
    --metrics-file "${METRICS_FILE}"
    --resume-mode "${RESUME_MODE}"
    --milestone-steps "${MILESTONE_STEPS}"
  )
  if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    runtime_prepare_args+=(--expected-final-step "${TOTAL_TRAINING_STEPS}")
  fi
  if (( MAX_ACTOR_CKPTS_TO_KEEP > 0 )); then
    runtime_prepare_args+=(--recovery-retention "${MAX_ACTOR_CKPTS_TO_KEEP}")
  fi
  "${python_bin}" "${SCRIPT_DIR}/scientific_runtime.py" "${runtime_prepare_args[@]}" \
    >"${RUN_DIR}/runtime_prepare.json"
fi

git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_DIR}/code_revision.txt"
git -C "${REPO_ROOT}" status --short >"${RUN_DIR}/git_status.txt"
git -C "${REPO_ROOT}" diff HEAD --binary >"${RUN_DIR}/tracked_changes.patch"
mkdir -p "${RUN_DIR}/reproduction_code/tests" "${RUN_DIR}/reproduction_code/verl/verl/utils"
cp "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/*.py "${SCRIPT_DIR}"/*.md "${SCRIPT_DIR}"/*.txt \
  "${RUN_DIR}/reproduction_code/"
cp "${SCRIPT_DIR}"/*.json "${RUN_DIR}/reproduction_code/"
cp "${SCRIPT_DIR}"/tests/*.py "${RUN_DIR}/reproduction_code/tests/"
# tracked_changes.patch captures tracked worker/trainer edits; this explicit
# copy preserves the new untracked helper without sweeping unrelated local
# files (which could include secrets or large private data) into an artifact.
cp "${REPO_ROOT}/verl/verl/utils/opd.py" "${RUN_DIR}/reproduction_code/verl/verl/utils/opd.py"
printenv | grep -E '^(CUDA_VISIBLE_DEVICES|HF_HOME|HF_HUB_OFFLINE|TRANSFORMERS_CACHE|TRANSFORMERS_OFFLINE|VERL_FILE_LOGGER_PATH|VLLM_USE_FLASHINFER_SAMPLER|NCCL_TIMEOUT|TORCH_NCCL_BLOCKING_WAIT|RAY_memory_usage_threshold)=' >"${RUN_DIR}/selected_environment.txt" || true
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf '# Resolved command record. Choose new metrics/checkpoint paths before replaying in place.\n'
  for variable_name in VERL_FILE_LOGGER_PATH PYTHONUNBUFFERED HYDRA_FULL_ERROR TOKENIZERS_PARALLELISM RAY_memory_usage_threshold NCCL_TIMEOUT TORCH_NCCL_BLOCKING_WAIT VLLM_USE_FLASHINFER_SAMPLER; do
    printf 'export %s=%q\n' "${variable_name}" "${!variable_name}"
  done
  for variable_name in CUDA_VISIBLE_DEVICES HF_HOME HF_HUB_OFFLINE TRANSFORMERS_CACHE TRANSFORMERS_OFFLINE; do
    if [[ -v "${variable_name}" ]]; then
      printf 'export %s=%q\n' "${variable_name}" "${!variable_name}"
    fi
  done
  printf 'cd %q\n' "${REPO_ROOT}"
  printf '%q ' "${command[@]}"
  printf '\n'
} >"${RUN_DIR}/command.sh"

cd "${REPO_ROOT}"
runtime_monitor_pid=""
runtime_stop_file="${RUN_DIR}/runtime_monitor.stop"
stop_runtime_monitor() {
  if [[ -n "${runtime_monitor_pid}" ]]; then
    touch "${runtime_stop_file}"
    wait "${runtime_monitor_pid}" 2>/dev/null || true
    runtime_monitor_pid=""
  fi
}
trap stop_runtime_monitor EXIT

if [[ "${runtime_control_enabled}" == "1" ]]; then
  runtime_monitor_args=(
    monitor
    --run-dir "${RUN_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --milestone-dir "${MILESTONE_DIR}"
    --milestone-steps "${MILESTONE_STEPS}"
    --telemetry-interval "${GPU_TELEMETRY_INTERVAL}"
    --stop-file "${runtime_stop_file}"
  )
  if (( MAX_ACTOR_CKPTS_TO_KEEP > 0 )); then
    runtime_monitor_args+=(--recovery-retention "${MAX_ACTOR_CKPTS_TO_KEEP}")
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    runtime_monitor_args+=(--cuda-devices "${CUDA_VISIBLE_DEVICES}")
  fi
  if [[ -n "${TRAINING_MATRIX_ROOT}" ]]; then
    runtime_monitor_args+=(--training-matrix-root "${TRAINING_MATRIX_ROOT}")
  fi
  "${python_bin}" "${SCRIPT_DIR}/scientific_runtime.py" "${runtime_monitor_args[@]}" \
    >"${RUN_DIR}/runtime_monitor.log" 2>&1 &
  runtime_monitor_pid=$!
fi

set +e
"${command[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
training_exit_code=$?
set -e

monitor_exit_code=0
if [[ -n "${runtime_monitor_pid}" ]]; then
  touch "${runtime_stop_file}"
  set +e
  wait "${runtime_monitor_pid}"
  monitor_exit_code=$?
  set -e
  runtime_monitor_pid=""
fi
trap - EXIT

if [[ "${runtime_control_enabled}" == "1" ]]; then
  set +e
  "${python_bin}" "${SCRIPT_DIR}/scientific_runtime.py" finalize-attempt \
    --run-dir "${RUN_DIR}" --exit-code "${training_exit_code}" \
    >"${RUN_DIR}/runtime_finalize.json"
  finalize_exit_code=$?
  set -e
  if (( training_exit_code == 0 && finalize_exit_code != 0 )); then
    exit "${finalize_exit_code}"
  fi
  # Canonical metric materialization: require steps exactly 1..N on success.
  # A validation failure here is fail-closed and marks the cell incomplete.
  if (( training_exit_code == 0 )); then
    set +e
    "${python_bin}" "${SCRIPT_DIR}/scientific_runtime.py" finalize-training \
      --cell-root "$(dirname -- "${RUN_DIR}")" \
      --expected-final-step "${TOTAL_TRAINING_STEPS:-0}" \
      --milestone-steps "${MILESTONE_STEPS}" \
      >"${RUN_DIR}/runtime_training_finalize.json"
    training_finalize_exit_code=$?
    set -e
    if (( training_finalize_exit_code != 0 )); then
      echo "Canonical training finalize failed; see ${RUN_DIR}/runtime_training_finalize.json" >&2
      exit "${training_finalize_exit_code}"
    fi
  fi
fi
if (( training_exit_code != 0 )); then
  exit "${training_exit_code}"
fi
if (( monitor_exit_code != 0 )); then
  echo "Runtime monitor failed; see ${RUN_DIR}/runtime_monitor.log" >&2
  exit "${monitor_exit_code}"
fi
