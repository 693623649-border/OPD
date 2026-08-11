#!/usr/bin/env bash
# Formal cold-start SFT launcher corresponding to paper Table 3.
# It uses the repository-vendored LLaMA-Factory 0.9.5 source tree and never
# installs packages.  PRESET=paper preserves the disclosed Table 3 values;
# PRESET=smoke is only a two-step 2xA100 integration check.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-${REPO_ROOT}/LlamaFactory}"
PYTHON_BIN="${SFT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
PRESET="${PRESET:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
N_GPUS="${N_GPUS:-2}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/artifacts/upstream/manual-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/sft}"
DATASET_JSONL="${DATASET_JSONL:-${REPO_ROOT}/artifacts/upstream/cold-start-rollout/cold_start_sft.jsonl}"

MODEL_ID="Qwen/Qwen3-1.7B-Base"
MODEL_REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
LLAMAFACTORY_REQUIRED_VERSION="0.9.5"

case "${PRESET}" in
  smoke)
    FIDELITY_LABEL="engineering-smoke; shortened sequence and batch on 2xA100; not a paper result"
    CUTOFF_LEN="${CUTOFF_LEN:-2048}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    MAX_SAMPLES="${MAX_SAMPLES:-4}"
    MAX_STEPS="${MAX_STEPS:-2}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    ;;
  paper)
    FIDELITY_LABEL="Table 3 disclosed parameters; 2xA100 hardware adaptation (paper GPU count undisclosed)"
    CUTOFF_LEN="14336"
    PER_DEVICE_BATCH_SIZE="8"
    MAX_SAMPLES=""
    MAX_STEPS=""
    SAVE_STEPS="${SAVE_STEPS:-200}"
    ;;
  *)
    echo "PRESET must be one of: smoke, paper" >&2
    exit 2
    ;;
esac

if [[ "${N_GPUS}" != "2" ]]; then
  echo "This local formal SFT launcher is scoped to exactly 2 GPUs; got N_GPUS=${N_GPUS}." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi
if (($#)); then
  echo "CLI hyperparameter overrides are disabled for the formal Table 3 launcher." >&2
  exit 2
fi

DATASET_DIR="${OUTPUT_DIR}/dataset_registry"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR}/checkpoints"
DATASET_INFO="${DATASET_DIR}/dataset_info.json"

lf_args=(
  --model_name_or_path "${MODEL_ID}"
  --model_revision "${MODEL_REVISION}"
  --trust_remote_code true
  --stage sft
  --do_train true
  --finetuning_type full
  --deepspeed "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json"
  --flash_attn fa2
  --enable_liger_kernel true
  --dataset formal_cold_start
  --dataset_dir "${DATASET_DIR}"
  --template qwen3
  --enable_thinking true
  --cutoff_len "${CUTOFF_LEN}"
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS:-16}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
  --output_dir "${TRAIN_OUTPUT_DIR}"
  --logging_steps 5
  --save_steps "${SAVE_STEPS}"
  --plot_loss true
  --overwrite_output_dir false
  --save_only_model true
  --report_to none
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps 1
  --gradient_checkpointing true
  --learning_rate 1.0e-5
  --num_train_epochs 1.0
  --lr_scheduler_type cosine
  --warmup_ratio 0.05
  --bf16 true
  --ddp_timeout 180000000
  --val_size 0
  --seed "${SEED}"
  --data_seed "${SEED}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  lf_args+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  lf_args+=(--max_steps "${MAX_STEPS}")
fi

# Invoke the checked-in package source directly.  This avoids accidentally
# selecting an unrelated globally installed `llamafactory-cli` executable.
command=("${PYTHON_BIN}" -m llamafactory.cli train "${lf_args[@]}")

printf '%s\n' \
  "Cold-start student SFT configuration (paper Table 3)" \
  "  fidelity           : ${FIDELITY_LABEL}" \
  "  student            : ${MODEL_ID}@${MODEL_REVISION}" \
  "  corpus             : ${DATASET_JSONL}" \
  "  framework          : vendored LLaMA-Factory ${LLAMAFACTORY_REQUIRED_VERSION} (${LLAMAFACTORY_ROOT})" \
  "  objective/template : full-parameter SFT / qwen3" \
  "  epochs/sequence    : 1 / ${CUTOFF_LEN}" \
  "  per-device/accum   : ${PER_DEVICE_BATCH_SIZE} / 1" \
  "  lr/schedule/warmup : 1e-5 / cosine / 0.05" \
  "  precision          : BF16" \
  "  local hardware     : 2xA100-80G (explicit adaptation)"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'PYTHONPATH=%q FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 ' \
    "${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

version_file="${LLAMAFACTORY_ROOT}/src/llamafactory/extras/env.py"
if [[ ! -f "${version_file}" ]]; then
  echo "Vendored LLaMA-Factory source is missing: ${version_file}" >&2
  exit 1
fi
if ! grep -Eq 'VERSION = "0\.9\.5([^"[:space:]]*)?"' "${version_file}"; then
  echo "Vendored LLaMA-Factory is not the required 0.9.5 source line: ${version_file}" >&2
  exit 1
fi
if [[ ! -f "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json" ]]; then
  echo "LLaMA-Factory DeepSpeed ZeRO-2 config is missing." >&2
  exit 1
fi
if [[ ! -f "${DATASET_JSONL}" ]]; then
  echo "Filtered cold-start JSONL is missing: ${DATASET_JSONL}" >&2
  echo "Complete cold-start-rollout first or set DATASET_JSONL explicitly." >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse SFT OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 1
fi

export PYTHONPATH="${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if ! "${PYTHON_BIN}" -c 'import deepspeed, fire, liger_kernel, llamafactory, matplotlib, omegaconf, sentencepiece, tiktoken, torch, torchaudio, torchdata, torchvision, transformers; assert llamafactory.__version__.startswith("0.9.5")'; then
  echo "SFT preflight failed. Use an isolated Python 3.11/3.12 environment with the vendored LLaMA-Factory 0.9.5 text-training dependencies." >&2
  echo "This launcher intentionally does not install or upgrade packages." >&2
  exit 1
fi

mkdir -p "${DATASET_DIR}"
"${PYTHON_BIN}" - "${DATASET_JSONL}" "${DATASET_INFO}" <<'PY'
import json
import os
import sys
from pathlib import Path

source = str(Path(sys.argv[1]).expanduser().resolve())
target = Path(sys.argv[2])
payload = {
    "formal_cold_start": {
        "file_name": source,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
}
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY

# LLaMA-Factory invokes the literal `torchrun` command for multi-GPU jobs.
# A venv created with --system-site-packages may not contain that entry point,
# in which case PATH would silently select the base interpreter and lose the
# isolated Transformers/DeepSpeed dependencies.  Bind torchrun to the exact
# SFT interpreter inside this immutable attempt directory.
LAUNCHER_BIN="${OUTPUT_DIR}/launcher_bin"
mkdir -p "${LAUNCHER_BIN}"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'exec %q -m torch.distributed.run "$@"\n' "${PYTHON_BIN}"
} >"${LAUNCHER_BIN}/torchrun"
chmod 0755 "${LAUNCHER_BIN}/torchrun"
export PATH="${LAUNCHER_BIN}:${PATH}"

{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'export PYTHONPATH=%q\n' "${PYTHONPATH}"
  printf 'export PATH=%q\n' "${PATH}"
  printf 'export FORCE_TORCHRUN=1\nexport NNODES=1\nexport NPROC_PER_NODE=2\n'
  printf 'cd %q\n' "${LLAMAFACTORY_ROOT}"
  printf '%q ' "${command[@]}"
  printf '\n'
} >"${OUTPUT_DIR}/command.sh"

export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE=2
export TOKENIZERS_PARALLELISM=true
cd "${LLAMAFACTORY_ROOT}"
"${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
touch "${OUTPUT_DIR}/_SUCCESS"
