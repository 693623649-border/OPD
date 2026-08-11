#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if (($# < 1 || $# > 2)); then
  echo "Usage: $0 <global_step_dir-or-actor_dir> [merged_output_dir]" >&2
  exit 2
fi

input_dir="$(realpath -- "$1")"
if [[ -d "${input_dir}/actor" ]]; then
  input_dir="${input_dir}/actor"
fi
if [[ ! -f "${input_dir}/fsdp_config.json" ]]; then
  echo "Not an actor FSDP checkpoint (missing fsdp_config.json): ${input_dir}" >&2
  exit 1
fi

if (($# == 2)); then
  output_dir="$(realpath -m -- "$2")"
else
  step_parent="$(dirname -- "${input_dir}")"
  output_dir="${step_parent}/actor_hf"
fi

if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing output: ${output_dir}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-${REPO_ROOT}/.venv-opd/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${PYTHON_FALLBACK:-python}"
fi

command=(
  "${python_bin}" -m verl.model_merger merge
  --backend fsdp
  --local_dir "${input_dir}"
  --target_dir "${output_dir}"
)

printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "$(dirname -- "${output_dir}")"
cd "${REPO_ROOT}"
"${command[@]}"

if [[ ! -f "${output_dir}/config.json" ]] || ! compgen -G "${output_dir}/*.safetensors" >/dev/null; then
  echo "Merge completed without an expected Hugging Face config/safetensors output" >&2
  exit 1
fi
echo "Merged Hugging Face checkpoint: ${output_dir}"
