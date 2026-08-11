#!/usr/bin/env bash
# Create an isolated Python 3.12 environment compatible with vendored VERL 0.7.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/.venv-opd}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
CONSTRAINTS_FILE="${OPD_CONSTRAINTS_FILE:-${SCRIPT_DIR}/constraints-2xa100-cu128.txt}"
# The host image may inject an unavailable private mirror.  Use public PyPI by
# default, while still allowing an explicit, trusted mirror for air-gapped use.
export PIP_INDEX_URL="${OPD_PIP_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "Validated dependency constraints do not exist: ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

python_version="$(${BOOTSTRAP_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.12" ]]; then
  echo "Python 3.12 is required; ${BOOTSTRAP_PYTHON} is ${python_version}" >&2
  exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  run "${BOOTSTRAP_PYTHON}" -m venv "${ENV_DIR}"
fi

venv_python="${ENV_DIR}/bin/python"
if [[ "${DRY_RUN}" == "1" && ! -x "${venv_python}" ]]; then
  # The path does not exist in a dry run; retaining it makes every printed
  # command directly reusable after the first command has run.
  :
elif [[ ! -x "${venv_python}" ]]; then
  echo "virtual environment was not created at ${ENV_DIR}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  venv_python_version="$("${venv_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${venv_python_version}" != "3.12" ]]; then
    echo "Existing environment at ${ENV_DIR} uses Python ${venv_python_version}; Python 3.12 is required" >&2
    exit 1
  fi
fi

run "${venv_python}" -m pip install --upgrade "pip==26.2" "setuptools==79.0.1" "wheel==0.47.0"
run "${venv_python}" -m pip install --no-cache-dir --constraint "${CONSTRAINTS_FILE}" "vllm==0.11.0"
run "${venv_python}" -m pip install --no-cache-dir \
  --constraint "${CONSTRAINTS_FILE}" \
  "transformers[hf_xet]==4.57.6" accelerate datasets peft hf-transfer \
  "numpy<2.0" "pyarrow>=19.0" pandas \
  "opencv-python-headless==4.11.0.86" \
  "scipy==1.16.3" "cupy-cuda12x==13.6.0" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
  "ray[default]==2.56.1" codetiming hydra-core pylatexenc \
  dill pybind11 liger-kernel mathruler latex2sympy2_extended math-verify \
  "nvidia-ml-py>=12.560.30" packaging matplotlib pytest \
  tensorboard wandb
run "${venv_python}" -m pip install --no-cache-dir --no-build-isolation \
  --constraint "${CONSTRAINTS_FILE}" "flashinfer-python==0.3.1"
FLASH_ATTN_WHEEL="${OPD_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl#sha256=15db5bb6524dcbf292c3c116aa4c2fa823b80abff0eb5bc58454107bca1ba0c2}"
run "${venv_python}" -m pip install --no-cache-dir \
  --constraint "${CONSTRAINTS_FILE}" \
  "${FLASH_ATTN_WHEEL}"
run "${venv_python}" -m pip install --no-deps --editable "${REPO_ROOT}/verl"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Environment would be created at %s\n' "${ENV_DIR}"
  exit 0
fi

lock_dir="${REPO_ROOT}/artifacts/environment"
mkdir -p "${lock_dir}"
"${venv_python}" -m pip check
"${venv_python}" -m pip freeze >"${lock_dir}/pip-freeze.txt"
"${venv_python}" - <<'PY'
import torch
import transformers
import verl
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("verl", getattr(verl, "__version__", "unknown"), verl.__file__)
PY
printf 'Environment ready. Activate with: source %q/bin/activate\n' "${ENV_DIR}"
