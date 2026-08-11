#!/usr/bin/env bash
# Thin, auditable wrapper around run_ablations.py for the 2xA100 lite suite.
#
# Usage:
#   bash reproduce_4b/run_lite.sh ACTION [core|extended] [smoke|calibration|pilot] [RUNNER_ARGS...]
#
# Examples:
#   bash reproduce_4b/run_lite.sh plan core pilot
#   CUDA_VISIBLE_DEVICES=0,1 bash reproduce_4b/run_lite.sh run core smoke --yes --keep-going
#   bash reproduce_4b/run_lite.sh status extended pilot

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv-opd/bin/python"
MATRIX="${SCRIPT_DIR}/lite_matrix.json"
CALIBRATION_GUARD="${SCRIPT_DIR}/verify_lite_calibration.py"

ACTION="${1:-plan}"
if (( $# > 0 )); then shift; fi
TIER="${1:-core}"
if (( $# > 0 )); then shift; fi
PROTOCOL="${1:-pilot}"
if (( $# > 0 )); then shift; fi

case "${ACTION}" in
  plan|run|status) ;;
  *)
    echo "ACTION must be plan, run, or status" >&2
    exit 2
    ;;
esac

tier_args=()
case "${TIER}" in
  core) ;;
  extended) tier_args+=(--include-extensions) ;;
  *)
    echo "TIER must be core or extended" >&2
    exit 2
    ;;
esac

case "${PROTOCOL}" in
  smoke|calibration|pilot) ;;
  *)
    echo "PROTOCOL must be smoke, calibration, or pilot" >&2
    exit 2
    ;;
esac

# The generic runner accepts argparse long-option abbreviations.  The lite
# entrypoint deliberately exposes a strict allow-list so abbreviated reserved
# options cannot turn Core into Extended or redirect its immutable identity.
runner_args=("$@")
selected_items=()
index=0
while (( index < ${#runner_args[@]} )); do
  argument="${runner_args[index]}"
  case "${argument}" in
    --group|--cell|--seed|--run-root)
      if (( index + 1 >= ${#runner_args[@]} )); then
        echo "${argument} requires a value" >&2
        exit 2
      fi
      value="${runner_args[index + 1]}"
      if [[ "${value}" == --* ]]; then
        echo "${argument} requires a value" >&2
        exit 2
      fi
      if [[ "${argument}" == "--group" || "${argument}" == "--cell" ]]; then
        selected_items+=("${value}")
      fi
      ((index += 2))
      ;;
    --group=*|--cell=*)
      value="${argument#*=}"
      if [[ -z "${value}" ]]; then
        echo "${argument%%=*} requires a value" >&2
        exit 2
      fi
      selected_items+=("${value}")
      ((index += 1))
      ;;
    --seed=*|--run-root=*)
      value="${argument#*=}"
      if [[ -z "${value}" ]]; then
        echo "${argument%%=*} requires a value" >&2
        exit 2
      fi
      ((index += 1))
      ;;
    --yes|--acknowledge-multi-day|--keep-going|--retry-failed)
      ((index += 1))
      ;;
    --matrix|--matrix=*|--protocol|--protocol=*|--python-bin|--python-bin=*|--include-extensions)
      echo "${argument} is reserved by run_lite.sh; choose tier/protocol positionally" >&2
      exit 2
      ;;
    --*)
      echo "unsupported or abbreviated runner argument: ${argument}" >&2
      exit 2
      ;;
    *)
      echo "unexpected positional runner argument: ${argument}" >&2
      exit 2
      ;;
  esac
done

if [[ "${TIER}" == "core" ]]; then
  for selected in "${selected_items[@]}"; do
    case "${selected}" in
      fig4_6_deepseek_teachers|fig8_cold_start|fig11_13_response_length|\
      fig4-6-deepseek-r1-7b|fig6-deepseek-justrl-success|\
      fig8-base-only-opd|fig8-sft-then-opd|\
      fig12-length-1024|fig12-length-3072|fig12-length-7168)
        echo "${selected} belongs to the extended tier; rerun with TIER=extended" >&2
        exit 2
        ;;
    esac
  done
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable Python environment: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${MATRIX}" ]]; then
  echo "Missing lite matrix: ${MATRIX}" >&2
  exit 2
fi

if [[ "${ACTION}" == "run" && "${TIER}" == "extended" && "${PROTOCOL}" == "pilot" ]]; then
  if [[ ! -f "${CALIBRATION_GUARD}" ]]; then
    echo "Missing Extended calibration guard: ${CALIBRATION_GUARD}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" "${CALIBRATION_GUARD}" --matrix "${MATRIX}" "${runner_args[@]}"
fi

# Fixed identity arguments intentionally come after user arguments so a caller
# cannot redirect this wrapper to the paper matrix or relabel the protocol.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_ablations.py" \
  "${ACTION}" "$@" \
  --matrix "${MATRIX}" \
  --protocol "${PROTOCOL}" \
  --python-bin "${PYTHON_BIN}" \
  "${tier_args[@]}"
