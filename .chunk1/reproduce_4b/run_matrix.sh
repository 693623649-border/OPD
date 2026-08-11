#!/usr/bin/env bash
# Print a mechanism-reproduction matrix by default; set ACTION=run explicitly
# to execute the experiments sequentially.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${ACTION:-print}"
PRESET="${PRESET:-pilot}"

case "${ACTION}" in
  print) dry_run=1 ;;
  run) dry_run=0 ;;
  *) echo "ACTION must be print or run" >&2; exit 2 ;;
esac

run_one() {
  local tag="$1"
  local pair="$2"
  local top_k="$3"
  local strategy="$4"
  printf '\n### %s\n' "${tag}"
  PRESET="${PRESET}" \
  MODEL_PAIR="${pair}" \
  TOP_K="${top_k}" \
  TOP_K_STRATEGY="${strategy}" \
  EXPERIMENT_TAG="${tag}" \
  DRY_RUN="${dry_run}" \
    "${SCRIPT_DIR}/run_opd_4b.sh"
}

# Teacher compatibility / new-capability comparison.
run_one success_grpo_teacher paper 16 only_stu
run_one mismatch_nonthinking_teacher mismatch 16 only_stu

# Figure-7-like support-set mechanism ablation.
run_one overlap_only paper 16 intersection
run_one non_overlap paper 16 union-intersection

# Figure-15/16-like support-size ablation.  The success_grpo_teacher run above
# is already the Top-16 only_stu reference, so do not launch it a second time.
run_one topk_1 paper 1 only_stu
run_one topk_4 paper 4 only_stu
