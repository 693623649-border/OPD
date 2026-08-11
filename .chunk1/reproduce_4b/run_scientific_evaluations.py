#!/usr/bin/env python3
"""Run the immutable trend/exact evaluation grids registered by scientific contracts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EVALUATION_IDS = {"trend": "trend-n4", "exact": "exact-avg16"}


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _positive_steps(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    steps = [step for step in value if step != 0]
    if (
        any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        raise ValueError(f"{label} must be sorted, unique, and non-negative")
    return steps


def registered_tier(contract: Mapping[str, Any], tier: str) -> tuple[dict[str, Any], list[int]]:
    if contract.get("protocol") != "scientific":
        raise ValueError("evaluation orchestration requires a scientific run contract")
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("run contract lacks evaluation metadata")
    sampling = evaluation.get(tier)
    if not isinstance(sampling, Mapping):
        raise ValueError(f"run contract lacks evaluation.{tier}")
    expected = {
        "trend": {"n": 4, "max_tokens": 4096},
        "exact": {"n": 16, "max_tokens": 31_744},
    }[tier]
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(
                f"registered {tier} sampling.{key}={sampling.get(key)!r}, expected {value}"
            )
    for key, value in {
        "temperature": 0.7,
        "top_p": 0.95,
        "thinking": "off",
        "seed": 42,
    }.items():
        if sampling.get(key) != value:
            raise ValueError(
                f"registered {tier} sampling.{key}={sampling.get(key)!r}, expected {value!r}"
            )
    steps = _positive_steps(evaluation.get(f"{tier}_steps"), f"evaluation.{tier}_steps")
    if not steps:
        raise ValueError(f"evaluation.{tier}_steps has no checkpoint target")
    return dict(sampling), steps


def completed_run_dir(cell_root: Path) -> Path:
    status = read_object(cell_root / "status.json", "cell status")
    if status.get("state") != "completed" and status.get("execution_state") not in {
        "training_complete",
        "evaluation_complete",
        "probe_complete",
        "rendered",
        "scientific_result_available",
    }:
        raise ValueError(f"{cell_root.name}: training is not complete")
    raw = status.get("run_dir")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{cell_root.name}: status lacks run_dir")
    run_dir = Path(raw).expanduser().resolve()
    if run_dir.parent != cell_root.resolve():
        raise ValueError(f"{cell_root.name}: status run_dir is outside the cell root")
    return run_dir


def _available_checkpoint_steps(run_dir: Path) -> set[int]:
    """Discover checkpoint steps that have actor weights (not pruned)."""
    discovered: set[int] = set()
    roots = (
        run_dir / "checkpoints",
        run_dir.parent / "checkpoints",
        run_dir / "milestones",
        run_dir.parent / "milestones",
    )
    for checkpoint_root in roots:
        for path in checkpoint_root.glob("global_step_*"):
            suffix = path.name.removeprefix("global_step_")
            if suffix.isdigit() and (path / "actor").is_dir():
                discovered.add(int(suffix))
    return discovered


def command_for_cell(
    repo_root: Path,
    cell_root: Path,
    tier: str,
    action: str,
    *,
    acknowledge_full_eval: bool,
) -> list[str]:
    contract = read_object(cell_root / "run_contract.json", "run contract")
    sampling, steps = registered_tier(contract, tier)
    run_dir = completed_run_dir(cell_root)
    # Guard exact-tier full evaluation before doing any filesystem work.
    if action == "run" and tier == "exact" and not acknowledge_full_eval:
        raise ValueError("exact tier requires --acknowledge-full-eval")
    # Filter to steps that have surviving actor checkpoints.  Rolling
    # checkpoints are pruned by max_actor_ckpt_to_keep, so only milestone
    # copies (or the last few checkpoints) survive.
    available = _available_checkpoint_steps(run_dir)
    filtered = [s for s in steps if s in available]
    if not filtered:
        raise FileNotFoundError(
            f"{cell_root.name}: no requested {tier} checkpoint steps have "
            f"actor weights; requested={steps}, available={sorted(available)}"
        )
    if len(filtered) < len(steps):
        skipped = sorted(set(steps) - set(filtered))
        print(
            f"[{cell_root.name}/{tier}] skipping {len(skipped)} checkpoint(s) "
            f"without actor weights: {skipped}",
            flush=True,
        )
    command = [
        str(repo_root / ".venv-opd/bin/python"),
        str(repo_root / "reproduce_4b/evaluate_ablation.py"),
        action,
        "--run-dir",
        str(run_dir),
        "--evaluation-id",
        EVALUATION_IDS[tier],
        "--n",
        str(sampling["n"]),
        "--max-tokens",
        str(sampling["max_tokens"]),
        "--seed",
        str(sampling["seed"]),
    ]
    for step in filtered:
        command.extend(["--checkpoint-step", str(step)])
    if action == "run":
        command.append("--yes")
        if acknowledge_full_eval:
            command.append("--acknowledge-full-eval")
    return command


def selected_cells(suite_root: Path, requested: set[str]) -> list[Path]:
    contract_cells = sorted(path.parent for path in suite_root.glob("*/run_contract.json"))
    by_name = {path.name: path for path in contract_cells}
    unknown = requested - set(by_name)
    if unknown:
        raise ValueError(f"unknown scientific cell(s): {sorted(unknown)}")
    cells = [by_name[name] for name in sorted(requested)] if requested else contract_cells
    if not cells:
        raise ValueError(f"no scientific run contracts under {suite_root}")
    return cells


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--tier", choices=("trend", "exact"), required=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--acknowledge-full-eval", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args(argv)
    try:
        if args.action == "run" and not args.yes:
            raise ValueError("action=run requires --yes")
        suite_root = args.suite_root.expanduser().resolve()
        cells = selected_cells(suite_root, set(args.cell))
        failures = 0
        for cell_root in cells:
            nested_action = "plan" if args.action == "plan" else args.action
            command = command_for_cell(
                repo_root,
                cell_root,
                args.tier,
                nested_action,
                acknowledge_full_eval=args.acknowledge_full_eval,
            )
            print(f"[{cell_root.name}/{args.tier}] " + " ".join(command), flush=True)
            if args.action == "plan":
                continue
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=os.environ.copy(),
                check=False,
            )
            if result.returncode != 0:
                failures += 1
                if not args.keep_going:
                    break
        return 1 if failures else 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
