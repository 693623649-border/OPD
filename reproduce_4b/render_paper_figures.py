#!/usr/bin/env python3
"""Render auditable local counterparts for paper Figures 1--23.

The renderer deliberately separates *an artifact can be visualized* from *the
artifact is paper-comparable*.  In particular, a completed smoke run may be
rendered as a pipeline diagnostic, but its protocol and non-comparable status
are written both into the image and ``render_manifest.json``.  Missing cells,
incomplete evaluations, ambiguous probes, and unsupported paper-authored
layouts are skipped rather than filled with paper numbers or proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import plot_metrics
import plot_position_entropy
import scientific_conclusions


FIGURE_IDS = tuple(f"figure-{number:02d}" for number in range(1, 24))
BENCHMARKS = ("AIME24", "AIME25", "AMC23")
EXPECTED_PROMPTS = {"AIME24": 30, "AIME25": 30, "AMC23": 83}
TRAINING_DYNAMIC_TYPES = {
    "opd_train_eval_plot",
    "opd_training_dynamics_plot",
    "metric_plot_reuse",
    "opd_train_eval_plot_original_scale",
}
BAR_ONLY_TYPES = {
    "opd_train_eval_bar_plot",
    "benchmark_breakdown_reuse",
}
STATIC_BLOCK_STATUSES = {
    "blocked",
    "blocked_on_local_hardware",
    "blocked_external",
    "blocked_hardware",
}
FIG19_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "policy_loss": (
        "actor/pg_loss",
        "actor/policy_loss",
        "policy_gradient/loss",
    ),
    "grad_norm": plot_metrics.METRIC_ALIASES["grad_norm"],
    "max_abs_adv_probability_difference": (
        "val-extrema/prob_diff_at_max_abs_adv_intersection",
    ),
}
FIG19_SCHEMA_KEY = "opd/figure19_metric_schema_version"
SCIENTIFIC_PROTOCOL = "scientific"
PAPER_TEXT_BOUNDARY_STEP = 200
FIGURE_WINDOW_FINAL_STEP = 260
POSITION_WINDOW_STEPS = tuple(range(180, 251, 10))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def safe_protocol(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized.lower() or "unknown"


def load_ledger(path: Path) -> dict[str, Mapping[str, Any]]:
    ledger = read_object(path)
    raw_entries = ledger.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("paper ledger must contain an entries list")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise ValueError("paper ledger contains an invalid entry")
        entry_id = str(raw["id"])
        if entry_id in entries:
            raise ValueError(f"paper ledger repeats {entry_id}")
        entries[entry_id] = raw
    missing = sorted(set(FIGURE_IDS) - set(entries))
    if missing:
        raise ValueError(f"paper ledger is missing figures: {missing}")
    return entries


@dataclass(frozen=True)
class EvaluationArtifact:
    source: Path
    label: str
    n: int
    checkpoint_step: int | None
    values: dict[str, float]
    summary: Mapping[str, Any]
    explicitly_paper_comparable: bool


@dataclass
class CellArtifact:
    cell_id: str
    label: str
    fidelity: str
    fingerprint: str
    environment: Mapping[str, Any]
    status: Mapping[str, Any] | None
    metrics_path: Path | None
    run: plot_metrics.RunMetrics | None
    complete: bool
    reason: str | None
    evaluations: list[EvaluationArtifact] = field(default_factory=list)


@dataclass
class SuiteArtifact:
    root: Path
    suite_id: str
    protocol: str
    seed: Any
    source_tree_sha256: Any
    cells: dict[str, CellArtifact]


@dataclass(frozen=True)
class ProbeArtifact:
    kind: str
    protocol: str
    source: Path
    summary: Mapping[str, Any]


def _metrics_path(root: Path, cell_id: str, status: Mapping[str, Any]) -> Path | None:
    # Scientific resume segments are merged into this canonical path.  It must
    # take precedence over a last-attempt log, which may begin after step 1.
    candidates: list[Path] = [root / cell_id / "metrics.jsonl"]
    raw_metrics = status.get("metrics_file")
    if isinstance(raw_metrics, str) and raw_metrics:
        candidates.append(Path(raw_metrics).expanduser())
    raw_run = status.get("run_dir")
    if isinstance(raw_run, str) and raw_run:
        candidates.append(Path(raw_run).expanduser() / "metrics.jsonl")
    attempt = status.get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0:
        candidates.append(root / cell_id / f"attempt-{attempt:04d}" / "metrics.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_evaluation(path: Path, label: str) -> EvaluationArtifact:
    payload = read_object(path)
    n = payload.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"{path}: evaluation n must be a positive integer")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(BENCHMARKS):
        raise ValueError(f"{path}: expected exactly the three paper benchmarks")
    metric_key = f"avg@{n}"
    values: dict[str, float] = {}
    for benchmark in BENCHMARKS:
        metrics = benchmarks[benchmark]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{path}: malformed {benchmark} metrics")
        value = finite_number(metrics.get(metric_key))
        if value is None:
            raise ValueError(f"{path}: missing finite {benchmark}/{metric_key}")
        values[benchmark] = value
    step = payload.get("checkpoint_step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step <= 0):
        raise ValueError(f"{path}: invalid checkpoint_step")
    return EvaluationArtifact(
        source=path.resolve(),
        label=label,
        n=n,
        checkpoint_step=step,
        values=values,
        summary=payload,
        explicitly_paper_comparable=payload.get("paper_comparable") is True,
    )


def _cell_evaluations(run_dir: Path | None, label: str) -> list[EvaluationArtifact]:
    if run_dir is None or not run_dir.is_dir():
        return []
    evaluations: list[EvaluationArtifact] = []
    seen: set[tuple[int | None, int]] = set()
    for path in sorted((run_dir / "evaluation").glob("global_step_*/summary.json")):
        item = _read_evaluation(path, label)
        key = (item.checkpoint_step, item.n)
        if key in seen:
            raise ValueError(f"{run_dir}: duplicate evaluation step/n {key}")
        seen.add(key)
        evaluations.append(item)
    return evaluations


def _last_run_step(run: plot_metrics.RunMetrics) -> int | None:
    steps = [int(point.step) for series in run.series.values() for point in series]
    return max(steps) if steps else None


def load_suite(path: Path) -> SuiteArtifact:
    root = path.expanduser().resolve()
    manifest = read_object(root / "suite_manifest.json")
    protocol = manifest.get("protocol")
    suite_id = manifest.get("suite_id")
    raw_cells = manifest.get("cells")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError(f"{root}: suite manifest has no protocol")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError(f"{root}: suite manifest has no suite_id")
    if not isinstance(raw_cells, list):
        raise ValueError(f"{root}: suite manifest has no cells list")
    cells: dict[str, CellArtifact] = {}
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping) or not isinstance(raw_cell.get("cell_id"), str):
            raise ValueError(f"{root}: suite manifest contains an invalid cell")
        cell_id = str(raw_cell["cell_id"])
        if cell_id in cells:
            raise ValueError(f"{root}: suite manifest repeats {cell_id}")
        fingerprint = raw_cell.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"{root}: {cell_id} has no fingerprint")
        label = str(raw_cell.get("label", cell_id))
        fidelity = str(raw_cell.get("fidelity", "unspecified"))
        environment = raw_cell.get("environment")
        if not isinstance(environment, Mapping):
            environment = {}
        status_path = root / cell_id / "status.json"
        status: Mapping[str, Any] | None = None
        metrics_path: Path | None = None
        run: plot_metrics.RunMetrics | None = None
        evaluations: list[EvaluationArtifact] = []
        complete = False
        reason: str | None = "status.json is missing"
        if status_path.is_file():
            status = read_object(status_path)
            if status.get("fingerprint") != fingerprint:
                reason = "status/manifest fingerprint mismatch"
            elif status.get("state") != "completed" and status.get("execution_state") not in {
                "training_complete",
                "evaluation_complete",
                "probe_complete",
                "rendered",
                "scientific_result_available",
            }:
                reason = (
                    "training execution state is "
                    f"{status.get('execution_state', status.get('state', 'unknown'))!r}, not complete"
                )
            else:
                metrics_path = _metrics_path(root, cell_id, status)
                if metrics_path is None:
                    reason = "completed status has no readable metrics.jsonl"
                else:
                    try:
                        run = plot_metrics.read_file_logger(metrics_path, label=label)
                        last_step = _last_run_step(run)
                        explicit_budget = environment.get("TOTAL_TRAINING_STEPS")
                        expected_step: int | None = None
                        if isinstance(explicit_budget, str) and explicit_budget.isdigit():
                            expected_step = int(explicit_budget)
                        elif isinstance(explicit_budget, int) and not isinstance(explicit_budget, bool):
                            expected_step = explicit_budget
                        if last_step is None:
                            reason = "metrics contain no recognized finite plotting series"
                        elif expected_step is not None and last_step < expected_step:
                            reason = f"metrics end at step {last_step}, before declared budget {expected_step}"
                        else:
                            raw_run_dir = status.get("run_dir")
                            run_dir = Path(raw_run_dir).expanduser() if isinstance(raw_run_dir, str) else metrics_path.parent
                            evaluations = _cell_evaluations(run_dir, label)
                            canonical_cell_root = root / cell_id
                            if canonical_cell_root.resolve() != run_dir.resolve():
                                canonical = _cell_evaluations(canonical_cell_root, label)
                                by_key = {
                                    (item.checkpoint_step, item.n): item for item in evaluations
                                }
                                for item in canonical:
                                    key = (item.checkpoint_step, item.n)
                                    if key in by_key:
                                        raise ValueError(
                                            f"{root / cell_id}: duplicate canonical evaluation step/n {key}"
                                        )
                                    by_key[key] = item
                                evaluations = list(by_key.values())
                            complete = True
                            reason = None
                    except (json.JSONDecodeError, OSError, ValueError) as exc:
                        reason = f"invalid metrics/evaluation artifact: {exc}"
        cells[cell_id] = CellArtifact(
            cell_id=cell_id,
            label=label,
            fidelity=fidelity,
            fingerprint=fingerprint,
            environment=dict(environment),
            status=status,
            metrics_path=metrics_path,
            run=run,
            complete=complete,
            reason=reason,
            evaluations=evaluations,
        )
    return SuiteArtifact(
        root=root,
        suite_id=suite_id,
        protocol=protocol,
        seed=manifest.get("seed"),
        source_tree_sha256=manifest.get("source_tree_sha256"),
        cells=cells,
    )


def _expand_suite_roots(paths: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_file() and path.name == "suite_manifest.json":
            discovered.append(path.parent)
        elif (path / "suite_manifest.json").is_file():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(item.parent for item in path.rglob("suite_manifest.json"))
        else:
            raise FileNotFoundError(f"suite root does not exist: {path}")
    unique = sorted(set(discovered))
    if paths and not unique:
        raise FileNotFoundError("no suite_manifest.json found under requested suite roots")
    return unique


def discover_model_evaluations(paths: Sequence[Path]) -> list[tuple[str, EvaluationArtifact]]:
    discovered: list[tuple[str, EvaluationArtifact]] = []
    seen: set[Path] = set()
    for raw in paths:
        root = raw.expanduser().resolve()
        candidates = [root / "summary.json"] if (root / "summary.json").is_file() else list(root.rglob("summary.json"))
        for path in sorted(candidates):
            if path in seen:
                continue
            try:
                summary = read_object(path)
                if "model" not in summary or "benchmarks" not in summary:
                    continue
                revision = summary.get("revision")
                model = summary.get("model")
                if not isinstance(model, str) or not isinstance(revision, str):
                    continue
                label = model.rsplit("/", 1)[-1]
                discovered.append((f"{model}@{revision}", _read_evaluation(path, label)))
                seen.add(path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
    return discovered


def infer_external_protocol(evaluation: EvaluationArtifact) -> str:
    if evaluation.explicitly_paper_comparable and evaluation.n == 16:
        return "paper"
    target = str(evaluation.summary.get("target_id", "")).lower()
    for protocol in ("smoke", "calibration", "pilot", "paper"):
        if protocol in target:
            return protocol
    return "noncomparable-eval"


def discover_probes(paths: Sequence[Path]) -> list[ProbeArtifact]:
    names = {
        "teacher_continuation_summary.json": "teacher_continuation",
        "sequence_reward_summary.json": "sequence_reward",
    }
    probes: list[ProbeArtifact] = []
    seen: set[Path] = set()
    for raw in paths:
        root = raw.expanduser().resolve()
        candidates: list[Path] = []
        if root.is_file() and root.name in names:
            candidates.append(root)
        elif root.is_dir():
            for name in names:
                candidates.extend(root.rglob(name))
        else:
            raise FileNotFoundError(f"probe artifact path does not exist: {root}")
        for path in sorted(candidates):
            if path in seen:
                continue
            summary = read_object(path)
            kind = names[path.name]
            target = " ".join(path.parts).lower()
            declared_protocol = summary.get("protocol")
            if isinstance(declared_protocol, str) and declared_protocol:
                protocol = declared_protocol
            else:
                protocol = next(
                    (
                        name
                        for name in ("scientific", "smoke", "calibration", "pilot", "paper")
                        if name in target
                    ),
                    "local-probe",
                )
            probes.append(ProbeArtifact(kind, protocol, path.resolve(), summary))
            seen.add(path)
    return probes


def _choose_suite(
    suites: Sequence[SuiteArtifact],
    protocol: str,
    required_cells: Sequence[str],
    preferred_suite_id: str | None,
) -> tuple[SuiteArtifact | None, list[str], list[str]]:
    candidates = [suite for suite in suites if suite.protocol == protocol]
    if not candidates:
        return None, [f"{cell_id}: no suite for protocol={protocol}" for cell_id in required_cells], []

    def score(suite: SuiteArtifact) -> tuple[int, int, int, str]:
        complete = sum(
            cell_id in suite.cells and suite.cells[cell_id].complete for cell_id in required_cells
        )
        present = sum(cell_id in suite.cells for cell_id in required_cells)
        preferred = int(suite.suite_id == preferred_suite_id)
        return (complete, present, preferred, str(suite.root))

    ranked = sorted(candidates, key=score, reverse=True)
    chosen = ranked[0]
    missing: list[str] = []
    for cell_id in required_cells:
        cell = chosen.cells.get(cell_id)
        if cell is None:
            missing.append(f"{cell_id}: absent from {chosen.root}")
        elif not cell.complete:
            missing.append(f"{cell_id}: {cell.reason}")
    equally_complete = [
        suite
        for suite in ranked[1:]
        if not missing
        and all(cell_id in suite.cells and suite.cells[cell_id].complete for cell_id in required_cells)
        and score(suite)[:3] == score(chosen)[:3]
    ]
    notes = []
    if equally_complete:
        notes.append(
            "multiple complete coherent suites exist; selected deterministically without mixing cells: "
            + str(chosen.root)
        )
    return chosen, missing, notes


def _common_latest_evaluations(
    cells: Sequence[CellArtifact],
) -> tuple[dict[str, EvaluationArtifact] | None, int | None, str | None]:
    available: list[dict[int, EvaluationArtifact]] = []
    for cell in cells:
        by_n: dict[int, EvaluationArtifact] = {}
        for item in cell.evaluations:
            previous = by_n.get(item.n)
            previous_step = previous.checkpoint_step or -1 if previous else -1
            item_step = item.checkpoint_step or -1
            if previous is None or item_step > previous_step:
                by_n[item.n] = item
        available.append(by_n)
    if not available:
        return None, None, "no cells"
    common = set(available[0])
    for mapping in available[1:]:
        common &= set(mapping)
    if not common:
        counts = ", ".join(f"{cell.cell_id}: n={sorted(mapping)}" for cell, mapping in zip(cells, available))
        return None, None, "no common complete three-benchmark evaluation (" + counts + ")"
    n = 16 if 16 in common else max(common)
    return {cell.cell_id: mapping[n] for cell, mapping in zip(cells, available)}, n, None


def _sampling_matches_paper(path: Path, n: int) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            found = False
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                sampling = payload.get("sampling")
                if not isinstance(sampling, Mapping):
                    return False
                expected = {
                    "n": n,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 31_744,
                    "thinking": "off",
                }
                for key, value in expected.items():
                    actual = sampling.get(key)
                    if isinstance(value, float):
                        if finite_number(actual) is None or not math.isclose(float(actual), value):
                            return False
                    elif actual != value:
                        return False
                found = True
            return found
    except (json.JSONDecodeError, OSError):
        return False


def evaluation_is_paper_comparable(item: EvaluationArtifact, protocol: str) -> bool:
    if item.n != 16 or protocol != "paper":
        return False
    if item.explicitly_paper_comparable:
        return True
    benchmarks = item.summary.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        return False
    for benchmark in BENCHMARKS:
        metrics = benchmarks.get(benchmark)
        if not isinstance(metrics, Mapping) or metrics.get("num_prompts") != EXPECTED_PROMPTS[benchmark]:
            return False
        raw_path = metrics.get("input_jsonl")
        if not isinstance(raw_path, str) or not _sampling_matches_paper(Path(raw_path), 16):
            return False
    return True


def static_fidelity_veto(entry: Mapping[str, Any]) -> str | None:
    """Return only immutable disclosure/hardware vetoes, never run-progress gaps.

    ``ledger.blocker`` also records transient observations such as "only a
    two-step smoke has run".  Using that free text as a permanent gate would
    make a later complete artifact impossible to promote.  Static blockers are
    therefore restricted to explicit blocked statuses and reconstruction or
    unpublished-checkpoint fidelity labels.
    """

    status = str(entry.get("status", "")).strip().lower()
    fidelity = str(entry.get("fidelity", "")).strip().lower()
    if status in STATIC_BLOCK_STATUSES:
        return f"ledger status={status} is a static fidelity veto"
    if "reconstruction" in status or "reconstruction" in fidelity:
        return "paper identity requires a reconstruction not disclosed by the authors"
    unpublished_markers = (
        "unpublished",
        "weights are unavailable",
        "missing_author",
        "missing author",
    )
    if any(marker in fidelity for marker in unpublished_markers):
        return "an exact author checkpoint/artifact is unpublished"
    return None


def position_entropy_evidence(
    entry: Mapping[str, Any],
    protocol: str,
    rows: Sequence[plot_position_entropy.PositionEntropyRow],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the incompatible text-budget and plotted-window evidence.

    The paper text specifies 200 steps, whereas Figures 13/23 reach roughly
    250--260.  The 200-step branch can authenticate the disclosed condition;
    the extension can only be an explicitly labelled reconstruction.  Neither
    is allowed to upgrade the complete original figure to paper-comparable.
    """

    def integer_setting(name: str) -> int | None:
        raw = environment.get(name)
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
        return None

    result: dict[str, Any] = {
        "text_condition_complete": False,
        "figure_window_complete": False,
        "paper_comparable": False,
        "conclusion_state": "not_assessed",
        "label": "incomplete",
    }
    if static_fidelity_veto(entry) is not None:
        result["label"] = "static-fidelity-veto"
        return result
    common_settings = {
        "POSITION_ENTROPY_START_STEP": 180,
        "POSITION_ENTROPY_LOG_FREQ": 10,
        "POSITION_ENTROPY_BIN_SIZE": 256,
        "MAX_RESPONSE_LENGTH": 15_360,
    }
    if any(integer_setting(name) != value for name, value in common_settings.items()):
        return result
    observed_steps = {row.step for row in rows}
    expected_bins = 15_360 // 256
    for row in rows:
        if row.bin_size != 256 or len(row.student) != expected_bins:
            return result
        if len(row.teacher) != expected_bins or len(row.counts) != expected_bins:
            return result
        if any(count < 0 or not float(count).is_integer() for count in row.counts):
            return result
        if not any(count > 0 for count in row.counts):
            return result

    total_steps = integer_setting("TOTAL_TRAINING_STEPS")
    if protocol == "paper" and total_steps == 200 and observed_steps == {180, 190, 200}:
        result.update(text_condition_complete=True, label="text-aligned-only")
        return result
    if total_steps == 260 and observed_steps == set(range(180, 261, 10)):
        result.update(
            text_condition_complete=True,
            figure_window_complete=True,
            label="text-boundary-plus-undisclosed-figure-window-reconstruction",
        )
    return result


def position_entropy_is_paper_comparable(
    entry: Mapping[str, Any],
    protocol: str,
    rows: Sequence[plot_position_entropy.PositionEntropyRow],
    environment: Mapping[str, Any],
) -> bool:
    """Compatibility wrapper; the complete original heatmap remains unverifiable."""

    return bool(position_entropy_evidence(entry, protocol, rows, environment)["paper_comparable"])


def _annotation(
    figure_id: str,
    protocol: str,
    fidelity: str,
    paper_comparable: bool,
    detail: str | None = None,
) -> str:
    claim = "PAPER-COMPARABLE" if paper_comparable else "NOT PAPER-COMPARABLE"
    line = f"{figure_id} | protocol={protocol} | {claim}"
    fidelity_line = textwrap.fill(f"fidelity: {fidelity}", width=115)
    return "\n".join(part for part in (line, fidelity_line, detail) if part)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)


def compact_condition_labels(cells: Sequence[CellArtifact], max_length: int = 44) -> list[str]:
    """Return readable, unique labels that cannot dominate six-panel legends."""

    labels: list[str] = []
    for cell in cells:
        label = re.sub(r"\s+", " ", cell.label).strip()
        label = re.sub(r"\s+student to its\s+", " → ", label, flags=re.IGNORECASE)
        label = re.sub(r"\s+student to\s+", " → ", label, flags=re.IGNORECASE)
        label = re.sub(r"\s+pre-RL checkpoint", " pre-RL", label, flags=re.IGNORECASE)
        label = re.sub(r",?\s+\d+\s+steps?\b", "", label, flags=re.IGNORECASE)
        label = re.sub(
            r"(?:\(author raw mass\)|,\s*author raw mass)",
            " (raw mass)",
            label,
            flags=re.IGNORECASE,
        )
        if len(label) > max_length:
            label = textwrap.shorten(label, width=max_length, placeholder="…")
        labels.append(label or cell.cell_id)
    if len(set(labels)) != len(labels):
        # Fingerprinted cell ids are the only unambiguous safe fallback when
        # prose labels collapse after compaction.
        labels = [cell.cell_id for cell in cells]
    return labels


def _annotate_figure(figure: Any, title: str) -> None:
    """Place the comparability claim and detail with explicit top padding."""

    lines = title.splitlines()
    claim = lines[0] if lines else title
    details = "\n".join(lines[1:])
    not_comparable = "NOT PAPER-COMPARABLE" in claim
    figure.suptitle(
        claim,
        x=0.5,
        y=0.992,
        fontsize=12,
        fontweight="bold",
        color="#8b1a1a" if not_comparable else "#1b5e20",
    )
    if details:
        figure.text(
            0.5,
            0.958,
            details,
            ha="center",
            va="top",
            fontsize=9.2,
            linespacing=1.25,
            wrap=True,
        )


def render_dynamics(
    cells: Sequence[CellArtifact], output: Path, title: str, overwrite: bool
) -> None:
    _prepare_output(output, overwrite)
    runs = [cell.run for cell in cells if cell.run is not None]
    if len(runs) != len(cells):
        raise ValueError("training group contains a cell without readable scalar metrics")
    plot_metrics.validate_metric_schemas(runs)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("training-dynamics rendering requires matplotlib") from exc

    compact = compact_condition_labels(cells)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    run_colors = {id(run): colors[index % len(colors)] for index, run in enumerate(runs)}
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.4))
    panels = (
        (axes[0, 0], "overlap_ratio", "Top-k overlap ratio", "ratio"),
        (axes[0, 1], "overlap_advantage", "Overlap-token advantage", "advantage"),
        (axes[1, 1], "grad_norm", "Gradient norm", "norm"),
        (axes[1, 2], "response_length", "Mean response length", "tokens"),
    )
    for axis, metric_name, panel_title, ylabel in panels:
        plotted = False
        for run in runs:
            points = run.series[metric_name]
            if not points:
                continue
            axis.plot(
                [point.step for point in points],
                [point.value for point in points],
                color=run_colors[id(run)],
            )
            plotted = True
        axis.set_title(panel_title)
        axis.set_xlabel("step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if not plotted:
            axis.text(0.5, 0.5, "metric not found", ha="center", va="center", transform=axis.transAxes)

    entropy_axis = axes[1, 0]
    entropy_plotted = False
    for run in runs:
        color = run_colors[id(run)]
        for points, linestyle, linewidth in (
            (run.series["student_entropy"], ":", 1.5),
            (run.series["teacher_entropy"], "--", 1.5),
            (plot_metrics.entropy_gap(run), "-", 2.2),
        ):
            if points:
                entropy_axis.plot(
                    [point.step for point in points],
                    [point.value for point in points],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
                entropy_plotted = True
    entropy_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    entropy_axis.set_title("Student/teacher entropy gap")
    entropy_axis.set_xlabel("step")
    entropy_axis.set_ylabel("entropy")
    entropy_axis.grid(alpha=0.25)
    if entropy_plotted:
        entropy_axis.legend(
            handles=[
                Line2D([], [], color="black", linestyle=":", label="student entropy"),
                Line2D([], [], color="black", linestyle="--", label="teacher entropy"),
                Line2D([], [], color="black", linewidth=2.2, label="absolute gap (Eq. 8)"),
            ],
            fontsize=7.5,
            loc="best",
        )
    else:
        entropy_axis.text(0.5, 0.5, "metrics not found", ha="center", va="center", transform=entropy_axis.transAxes)

    mass_axis = axes[0, 2]
    mass_plotted = False
    for run in runs:
        color = run_colors[id(run)]
        for metric_name, linestyle in (
            ("student_overlap_mass", "-"),
            ("teacher_overlap_mass", "--"),
        ):
            points = run.series[metric_name]
            if points:
                mass_axis.plot(
                    [point.step for point in points],
                    [point.value for point in points],
                    color=color,
                    linestyle=linestyle,
                )
                mass_plotted = True
    mass_axis.set_title("Probability mass on Top-k overlap")
    mass_axis.set_xlabel("step")
    mass_axis.set_ylabel("probability mass")
    mass_axis.grid(alpha=0.25)
    if mass_plotted:
        mass_axis.legend(
            handles=[
                Line2D([], [], color="black", linestyle="-", label="student"),
                Line2D([], [], color="black", linestyle="--", label="teacher"),
            ],
            fontsize=7.5,
            loc="best",
        )
    else:
        mass_axis.text(0.5, 0.5, "metrics not found", ha="center", va="center", transform=mass_axis.transAxes)

    condition_handles = [
        Line2D([], [], color=run_colors[id(run)], linewidth=2.2, label=label)
        for run, label in zip(runs, compact)
    ]
    figure.legend(
        handles=condition_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(4, len(condition_handles)),
        fontsize=8.5,
        title="conditions",
        title_fontsize=8.5,
        frameon=True,
    )
    _annotate_figure(figure, title)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.115, top=0.845, wspace=0.29, hspace=0.38)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)


def render_grouped_bar(
    evaluations: Mapping[str, EvaluationArtifact],
    labels: Mapping[str, str],
    output: Path,
    title: str,
    overwrite: bool,
) -> None:
    _prepare_output(output, overwrite)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("grouped-bar rendering requires matplotlib and numpy") from exc
    keys = list(evaluations)
    if not keys:
        raise ValueError("cannot render an empty evaluation group")
    n_values = {evaluations[key].n for key in keys}
    if len(n_values) != 1:
        raise ValueError(f"evaluation group mixes n values: {sorted(n_values)}")
    n = next(iter(n_values))
    positions = np.arange(len(BENCHMARKS), dtype=float)
    width = min(0.24, 0.78 / len(keys))
    figure, axis = plt.subplots(figsize=(max(8.0, 1.25 * len(keys) + 5.5), 5.5))
    center = (len(keys) - 1) / 2
    for index, key in enumerate(keys):
        item = evaluations[key]
        values = [item.values[benchmark] for benchmark in BENCHMARKS]
        bars = axis.bar(positions + (index - center) * width, values, width, label=labels[key])
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(positions, BENCHMARKS)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel(f"avg@{n}")
    axis.set_title(title, fontsize=10)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def compose_images(
    inputs: Sequence[Path], output: Path, title: str, overwrite: bool
) -> None:
    _prepare_output(output, overwrite)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("composite rendering requires matplotlib") from exc
    figure, axes = plt.subplots(len(inputs), 1, figsize=(18, 8.5 * len(inputs)), squeeze=False)
    for axis, path in zip(axes[:, 0], inputs):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_training_and_evaluation(
    cells: Sequence[CellArtifact],
    evaluations: Mapping[str, EvaluationArtifact],
    output: Path,
    title: str,
    overwrite: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="opd-paper-render-") as directory:
        temp = Path(directory)
        dynamics = temp / "dynamics.png"
        bars = temp / "benchmarks.png"
        render_dynamics(cells, dynamics, title, overwrite=True)
        render_grouped_bar(
            evaluations,
            {cell.cell_id: cell.label for cell in cells},
            bars,
            title,
            overwrite=True,
        )
        compose_images((dynamics, bars), output, title, overwrite)


def read_position_entropy_strict(
    path: Path,
) -> list[plot_position_entropy.PositionEntropyRow]:
    """Read position entropy while requiring an exact integer schema marker.

    Python considers ``True == 1`` and ``1.0 == 1``.  Those values are not
    valid schema provenance, so the renderer authenticates the raw JSON before
    delegating array validation to ``plot_position_entropy``.
    """

    required = {
        "position-entropy/bin_size",
        "position-entropy/student_mean_by_bin",
        "position-entropy/teacher_mean_by_bin",
        "position-entropy/token_count_by_bin",
    }
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: metric row must be an object")
            data = payload.get("data", payload)
            if not isinstance(data, Mapping):
                raise ValueError(f"{path}:{line_number}: metric data must be an object")
            present = required.intersection(data)
            if not present:
                continue
            missing = sorted(required - set(data))
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: incomplete position-entropy producer keys: {missing}"
                )
            schema = data.get("opd/position_entropy_schema_version")
            if type(schema) is not int or schema != 1:
                raise ValueError(
                    f"{path}:{line_number}: position entropy requires exact integer "
                    "opd/position_entropy_schema_version=1"
                )
    try:
        return plot_position_entropy.read_position_entropy(path)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: malformed position-entropy producer row: {exc}") from exc


def render_position_entropy_rows(
    rows: Sequence[plot_position_entropy.PositionEntropyRow],
    output: Path,
    metric: str,
    title: str,
    overwrite: bool,
) -> None:
    _prepare_output(output, overwrite)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("position-entropy rendering requires matplotlib and numpy") from exc
    if not rows:
        raise ValueError("cannot render empty position-entropy rows")
    metrics = ("student", "teacher") if metric == "both" else (metric,)
    if any(name not in {"student", "teacher"} for name in metrics):
        raise ValueError(f"unsupported position-entropy metric {metric!r}")
    max_bins = max(len(getattr(row, name)) for row in rows for name in metrics)
    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(13, 5.3 * len(metrics)),
        squeeze=False,
    )
    for axis, name in zip(axes[:, 0], metrics):
        matrix = np.full((len(rows), max_bins), np.nan, dtype=float)
        for row_index, row in enumerate(rows):
            values = getattr(row, name)
            valid = [
                value if row.counts[index] > 0 else np.nan
                for index, value in enumerate(values)
            ]
            matrix[row_index, : len(valid)] = valid
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", origin="upper")
        axis.set_yticks(range(len(rows)), [str(row.step) for row in rows])
        bin_size = rows[0].bin_size
        tick_bins = sorted(
            set([0, max_bins - 1, *range(0, max_bins, max(1, 1024 // bin_size))])
        )
        axis.set_xticks(tick_bins, [f"{index * bin_size / 1024:g}K" for index in tick_bins])
        axis.set_xlabel("output position")
        axis.set_ylabel("training step")
        axis.set_title(f"{name.capitalize()} entropy by output position", fontsize=11)
        late_indices = [index for index, row in enumerate(rows) if row.step > 200]
        if late_indices:
            first_late = min(late_indices)
            axis.axhline(first_late - 0.5, color="white", linewidth=1.5, linestyle="--")
            axis.axhspan(
                first_late - 0.5,
                len(rows) - 0.5,
                color="white",
                alpha=0.08,
                label="undisclosed >200-step reconstruction",
            )
            axis.legend(loc="upper right", fontsize=8)
        figure.colorbar(image, ax=axis, label="entropy", pad=0.025)
    _annotate_figure(figure, title)
    figure.subplots_adjust(
        left=0.075,
        right=0.92,
        bottom=0.13,
        top=0.79 if len(title.splitlines()) >= 3 else 0.84,
        hspace=0.42,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.24)
    plt.close(figure)


def _validate_continuation_summary(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = summary.get("prefix_results")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("continuation summary lacks the four prefix results")
    expected = (1024, 4096, 8192, 16384)
    observed: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("continuation prefix result is malformed")
        prefix = row.get("prefix_length")
        gain = finite_number(row.get("accuracy_gain"))
        if not isinstance(prefix, int) or gain is None:
            raise ValueError("continuation prefix result lacks prefix_length/accuracy_gain")
        observed.append(prefix)
    if tuple(observed) != expected:
        raise ValueError(f"continuation prefixes are {observed}, expected {list(expected)}")
    return rows


def render_continuation(
    summary: Mapping[str, Any], output: Path, title: str, overwrite: bool
) -> None:
    _prepare_output(output, overwrite)
    rows = _validate_continuation_summary(summary)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("continuation rendering requires matplotlib") from exc
    gains = [float(row["accuracy_gain"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(range(4), gains, marker="o", linewidth=2)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xticks(range(4), ["1K", "4K", "8K", "16K"])
    axis.set_ylabel("teacher accuracy - student accuracy")
    axis.set_xlabel("student prefix length")
    axis.set_title(title, fontsize=9)
    axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_sequence_reward(
    summary: Mapping[str, Any], output: Path, title: str, overwrite: bool
) -> None:
    _prepare_output(output, overwrite)
    correct = summary.get("correct")
    incorrect = summary.get("incorrect")
    auroc = finite_number(summary.get("auroc"))
    if not isinstance(correct, Mapping) or not isinstance(incorrect, Mapping) or auroc is None:
        raise ValueError("sequence-reward summary lacks distributions/AUROC")
    means = [finite_number(correct.get("mean")), finite_number(incorrect.get("mean"))]
    stds = [finite_number(correct.get("std_population")), finite_number(incorrect.get("std_population"))]
    counts = [correct.get("n"), incorrect.get("n")]
    if any(value is None for value in means + stds) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts
    ):
        raise ValueError("sequence-reward distribution statistics are incomplete")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sequence-reward rendering requires matplotlib") from exc
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = [f"correct (N={counts[0]})", f"incorrect (N={counts[1]})"]
    bars = axis.bar(labels, means, yerr=stds, capsize=6, alpha=0.75)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("sequence mean reward (mean ± population std)")
    axis.set_title(title + f"\nAUROC={auroc:.4f}", fontsize=9)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _evaluations_at_step(
    cells: Sequence[CellArtifact], *, step: int, n: int = 16
) -> dict[str, EvaluationArtifact]:
    selected: dict[str, EvaluationArtifact] = {}
    for cell in cells:
        matches = [
            item for item in cell.evaluations if item.checkpoint_step == step and item.n == n
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{cell.cell_id}: expected exactly one avg@{n} evaluation at step {step}, "
                f"found {len(matches)}"
            )
        selected[cell.cell_id] = matches[0]
    return selected


def _aggregate_evaluation(item: EvaluationArtifact) -> float:
    return sum(item.values[benchmark] for benchmark in BENCHMARKS) / len(BENCHMARKS)


def _require_metric_window(
    cell: CellArtifact, metric: str, *, final_step: int = FIGURE_WINDOW_FINAL_STEP
) -> list[plot_metrics.Point]:
    if cell.run is None:
        raise ValueError(f"{cell.cell_id}: no readable metrics")
    points = cell.run.series[metric]
    by_step = {int(point.step): point for point in points if float(point.step).is_integer()}
    expected = set(range(1, final_step + 1))
    missing = sorted(expected - set(by_step))
    if missing:
        raise ValueError(
            f"{cell.cell_id}: {metric} lacks {len(missing)} registered steps; first={missing[:8]}"
        )
    return [by_step[step] for step in range(1, final_step + 1)]


def _overlap_drop(cell: CellArtifact) -> float:
    points = _require_metric_window(cell, "overlap_ratio")
    values = {int(point.step): point.value for point in points}
    before = sum(values[step] for step in range(180, 201)) / 21
    after = sum(values[step] for step in range(240, 261)) / 21
    return before - after


def _condition_key(cell: CellArtifact, environment_key: str) -> str:
    raw = cell.environment.get(environment_key)
    if raw is None:
        raise ValueError(f"{cell.cell_id}: missing {environment_key}")
    return str(raw)


def render_scientific_figure11(
    cells: Sequence[CellArtifact],
    evaluations: Mapping[str, EvaluationArtifact],
    continuation_summary: Mapping[str, Any],
    output: Path,
    title: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Render the six-length endpoint and exact continuation probe together."""

    _prepare_output(output, overwrite)
    continuation_assessment = scientific_conclusions.assess_continuation(
        continuation_summary
    )
    labels = [cell.label for cell in cells]
    scores = [_aggregate_evaluation(evaluations[cell.cell_id]) for cell in cells]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Figure 11 rendering requires matplotlib") from exc
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.3))
    bars = axes[0].bar(labels, scores, color="#4C78A8")
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_ylim(0.0, max(1.0, max(scores) * 1.15))
    axes[0].set_ylabel("mean avg@16 across AIME24/AIME25/AMC23")
    axes[0].set_xlabel("maximum response length")
    axes[0].set_title("(a) Response-length endpoint at step 200")
    axes[0].grid(axis="y", alpha=0.2)
    rows = continuation_summary.get("prefix_results")
    if continuation_assessment["underpowered"]:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.55,
            "INCONCLUSIVE\n"
            f"{continuation_assessment['num_selected_rollouts']} strict >16K rollouts\n"
            "pre-registered minimum: 30; no resampling",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        checked = _validate_continuation_summary(continuation_summary)
        gains = [float(row["accuracy_gain"]) for row in checked]
        ci = continuation_summary["bootstrap"]["prefix_gain_ci"]
        lower = [gains[i] - float(ci[str(row["prefix_length"])]["lower"]) for i, row in enumerate(checked)]
        upper = [float(ci[str(row["prefix_length"])]["upper"]) - gains[i] for i, row in enumerate(checked)]
        axes[1].errorbar(range(4), gains, yerr=[lower, upper], marker="o", capsize=4)
        axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[1].set_xticks(range(4), ["1K", "4K", "8K", "16K"])
        axes[1].set_ylabel("teacher accuracy - student accuracy")
        axes[1].set_xlabel("exact student token prefix")
        axes[1].set_title("(b) Teacher continuation (95% paired bootstrap CI)")
        axes[1].grid(alpha=0.2)
    _annotate_figure(figure, title)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.79, wspace=0.28)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)
    return continuation_assessment


def render_scientific_dynamics(
    cells: Sequence[CellArtifact],
    output: Path,
    title: str,
    overwrite: bool,
    *,
    figure_kind: str,
) -> dict[str, Any]:
    """Render the exact three panels used for Figures 12 and 16."""

    _prepare_output(output, overwrite)
    metrics = ("overlap_ratio", "student_entropy", "grad_norm")
    loaded = {
        cell.cell_id: {
            metric: _require_metric_window(cell, metric) for metric in metrics
        }
        for cell in cells
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long-horizon dynamics rendering requires matplotlib") from exc
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    panels = (
        ("overlap_ratio", "Top-k overlap", "ratio"),
        ("student_entropy", "Student entropy", "entropy"),
        ("grad_norm", "Gradient norm", "norm"),
    )
    for axis, (metric, panel_title, ylabel) in zip(axes, panels):
        for cell in cells:
            points = loaded[cell.cell_id][metric]
            axis.plot([point.step for point in points], [point.value for point in points], label=cell.label)
        axis.axvline(PAPER_TEXT_BOUNDARY_STEP, color="black", linestyle="--", linewidth=1.0)
        axis.axvspan(PAPER_TEXT_BOUNDARY_STEP, FIGURE_WINDOW_FINAL_STEP, color="#999999", alpha=0.08)
        axis.set_xlim(1, FIGURE_WINDOW_FINAL_STEP)
        axis.set_xlabel("training step")
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title)
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=8, loc="best")
    _annotate_figure(figure, title)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.13, top=0.79, wspace=0.27)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(figure)
    drops = {cell.label: _overlap_drop(cell) for cell in cells}
    return {
        "figure_kind": figure_kind,
        "overlap_drop_180_200_to_240_260": drops,
        "paper_text_boundary": PAPER_TEXT_BOUNDARY_STEP,
        "figure_window_final_step": FIGURE_WINDOW_FINAL_STEP,
    }


def render_scientific_endpoint_bars(
    cells: Sequence[CellArtifact],
    evaluations: Mapping[str, EvaluationArtifact],
    output: Path,
    title: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Render Figure 15 as the five pre-registered aggregate endpoints."""

    _prepare_output(output, overwrite)
    scores = {cell.label: _aggregate_evaluation(evaluations[cell.cell_id]) for cell in cells}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Figure 15 rendering requires matplotlib") from exc
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    bars = axis.bar(list(scores), list(scores.values()), color="#59A14F")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylim(0.0, max(1.0, max(scores.values()) * 1.15))
    axis.set_ylabel("mean avg@16 across AIME24/AIME25/AMC23")
    axis.set_xlabel("student support")
    axis.set_title(title, fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    normalized = {}
    for cell in cells:
        key = _condition_key(cell, "TOP_K")
        normalized["sampled" if key == "0" else key] = scores[cell.label]
    return scientific_conclusions.assess_topk(normalized)


def render_scientific_dual_reward(
    summary: Mapping[str, Any], output: Path, title: str, overwrite: bool
) -> dict[str, Any]:
    """Render both Figure 14 teachers from one shared-action summary."""

    assessment = scientific_conclusions.assess_dual_teacher(summary)
    _prepare_output(output, overwrite)
    teachers = summary["teachers"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Figure 14 rendering requires matplotlib and numpy") from exc
    labels = list(teachers)
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for offset, group, color in ((-width / 2, "correct", "#4C78A8"), (width / 2, "incorrect", "#E45756")):
        means = [float(teachers[label][group]["mean"]) for label in labels]
        stds = [float(teachers[label][group]["std_population"]) for label in labels]
        bars = axis.bar(x + offset, means, width, yerr=stds, capsize=4, label=group, color=color, alpha=0.78)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    auc_labels = []
    for label in labels:
        item = teachers[label]
        ci = item["auroc_bootstrap_ci"]
        auc_labels.append(f"{label}\nAUROC {item['auroc']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    axis.set_xticks(x, auc_labels)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("sequence mean reward (mean ± population std)")
    axis.set_title(title, fontsize=9)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return assessment


def read_figure19_series(path: Path) -> dict[str, list[plot_metrics.Point]]:
    """Read the three Figure 19 diagnostics and report absent keys exactly."""

    series = {name: [] for name in FIG19_METRIC_ALIASES}
    seen_steps: set[float] = set()
    figure19_schema_seen = False
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: metric row must be an object")
            data = payload.get("data", payload)
            step = finite_number(payload.get("step", data.get("step") if isinstance(data, Mapping) else None))
            if not isinstance(data, Mapping) or step is None:
                continue
            if step in seen_steps:
                raise ValueError(f"{path}:{line_number}: duplicate step {step:g}")
            seen_steps.add(step)
            raw_figure19_schema = data.get(FIG19_SCHEMA_KEY)
            if raw_figure19_schema is not None and (
                type(raw_figure19_schema) is not int or raw_figure19_schema != 1
            ):
                raise ValueError(
                    f"{path}:{line_number}: {FIG19_SCHEMA_KEY} must be exact integer 1"
                )
            if raw_figure19_schema == 1:
                figure19_schema_seen = True
            for name, aliases in FIG19_METRIC_ALIASES.items():
                value: float | None = None
                for alias in aliases:
                    value = finite_number(data.get(alias))
                    if value is not None:
                        break
                if value is None:
                    for actual_key, raw_value in data.items():
                        if any(str(actual_key).endswith("/" + alias) for alias in aliases):
                            value = finite_number(raw_value)
                            if value is not None:
                                break
                if value is not None:
                    if (
                        name == "max_abs_adv_probability_difference"
                        and raw_figure19_schema != 1
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: probability-difference value lacks "
                            f"same-row {FIG19_SCHEMA_KEY}=1 provenance"
                        )
                    series[name].append(plot_metrics.Point(step, value))
    missing = [
        f"{name} (accepted keys: {', '.join(FIG19_METRIC_ALIASES[name])})"
        for name, points in series.items()
        if not points
    ]
    if missing:
        raise ValueError("Figure 19 metric series missing from actual log: " + "; ".join(missing))
    if not figure19_schema_seen:
        raise ValueError(
            f"Figure 19 probability-difference series lacks {FIG19_SCHEMA_KEY}=1 provenance"
        )
    step_sets = {name: {point.step for point in points} for name, points in series.items()}
    if len({frozenset(steps) for steps in step_sets.values()}) != 1:
        detail = ", ".join(f"{name}={sorted(steps)}" for name, steps in step_sets.items())
        raise ValueError(f"Figure 19 producer series do not share identical step coverage: {detail}")
    return series


def render_figure19(
    cells: Sequence[CellArtifact], output: Path, title: str, overwrite: bool
) -> None:
    _prepare_output(output, overwrite)
    loaded: list[tuple[CellArtifact, dict[str, list[plot_metrics.Point]]]] = []
    for cell in cells:
        if cell.metrics_path is None:
            raise ValueError(f"{cell.cell_id}: no metrics path")
        loaded.append((cell, read_figure19_series(cell.metrics_path)))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Figure 19 rendering requires matplotlib") from exc
    panels = (
        ("policy_loss", "Policy-gradient loss", "loss"),
        ("grad_norm", "Gradient norm", "norm"),
        (
            "max_abs_adv_probability_difference",
            "Probability difference at largest-|advantage| token",
            "p_student(v*) - p_teacher(v*)",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    for axis, (name, panel_title, ylabel) in zip(axes, panels):
        for cell, metrics in loaded:
            points = metrics[name]
            axis.plot(
                [point.step for point in points],
                [point.value for point in points],
                label=cell.label,
            )
        if name == "max_abs_adv_probability_difference":
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        axis.set_title(panel_title)
        axis.set_xlabel("step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _select_probe(
    probes: Sequence[ProbeArtifact], kind: str, protocol: str
) -> tuple[ProbeArtifact | None, str | None]:
    matching = [probe for probe in probes if probe.kind == kind and probe.protocol == protocol]
    if not matching:
        matching = [probe for probe in probes if probe.kind == kind and probe.protocol == "local-probe"]
    if len(matching) == 1:
        return matching[0], None
    if not matching:
        return None, f"missing {kind} summary artifact"
    return None, f"ambiguous {kind} summaries: {[str(probe.source) for probe in matching]}"


def _base_variant(
    entry: Mapping[str, Any], protocol: str, suite: SuiteArtifact | None
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "status": "skipped",
        "paper_comparable": False,
        "fidelity": str(entry.get("fidelity", "unspecified")),
        "static_fidelity_veto": static_fidelity_veto(entry),
        "ledger_status": entry.get("status"),
        "suite_root": str(suite.root) if suite else None,
        "suite_id": suite.suite_id if suite else None,
        "seed": suite.seed if suite else None,
        "source_tree_sha256": suite.source_tree_sha256 if suite else None,
        "output": None,
        "render_kind": None,
        "missing_producers": [],
        "notes": [],
    }


def _render_training_variant(
    entry: Mapping[str, Any],
    suite: SuiteArtifact,
    protocol: str,
    output_dir: Path,
    probes: Sequence[ProbeArtifact],
    overwrite: bool,
) -> dict[str, Any]:
    figure_id = str(entry["id"])
    producer = entry["producer"]
    required = [str(cell_id) for cell_id in producer.get("cell_ids", [])]
    cells = [suite.cells[cell_id] for cell_id in required]
    variant = _base_variant(entry, protocol, suite)
    output = output_dir / f"{figure_id}_{safe_protocol(protocol)}.png"
    producer_type = str(producer.get("type", ""))
    evaluations, n, evaluation_error = _common_latest_evaluations(cells)

    evaluation_paper = bool(evaluations) and all(
        evaluation_is_paper_comparable(item, protocol) for item in evaluations.values()
    )
    paper_comparable = False
    render_kind: str | None = None
    detail: str | None = None
    fidelity = str(entry.get("fidelity", "unspecified"))

    try:
        if protocol == SCIENTIFIC_PROTOCOL and figure_id == "figure-11":
            exact = _evaluations_at_step(cells, step=PAPER_TEXT_BOUNDARY_STEP, n=16)
            probe, probe_error = _select_probe(probes, "teacher_continuation", protocol)
            if probe is None:
                raise ValueError(probe_error or "missing scientific continuation probe")
            length_keys = {
                "512": "0.5K",
                "1024": "1K",
                "3072": "3K",
                "7168": "7K",
                "10240": "10K",
                "15360": "15K",
            }
            scores = {
                length_keys[_condition_key(cell, "MAX_RESPONSE_LENGTH")]: _aggregate_evaluation(
                    exact[cell.cell_id]
                )
                for cell in cells
            }
            by_length = {
                _condition_key(cell, "MAX_RESPONSE_LENGTH"): cell for cell in cells
            }
            moderate_drop = sum(_overlap_drop(by_length[key]) for key in ("3072", "7168")) / 2
            long_drop = sum(_overlap_drop(by_length[key]) for key in ("10240", "15360")) / 2
            length_assessment = scientific_conclusions.assess_length_sweet_spot(
                scores,
                moderate_overlap_drop=moderate_drop,
                long_overlap_drop=long_drop,
            )
            detail = "step-200 avg@16; 2K fixed prompts; strict >16K paired continuation"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            continuation_assessment = render_scientific_figure11(
                cells, exact, probe.summary, output, title, overwrite
            )
            states = {
                length_assessment["conclusion_state"],
                continuation_assessment["conclusion_state"],
            }
            if states == {"replicated"}:
                conclusion_state = "replicated"
            elif "not_replicated_at_seed_42" in states:
                conclusion_state = "not_replicated_at_seed_42"
            else:
                conclusion_state = "inconclusive"
            variant["claim_assessment"] = {
                "length": length_assessment,
                "continuation": continuation_assessment,
            }
            variant["conclusion_state"] = conclusion_state
            variant["probe_source"] = str(probe.source)
            n = 16
            render_kind = "scientific_figure11_length_endpoint+exact_token_continuation"
        elif protocol == SCIENTIFIC_PROTOCOL and figure_id == "figure-12":
            detail = "steps 1-200 paper-text boundary; 201-260 figure-window reconstruction"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            dynamics = render_scientific_dynamics(
                cells, output, title, overwrite, figure_kind="response-length"
            )
            drops = dynamics["overlap_drop_180_200_to_240_260"]
            by_length = {
                _condition_key(cell, "MAX_RESPONSE_LENGTH"): cell.label for cell in cells
            }
            moderate = sum(drops[by_length[key]] for key in ("3072", "7168")) / 2
            long = sum(drops[by_length[key]] for key in ("10240", "15360")) / 2
            gap = long - moderate
            conclusion_state = "replicated" if gap >= 0.03 else "not_replicated_at_seed_42"
            variant["claim_assessment"] = {
                **dynamics,
                "late_overlap_drop_gap": gap,
                "threshold": 0.03,
            }
            variant["conclusion_state"] = conclusion_state
            render_kind = "scientific_figure12_overlap_entropy_grad_norm"
        elif protocol == SCIENTIFIC_PROTOCOL and figure_id in {"figure-13", "figure-23"}:
            metric = "student" if figure_id == "figure-13" else "teacher"
            if cells[0].metrics_path is None:
                raise ValueError("position-entropy cell has no canonical metrics path")
            all_rows = read_position_entropy_strict(cells[0].metrics_path)
            entropy_evidence = position_entropy_evidence(
                entry, protocol, all_rows, cells[0].environment
            )
            if not entropy_evidence["figure_window_complete"]:
                raise ValueError("scientific position-entropy producer lacks exact 180:10:260 window")
            main_rows = [row for row in all_rows if row.step in POSITION_WINDOW_STEPS]
            if tuple(row.step for row in main_rows) != POSITION_WINDOW_STEPS:
                raise ValueError("main entropy heatmap lacks exact steps 180:10:250")
            detail = "main heatmap=steps 180:10:250; step 260 retained as supplemental evidence"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            render_position_entropy_rows(main_rows, output, metric, title, overwrite)
            variant.update(entropy_evidence)
            variant["main_window_steps"] = list(POSITION_WINDOW_STEPS)
            variant["supplemental_step"] = 260
            variant["conclusion_state"] = "not_assessed"
            render_kind = f"scientific_{metric}_position_entropy_180_250"
        elif protocol == SCIENTIFIC_PROTOCOL and figure_id == "figure-14":
            probe, probe_error = _select_probe(probes, "sequence_reward", protocol)
            if probe is None:
                raise ValueError(probe_error or "missing scientific dual-teacher reward probe")
            detail = "1,070 prompts x4 fixed actions; shared fingerprint; prompt-bootstrap AUROC CI"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            assessment = render_scientific_dual_reward(
                probe.summary, output, title, overwrite
            )
            variant["claim_assessment"] = assessment
            variant["conclusion_state"] = assessment["conclusion_state"]
            variant["probe_source"] = str(probe.source)
            render_kind = "scientific_figure14_dual_teacher_shared_action_auroc"
        elif protocol == SCIENTIFIC_PROTOCOL and figure_id == "figure-15":
            exact = _evaluations_at_step(cells, step=PAPER_TEXT_BOUNDARY_STEP, n=16)
            detail = "five registered support conditions; step-200 avg@16"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            assessment = render_scientific_endpoint_bars(
                cells, exact, output, title, overwrite
            )
            variant["claim_assessment"] = assessment
            variant["conclusion_state"] = assessment["conclusion_state"]
            n = 16
            render_kind = "scientific_figure15_five_support_endpoint_bars"
        elif protocol == SCIENTIFIC_PROTOCOL and figure_id == "figure-16":
            detail = "Top-1/4/16/64 dynamics through step 260; sampled condition belongs to Fig.15"
            title = _annotation(figure_id, protocol, fidelity, False, detail)
            dynamics = render_scientific_dynamics(
                cells, output, title, overwrite, figure_kind="top-k"
            )
            all_ids = ["fig16-topk-sampled", "fig16-topk-1", "fig16-topk-4", "fig16-topk-16", "fig16-topk-64"]
            if any(cell_id not in suite.cells or not suite.cells[cell_id].complete for cell_id in all_ids):
                raise ValueError("Figure 16 conclusion requires all five Figure 15 endpoint cells")
            all_cells = [suite.cells[cell_id] for cell_id in all_ids]
            exact = _evaluations_at_step(all_cells, step=PAPER_TEXT_BOUNDARY_STEP, n=16)
            normalized = {}
            for cell in all_cells:
                key = _condition_key(cell, "TOP_K")
                normalized["sampled" if key == "0" else key] = _aggregate_evaluation(exact[cell.cell_id])
            assessment = scientific_conclusions.assess_topk(normalized)
            variant["claim_assessment"] = {"endpoint": assessment, "dynamics": dynamics}
            variant["conclusion_state"] = assessment["conclusion_state"]
            render_kind = "scientific_figure16_topk_overlap_entropy_grad_norm"
        elif producer_type == "composite_summary":
            raise ValueError("paper-authored overview/composite layout is not reconstructed from partial local metrics")
        if producer_type in TRAINING_DYNAMIC_TYPES:
            if evaluations:
                paper_comparable = evaluation_paper and static_fidelity_veto(entry) is None
                detail = f"latest common evaluation: avg@{n}"
                title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
                render_training_and_evaluation(cells, evaluations, output, title, overwrite)
                render_kind = "training_diagnostics+latest_checkpoint_three_benchmark_grouped_bar"
            else:
                paper_comparable = (
                    producer_type in {"opd_training_dynamics_plot", "metric_plot_reuse"}
                    and protocol == "paper"
                    and static_fidelity_veto(entry) is None
                )
                detail = "training diagnostics only; avg@16 evaluation unavailable"
                title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
                render_dynamics(cells, output, title, overwrite)
                render_kind = "training_diagnostics"
                if evaluation_error:
                    variant["missing_producers"].append(evaluation_error)
        elif producer_type in BAR_ONLY_TYPES:
            if not evaluations:
                raise ValueError(evaluation_error or "missing complete three-benchmark evaluation")
            paper_comparable = evaluation_paper and static_fidelity_veto(entry) is None
            detail = f"latest common checkpoint evaluation: avg@{n}"
            title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
            render_grouped_bar(
                evaluations,
                {cell.cell_id: cell.label for cell in cells},
                output,
                title,
                overwrite,
            )
            render_kind = "latest_checkpoint_three_benchmark_grouped_bar"
        elif producer_type in {"position_entropy_heatmap", "position_entropy_heatmap_reuse"}:
            metric = "student" if figure_id == "figure-13" else "teacher"
            if cells[0].metrics_path is None:
                raise ValueError("position-entropy cell has no metrics path")
            position_rows = read_position_entropy_strict(cells[0].metrics_path)
            entropy_evidence = position_entropy_evidence(
                entry, protocol, position_rows, cells[0].environment
            )
            paper_comparable = bool(entropy_evidence["paper_comparable"])
            variant.update(entropy_evidence)
            observed_steps = [row.step for row in position_rows]
            detail = (
                f"positional-entropy steps={observed_steps}; "
                f"evidence={entropy_evidence['label']}"
            )
            title = _annotation(
                figure_id,
                protocol,
                fidelity,
                paper_comparable,
                detail,
            )
            render_position_entropy_rows(
                position_rows, output, metric, title, overwrite
            )
            render_kind = f"{metric}_position_entropy_heatmap"
        elif producer_type == "mixed_train_eval_and_continuation_probe":
            if not evaluations:
                raise ValueError(evaluation_error or "Figure 11(a) lacks complete benchmark evaluation")
            probe, probe_error = _select_probe(probes, "teacher_continuation", protocol)
            if probe is None:
                raise ValueError(probe_error or "missing continuation probe")
            _validate_continuation_summary(probe.summary)
            probe_exact = probe.summary.get("sampling_provenance") == "author-code-verified"
            paper_comparable = (
                evaluation_paper and probe_exact and static_fidelity_veto(entry) is None
            )
            detail = f"avg@{n} plus continuation probe={probe.source}"
            title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
            with tempfile.TemporaryDirectory(prefix="opd-paper-render-") as directory:
                temp = Path(directory)
                training = temp / "training.png"
                continuation = temp / "continuation.png"
                render_training_and_evaluation(cells, evaluations, training, title, overwrite=True)
                render_continuation(probe.summary, continuation, title, overwrite=True)
                compose_images((training, continuation), output, title, overwrite)
            render_kind = "length_training_evaluation+teacher_continuation_probe"
            variant["probe_source"] = str(probe.source)
        elif producer_type == "rollout_reward_auroc_analysis":
            probe, probe_error = _select_probe(probes, "sequence_reward", protocol)
            if probe is None:
                raise ValueError(probe_error or "missing sequence-reward probe")
            paper_comparable = (
                probe.summary.get("paper_comparable") is True
                and protocol == "paper"
                and static_fidelity_veto(entry) is None
            )
            detail = f"fixed-rollout reward probe={probe.source}"
            title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
            render_sequence_reward(probe.summary, output, title, overwrite)
            render_kind = "sequence_reward_summary_with_auroc"
            variant["probe_source"] = str(probe.source)
        elif producer_type == "optimization_diagnostics_reuse":
            paper_comparable = protocol == "paper" and static_fidelity_veto(entry) is None
            detail = (
                "requires actor/pg_loss, actor/grad_norm, and schema-v1 "
                "largest-|advantage| token probability difference"
            )
            title = _annotation(figure_id, protocol, fidelity, paper_comparable, detail)
            render_figure19(cells, output, title, overwrite)
            render_kind = "figure19_optimization_diagnostics"
        else:
            raise ValueError(f"unsupported producer type {producer_type!r}")
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        variant["missing_producers"].append(str(exc))
        return variant

    variant.update(
        {
            "status": "rendered",
            "paper_comparable": paper_comparable,
            "output": str(output.resolve()),
            "render_kind": render_kind,
            "annotation": _annotation(figure_id, protocol, fidelity, paper_comparable, detail),
            "evaluation_n": n,
            "producer_cells": [
                {
                    "cell_id": cell.cell_id,
                    "fingerprint": cell.fingerprint,
                    "metrics_path": str(cell.metrics_path),
                    "last_metric_step": _last_run_step(cell.run) if cell.run else None,
                }
                for cell in cells
            ],
        }
    )
    return variant


def _render_model_evaluation_variants(
    entry: Mapping[str, Any],
    evaluations: Sequence[tuple[str, EvaluationArtifact]],
    output_dir: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    expected = [str(model) for model in entry["producer"].get("models", [])]
    grouped: dict[str, list[tuple[str, EvaluationArtifact]]] = {}
    for identity, item in evaluations:
        grouped.setdefault(infer_external_protocol(item), []).append((identity, item))
    if not grouped:
        grouped = {"unavailable": []}
    variants: list[dict[str, Any]] = []
    for protocol, items in sorted(grouped.items()):
        variant = _base_variant(entry, protocol, None)
        by_identity: dict[str, list[EvaluationArtifact]] = {}
        for identity, item in items:
            by_identity.setdefault(identity, []).append(item)
        missing = [identity for identity in expected if identity not in by_identity]
        ambiguous = [identity for identity in expected if len(by_identity.get(identity, [])) > 1]
        if missing:
            variant["missing_producers"].extend(f"missing model evaluation: {item}" for item in missing)
        if ambiguous:
            variant["missing_producers"].extend(f"ambiguous model evaluation: {item}" for item in ambiguous)
        if missing or ambiguous:
            variants.append(variant)
            continue
        selected = {identity: by_identity[identity][0] for identity in expected}
        n_values = {item.n for item in selected.values()}
        if len(n_values) != 1:
            variant["missing_producers"].append(f"model evaluations mix n values: {sorted(n_values)}")
            variants.append(variant)
            continue
        n = next(iter(n_values))
        paper_comparable = all(
            evaluation_is_paper_comparable(item, protocol) for item in selected.values()
        ) and static_fidelity_veto(entry) is None
        fidelity = str(entry.get("fidelity", "unspecified"))
        title = _annotation(
            str(entry["id"]),
            protocol,
            fidelity,
            paper_comparable,
            f"immutable model evaluation: avg@{n}",
        )
        output = output_dir / f"{entry['id']}_{safe_protocol(protocol)}.png"
        try:
            render_grouped_bar(
                selected,
                {identity: item.label for identity, item in selected.items()},
                output,
                title,
                overwrite,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
            variant["missing_producers"].append(str(exc))
            variants.append(variant)
            continue
        variant.update(
            {
                "status": "rendered",
                "paper_comparable": paper_comparable,
                "output": str(output.resolve()),
                "render_kind": "immutable_model_three_benchmark_grouped_bar",
                "annotation": title,
                "evaluation_n": n,
                "model_evaluations": [
                    {"identity": identity, "summary": str(item.source)}
                    for identity, item in selected.items()
                ],
            }
        )
        variants.append(variant)
    return variants


def render_all(
    *,
    ledger_path: Path,
    suite_roots: Sequence[Path],
    model_eval_roots: Sequence[Path],
    probe_roots: Sequence[Path],
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    entries = load_ledger(ledger_path)
    roots = _expand_suite_roots(suite_roots)
    suites = [load_suite(path) for path in roots]
    model_evaluations = discover_model_evaluations(model_eval_roots)
    probes = discover_probes(probe_roots)
    protocols = sorted({suite.protocol for suite in suites})
    preferred_suite_id = read_object(ledger_path).get("suite_id")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figure_records: list[dict[str, Any]] = []

    for figure_id in FIGURE_IDS:
        entry = entries[figure_id]
        producer = entry.get("producer")
        if not isinstance(producer, Mapping):
            raise ValueError(f"{figure_id}: producer must be an object")
        required = producer.get("cell_ids")
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ValueError(f"{figure_id}: producer.cell_ids must be a string list")

        if producer.get("type") == "teacher_evaluation_only":
            variants = _render_model_evaluation_variants(
                entry, model_evaluations, destination, overwrite
            )
        else:
            figure_protocols = protocols or ["unavailable"]
            variants = []
            for protocol in figure_protocols:
                suite, missing, notes = _choose_suite(
                    suites, protocol, required, preferred_suite_id if isinstance(preferred_suite_id, str) else None
                )
                variant = _base_variant(entry, protocol, suite)
                variant["notes"].extend(notes)
                if suite is None or missing:
                    variant["missing_producers"].extend(missing)
                    variants.append(variant)
                    continue
                rendered = _render_training_variant(
                    entry, suite, protocol, destination, probes, overwrite
                )
                rendered["notes"].extend(notes)
                variants.append(rendered)

        rendered_outputs = [variant["output"] for variant in variants if variant["status"] == "rendered"]
        figure_records.append(
            {
                "id": figure_id,
                "paper_location": entry.get("paper_location"),
                "status": "rendered" if rendered_outputs else "skipped",
                "outputs": rendered_outputs,
                "variants": variants,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "ledger": str(ledger_path.expanduser().resolve()),
        "suite_roots": [str(path) for path in roots],
        "model_eval_roots": [str(path.expanduser().resolve()) for path in model_eval_roots],
        "probe_roots": [str(path.expanduser().resolve()) for path in probe_roots],
        "policy": {
            "missing_producer": "skip (fail-closed)",
            "smoke": "renderable diagnostic, never paper-comparable",
            "benchmark_comparability": "requires avg@16 and verified/explicit full paper sampling",
        },
        "summary": {
            "figures": len(figure_records),
            "rendered": sum(record["status"] == "rendered" for record in figure_records),
            "skipped": sum(record["status"] == "skipped" for record in figure_records),
            "paper_comparable_variants": sum(
                variant.get("paper_comparable") is True
                for record in figure_records
                for variant in record["variants"]
            ),
            "conclusions": {
                state: sum(
                    variant.get("conclusion_state") == state
                    for record in figure_records
                    for variant in record["variants"]
                )
                for state in (
                    "replicated",
                    "not_replicated_at_seed_42",
                    "inconclusive",
                    "not_assessed",
                )
            },
        },
        "figures": figure_records,
    }
    manifest_path = destination / "render_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite")
    atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_ledger = Path(__file__).with_name("paper_experiment_ledger.json")
    parser.add_argument("--ledger", type=Path, default=default_ledger)
    parser.add_argument(
        "--suite-root",
        type=Path,
        action="append",
        default=[],
        help="Suite root containing suite_manifest.json; repeatable.",
    )
    parser.add_argument(
        "--model-eval-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing immutable model summary.json artifacts; repeatable.",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing continuation/reward probe summaries; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = render_all(
            ledger_path=args.ledger,
            suite_roots=args.suite_root,
            model_eval_roots=args.model_eval_root,
            probe_roots=args.probe_root,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
    print(f"wrote {args.output_dir / 'render_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
