#!/usr/bin/env python3
"""Pre-registered conclusion predicates for the long-horizon OPD figures.

These pure functions distinguish a valid negative result from an engineering
failure.  They never inspect process exit codes: callers must first establish
that all required producers are scientifically valid and then pass only finite
measurements into these predicates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


CONCLUSIONS = {
    "replicated",
    "not_replicated_at_seed_42",
    "inconclusive",
}


def _finite_map(values: Mapping[str, Any], expected: set[str], label: str) -> dict[str, float]:
    if set(values) != expected:
        raise ValueError(f"{label} keys are {sorted(values)}, expected {sorted(expected)}")
    output: dict[str, float] = {}
    for key, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"{label}.{key} must be finite")
        output[key] = float(raw)
    return output


def assess_length_sweet_spot(
    scores: Mapping[str, Any],
    *,
    moderate_overlap_drop: float,
    long_overlap_drop: float,
) -> dict[str, Any]:
    """Assess Figures 11(a)/12 using the registered six-length predicate."""

    checked = _finite_map(scores, {"0.5K", "1K", "3K", "7K", "10K", "15K"}, "length scores")
    for label, value in (
        ("moderate_overlap_drop", moderate_overlap_drop),
        ("long_overlap_drop", long_overlap_drop),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    moderate_beats_edges = min(checked["3K"], checked["7K"]) > max(
        checked["0.5K"], checked["1K"], checked["10K"], checked["15K"]
    )
    instability_gap = float(long_overlap_drop) - float(moderate_overlap_drop)
    instability = instability_gap >= 0.03
    replicated = moderate_beats_edges and instability
    return {
        "conclusion_state": "replicated" if replicated else "not_replicated_at_seed_42",
        "moderate_lengths_beat_short_and_long": moderate_beats_edges,
        "late_overlap_drop_gap": instability_gap,
        "late_instability_threshold_met": instability,
        "rule": "both 3K/7K exceed every 0.5K/1K/10K/15K score and long-minus-moderate overlap drop >=0.03",
    }


def assess_topk(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Assess Figures 15/16 from the five registered endpoint conditions."""

    checked = _finite_map(scores, {"sampled", "1", "4", "16", "64"}, "Top-k scores")
    best = max(checked.values())
    top1_worst = checked["1"] == min(checked.values()) and sum(
        value == checked["1"] for value in checked.values()
    ) == 1
    top4_near_best = best - checked["4"] <= 0.02
    above_four_gain = max(checked["16"], checked["64"]) - checked["4"]
    above_four_marginal = above_four_gain <= 0.02
    sampled_comparable = best - checked["sampled"] <= 0.03
    replicated = top1_worst and top4_near_best and above_four_marginal and sampled_comparable
    return {
        "conclusion_state": "replicated" if replicated else "not_replicated_at_seed_42",
        "top1_is_unique_worst": top1_worst,
        "top4_gap_to_best": best - checked["4"],
        "gain_above_top4": above_four_gain,
        "sampled_gap_to_best": best - checked["sampled"],
        "rule": "Top-1 unique worst; Top-4 and k>4 within 0.02; sampled within 0.03 of best",
    }


def assess_continuation(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and expose the Figure 11(b) pre-registered decision."""

    state = summary.get("conclusion_state")
    if state not in CONCLUSIONS:
        raise ValueError("continuation summary lacks a registered conclusion_state")
    count = summary.get("num_selected_rollouts")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("continuation num_selected_rollouts must be non-negative")
    if count < 30:
        if state != "inconclusive":
            raise ValueError("a continuation probe with fewer than 30 rollouts must be inconclusive")
        return {"conclusion_state": state, "underpowered": True, "num_selected_rollouts": count}
    bootstrap = summary.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("powered continuation summary lacks bootstrap evidence")
    return {"conclusion_state": state, "underpowered": False, "num_selected_rollouts": count}


def assess_dual_teacher(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared-action/two-teacher Figure 14 conclusion."""

    state = summary.get("conclusion_state")
    if state not in CONCLUSIONS:
        raise ValueError("dual-teacher summary lacks a registered conclusion_state")
    teachers = summary.get("teachers")
    if not isinstance(teachers, Mapping) or len(teachers) != 2:
        raise ValueError("Figure 14 requires exactly two teachers")
    fingerprint = summary.get("action_batch_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Figure 14 lacks a SHA-256 shared action fingerprint")
    for label, teacher in teachers.items():
        if not isinstance(teacher, Mapping) or teacher.get("action_batch_sha256") != fingerprint:
            raise ValueError(f"teacher {label!r} is not bound to the shared action batch")
    return {"conclusion_state": state, "num_teachers": 2, "action_batch_sha256": fingerprint}

