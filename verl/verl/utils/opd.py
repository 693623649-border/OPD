"""Small, testable helpers for OPD Top-k rewards and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def reward_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    weight_mode: str,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return masked reward weights, optionally normalized on the support.

    The released OPD implementation uses unnormalized probability mass only for
    the Non-Overlap (symmetric-difference) arm.  Keeping ``normalize`` explicit
    prevents that author-code convention from being silently mixed with the
    scale-controlled variant that renormalizes every selected support.
    """

    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.shape != valid_mask.shape:
        raise ValueError("student/teacher log-probs and valid_mask must have the same shape")
    valid_mask = valid_mask.bool()
    if weight_mode == "student_p":
        source = student_log_probs
    elif weight_mode == "teacher_p":
        source = teacher_log_probs
    elif weight_mode == "none":
        source = torch.zeros_like(student_log_probs)
    else:
        raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")

    masked = torch.where(valid_mask, source, torch.full_like(source, -torch.inf))
    if normalize:
        masked = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    weights = torch.where(valid_mask, torch.exp(masked), torch.zeros_like(masked))
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_reward_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    weight_mode: str,
) -> torch.Tensor:
    """Backward-compatible helper for weights normalized on a selected support."""

    return reward_weights(
        student_log_probs,
        teacher_log_probs,
        valid_mask,
        weight_mode,
        normalize=True,
    )


def build_topk_distillation_tensors(
    *,
    strategy: str,
    reward_weight_mode: str,
    student_ids: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    teacher_ids: torch.Tensor | None = None,
    teacher_log_probs: torch.Tensor | None = None,
    student_on_teacher_log_probs: torch.Tensor | None = None,
    student_in_teacher_mask: torch.Tensor | None = None,
    teacher_in_student_mask: torch.Tensor | None = None,
    support_weight_normalization: str = "author",
) -> dict[str, torch.Tensor]:
    """Build token-aligned replay tensors and reverse-KL rewards for one strategy."""

    if support_weight_normalization not in {"author", "selected"}:
        raise ValueError(
            "support_weight_normalization must be 'author' or 'selected', got "
            f"{support_weight_normalization!r}"
        )

    def required(name: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{strategy} requires {name}")
        return value

    if strategy == "only_stu":
        valid_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
        weights = normalized_reward_weights(
            student_log_probs, teacher_on_student_log_probs, valid_mask, reward_weight_mode
        )
        rewards = -(student_log_probs - teacher_on_student_log_probs) * weights
        return {"rm_scores": rewards}

    if strategy == "only_tch":
        teacher_ids = required("teacher_ids", teacher_ids)
        teacher_log_probs = required("teacher_log_probs", teacher_log_probs)
        student_on_teacher_log_probs = required(
            "student_on_teacher_log_probs", student_on_teacher_log_probs
        )
        valid_mask = torch.ones_like(student_on_teacher_log_probs, dtype=torch.bool)
        weights = normalized_reward_weights(
            student_on_teacher_log_probs, teacher_log_probs, valid_mask, reward_weight_mode
        )
        rewards = -(student_on_teacher_log_probs - teacher_log_probs) * weights
        return {
            "rm_scores": rewards,
            # update_policy treats any non-student support as a generic union replay support.
            "union_top_k_ids": teacher_ids,
            "union_top_k_log_probs": student_on_teacher_log_probs,
            "student_log_probs_on_teacher_ids": student_on_teacher_log_probs,
        }

    if strategy == "intersection":
        student_in_teacher_mask = required("student_in_teacher_mask", student_in_teacher_mask).bool()
        weights = normalized_reward_weights(
            student_log_probs,
            teacher_on_student_log_probs,
            student_in_teacher_mask,
            reward_weight_mode,
        )
        rewards = -(student_log_probs - teacher_on_student_log_probs) * weights
        rewards = torch.where(student_in_teacher_mask, rewards, torch.zeros_like(rewards))
        return {"rm_scores": rewards}

    if strategy in {"union", "union-intersection"}:
        teacher_ids = required("teacher_ids", teacher_ids)
        teacher_log_probs = required("teacher_log_probs", teacher_log_probs)
        student_on_teacher_log_probs = required(
            "student_on_teacher_log_probs", student_on_teacher_log_probs
        )
        student_in_teacher_mask = required("student_in_teacher_mask", student_in_teacher_mask).bool()
        teacher_in_student_mask = required("teacher_in_student_mask", teacher_in_student_mask).bool()

        union_ids = torch.cat([student_ids, teacher_ids], dim=-1)
        student_union = torch.cat([student_log_probs, student_on_teacher_log_probs], dim=-1)
        teacher_union = torch.cat([teacher_on_student_log_probs, teacher_log_probs], dim=-1)
        if strategy == "union":
            valid_mask = torch.cat(
                [torch.ones_like(student_ids, dtype=torch.bool), ~teacher_in_student_mask], dim=-1
            )
        else:
            valid_mask = torch.cat([~student_in_teacher_mask, ~teacher_in_student_mask], dim=-1)

        # The author code deliberately leaves the Non-Overlap arm at its raw
        # (usually small) probability mass.  ``selected`` is a scale-controlled
        # robustness variant that renormalizes the symmetric difference.
        normalize = strategy != "union-intersection" or support_weight_normalization == "selected"
        weights = reward_weights(
            student_union,
            teacher_union,
            valid_mask,
            reward_weight_mode,
            normalize=normalize,
        )
        rewards = -(student_union - teacher_union) * weights
        rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
        return {
            "rm_scores": rewards,
            "union_top_k_ids": union_ids,
            "union_top_k_log_probs": student_union,
            "student_log_probs_on_teacher_ids": student_on_teacher_log_probs,
        }

    raise ValueError(f"Unknown top_k_strategy: {strategy}")


def validate_topk_replay_tensors(batch: Mapping[str, torch.Tensor]) -> tuple[str, str]:
    """Select matching token-ID/old-log-prob keys or fail before PPO replay."""

    if "union_top_k_ids" in batch:
        id_key, log_prob_key = "union_top_k_ids", "union_top_k_log_probs"
    else:
        id_key, log_prob_key = "student_top_k_ids", "student_top_k_log_probs"
    if id_key not in batch or log_prob_key not in batch:
        raise ValueError(f"3D Top-k replay requires both {id_key} and {log_prob_key}")
    if batch[id_key].shape != batch[log_prob_key].shape:
        raise ValueError(
            f"Top-k replay tensor shape mismatch: {id_key}={tuple(batch[id_key].shape)} "
            f"vs {log_prob_key}={tuple(batch[log_prob_key].shape)}"
        )
    return id_key, log_prob_key


def eq7_intersection_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    overlap_mask: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute the paper's Eq. (7), including per-intersection renormalization."""

    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.shape != overlap_mask.shape:
        raise ValueError("Eq. (7) inputs must have the same [batch, sequence, k] shape")
    valid = overlap_mask.bool() & torch.isfinite(student_log_probs) & torch.isfinite(teacher_log_probs)
    if response_mask is not None:
        if response_mask.shape != student_log_probs.shape[:2]:
            raise ValueError("response_mask must match the batch/sequence dimensions")
        valid = valid & response_mask.bool().unsqueeze(-1)
    valid_positions = valid.any(dim=-1)
    if not bool(valid_positions.any()):
        return None

    neg_inf = torch.full_like(student_log_probs, -torch.inf)
    student_masked = torch.where(valid, student_log_probs, neg_inf)
    teacher_masked = torch.where(valid, teacher_log_probs, neg_inf)
    student_bar_log = student_log_probs - torch.logsumexp(student_masked, dim=-1, keepdim=True)
    teacher_bar_log = teacher_log_probs - torch.logsumexp(teacher_masked, dim=-1, keepdim=True)
    token_advantage = torch.where(
        valid,
        torch.exp(student_bar_log) * (teacher_bar_log - student_bar_log),
        torch.zeros_like(student_log_probs),
    )
    token_advantage = torch.nan_to_num(token_advantage, nan=0.0, posinf=0.0, neginf=0.0)
    counts = valid.sum(dim=-1).clamp_min(1)
    per_position = token_advantage.sum(dim=-1) / counts
    return per_position[valid_positions].mean()


def student_side_topk_advantages(
    advantages: torch.Tensor,
    top_k: int,
    *,
    union_support: bool,
) -> torch.Tensor:
    """Return advantages aligned with the student's Top-k token IDs.

    Union replay stores ``[student Top-k, teacher Top-k]`` along the last
    dimension.  Figure 19 is defined on the overlap entries of the *student*
    Top-k support, so the teacher-side half must never participate in the
    largest-absolute-advantage selection.
    """

    if advantages.ndim != 3:
        raise ValueError("advantages must have [batch, sequence, support] shape")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    expected_width = 2 * top_k if union_support else top_k
    if advantages.shape[-1] != expected_width:
        support_name = "union" if union_support else "student"
        raise ValueError(
            f"{support_name} advantages must have support width {expected_width}, "
            f"got {advantages.shape[-1]}"
        )
    return advantages[..., :top_k]


def prob_diff_at_max_abs_advantage(
    student_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    overlap_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Compute the Figure 19 probability difference at ``argmax |adv|``.

    For every valid response state, this selects one token from the overlap of
    the student and teacher Top-k supports using the largest absolute
    student-side training advantage.  It then records
    ``p_student(v*) - p_teacher(v*)`` and averages those signed differences over
    states.  Non-finite candidates are excluded.  States without a finite
    overlap candidate are omitted; ``None`` is returned if no state remains.

    The three token-valued inputs and ``overlap_mask`` must be aligned to the
    student's Top-k IDs.  In particular, callers using union replay must first
    select the student-side half with :func:`student_side_topk_advantages`.
    """

    token_tensors = {
        "student_advantages": student_advantages,
        "student_log_probs": student_log_probs,
        "teacher_on_student_log_probs": teacher_on_student_log_probs,
        "overlap_mask": overlap_mask,
    }
    if any(tensor.ndim != 3 for tensor in token_tensors.values()):
        raise ValueError("Figure 19 token inputs must have [batch, sequence, k] shape")
    expected_shape = student_advantages.shape
    mismatched = {
        name: tuple(tensor.shape)
        for name, tensor in token_tensors.items()
        if tensor.shape != expected_shape
    }
    if mismatched:
        raise ValueError(
            "Figure 19 token inputs must have the same shape; "
            f"expected {tuple(expected_shape)}, got {mismatched}"
        )
    if response_mask.ndim != 2 or response_mask.shape != expected_shape[:2]:
        raise ValueError("response_mask must match the [batch, sequence] dimensions")
    if not all(
        tensor.is_floating_point()
        for tensor in (student_advantages, student_log_probs, teacher_on_student_log_probs)
    ):
        raise ValueError("advantages and log-probabilities must be floating-point tensors")
    devices = {tensor.device for tensor in (*token_tensors.values(), response_mask)}
    if len(devices) != 1:
        raise ValueError("all Figure 19 inputs must be on the same device")

    # An empty Top-k axis is a well-defined empty support, not an argmax error.
    if expected_shape[-1] == 0:
        return None

    overlap = overlap_mask.bool()
    response = response_mask.bool()
    if overlap_mask.is_floating_point():
        overlap = overlap & torch.isfinite(overlap_mask)
    if response_mask.is_floating_point():
        response = response & torch.isfinite(response_mask)
    valid = (
        overlap
        & response.unsqueeze(-1)
        & torch.isfinite(student_advantages)
        & torch.isfinite(student_log_probs)
        & torch.isfinite(teacher_on_student_log_probs)
    )
    valid_states = valid.any(dim=-1)
    if not bool(valid_states.any()):
        return None

    absolute_advantages = torch.where(
        valid,
        student_advantages.abs(),
        torch.full_like(student_advantages, -torch.inf),
    )
    selected_indices = absolute_advantages.argmax(dim=-1, keepdim=True)
    selected_student_log_probs = student_log_probs.gather(-1, selected_indices).squeeze(-1)
    selected_teacher_log_probs = teacher_on_student_log_probs.gather(-1, selected_indices).squeeze(-1)
    probability_difference = selected_student_log_probs.exp() - selected_teacher_log_probs.exp()
    if not bool(torch.isfinite(probability_difference[valid_states]).all()):
        raise ValueError("selected Figure 19 probability differences must be finite")
    return probability_difference[valid_states].mean()


def eq8_absolute_entropy_gap(
    student_entropy: torch.Tensor,
    teacher_entropy: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Compute paper Eq. (8): mean per-state ``|H_teacher - H_student|``."""

    if student_entropy.shape != teacher_entropy.shape or student_entropy.shape != response_mask.shape:
        raise ValueError("entropy tensors and response_mask must have the same [batch, sequence] shape")
    valid = response_mask.bool() & torch.isfinite(student_entropy) & torch.isfinite(teacher_entropy)
    if not bool(valid.any()):
        return None
    return torch.abs(teacher_entropy - student_entropy)[valid].mean()


def binned_masked_mean(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average a ``[batch, sequence]`` diagnostic into fixed position bins.

    Returns ``(means, counts)``. Empty bins have a finite zero mean and a zero
    count so downstream heatmap code can mask them without serializing NaNs.
    """

    if values.shape != response_mask.shape or values.ndim != 2:
        raise ValueError("values and response_mask must have the same [batch, sequence] shape")
    if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")

    valid = response_mask.bool() & torch.isfinite(values)
    sequence_length = values.shape[1]
    num_bins = (sequence_length + bin_size - 1) // bin_size
    padded_length = num_bins * bin_size
    pad = padded_length - sequence_length
    if pad:
        values = torch.nn.functional.pad(values, (0, pad), value=0.0)
        valid = torch.nn.functional.pad(valid, (0, pad), value=False)
    valid_float = valid.to(values.dtype)
    sums = (torch.where(valid, values, torch.zeros_like(values)) * valid_float).reshape(
        values.shape[0], num_bins, bin_size
    ).sum(dim=(0, 2))
    counts = valid_float.reshape(values.shape[0], num_bins, bin_size).sum(dim=(0, 2))
    means = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    return means, counts
