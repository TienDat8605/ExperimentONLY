"""Continuous, per-head adaptive mask accumulation for Proposal 6."""

from typing import Dict, Iterable, Optional, Tuple

import torch


def resolve_proposal6_layers(
    num_layers: int,
    mask_layers: Optional[Iterable[int]],
) -> Tuple[int, ...]:
    """Resolve exact update layers while reserving the final integration layer."""
    if num_layers < 2:
        raise ValueError("Proposal 6 requires at least two decoder layers")

    final_layer = num_layers - 1
    selected = list(range(final_layer)) if mask_layers is None else list(mask_layers)
    if not selected:
        raise ValueError("Proposal 6 requires at least one accumulation layer")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Proposal 6 accumulation layers must be unique: {selected}")

    invalid = [layer for layer in selected if layer < 0 or layer >= final_layer]
    if invalid:
        raise ValueError(
            f"Proposal 6 layers must be between 0 and {final_layer - 1}; "
            f"layer {final_layer} is reserved for final integration. Invalid: {invalid}"
        )
    return tuple(sorted(selected))


def validate_alpha_bounds(alpha_min: float, alpha_max: float) -> None:
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError(
            "Proposal 6 mask alpha bounds must satisfy "
            f"0 <= min <= max <= 1; got min={alpha_min}, max={alpha_max}"
        )


def continuous_tver_evidence(ratio: torch.Tensor) -> torch.Tensor:
    """Map per-head TVER ratios to stable visual confidence in [0, 1]."""
    ratio = ratio.reshape(-1)
    centered = ratio - ratio.mean()
    scale = ratio.std(unbiased=False).clamp_min(1e-6)
    return torch.sigmoid(centered / scale)


def proposal6_keep_masks(cumulative_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint and exhaustive visual/text keep masks."""
    visual_keep = cumulative_mask[0] >= 0.5
    text_keep = ~visual_keep
    return visual_keep, text_keep


def update_proposal6_mask(
    ratio: torch.Tensor,
    cumulative_mask: Optional[torch.Tensor],
    alpha_min: float = 0.05,
    alpha_max: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Update continuous TVER evidence with confidence-innovation per-head gains."""
    validate_alpha_bounds(alpha_min, alpha_max)
    evidence = continuous_tver_evidence(ratio)
    confidence = torch.abs(2.0 * evidence - 1.0)

    if cumulative_mask is None:
        visual_state = evidence
        innovation = torch.zeros_like(evidence)
        gain = torch.zeros_like(evidence)
        initialized = torch.tensor(True, device=evidence.device)
    else:
        previous_visual = cumulative_mask[0].to(device=evidence.device, dtype=evidence.dtype)
        innovation = torch.abs(evidence - previous_visual)
        gain = alpha_min + (alpha_max - alpha_min) * confidence * innovation
        visual_state = gain * evidence + (1.0 - gain) * previous_visual
        initialized = torch.tensor(False, device=evidence.device)

    cumulative_mask = torch.stack((visual_state, 1.0 - visual_state), dim=0)
    visual_keep, text_keep = proposal6_keep_masks(cumulative_mask)
    stats = {
        "evidence": evidence,
        "confidence": confidence,
        "innovation": innovation,
        "gain": gain,
        "visual_suppressed": (~visual_keep).sum(),
        "text_suppressed": (~text_keep).sum(),
        "initialized": initialized,
    }
    return cumulative_mask, stats
