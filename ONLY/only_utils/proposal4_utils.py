"""Pure tensor utilities for Layer-Local Consensus ONLY (proposal 4)."""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def normalized_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entropy normalized by support size, so text length cannot drive the score."""
    support = probabilities.shape[-1]
    if support < 2:
        return torch.zeros(probabilities.shape[:-1], dtype=probabilities.dtype, device=probabilities.device)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    return entropy / math.log(support)


def soft_head_gate(
    attention_probabilities: torch.Tensor,
    image_token_start: int,
    image_token_end: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a layer-local soft head gate and normalized entropy score."""
    key_length = attention_probabilities.shape[-1]
    if not (0 <= image_token_start < image_token_end <= key_length):
        raise ValueError(
            f"Invalid image span [{image_token_start}, {image_token_end}) for key length {key_length}"
        )
    text_probabilities = torch.cat(
        (
            attention_probabilities[:, :, -1, :image_token_start],
            attention_probabilities[:, :, -1, image_token_end:],
        ),
        dim=-1,
    )
    visual_probabilities = attention_probabilities[:, :, -1, image_token_start:image_token_end]
    score = (
        normalized_entropy(text_probabilities) - normalized_entropy(visual_probabilities)
    ).mean(dim=0)

    median = score.median()
    mad = (score - median).abs().median()
    robust_scale = (1.4826 * mad).clamp_min(1e-6)
    gate = torch.sigmoid((score - median) / (max(float(temperature), 1e-3) * robust_scale))
    return gate, score


def consensus_logits(
    normal_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    gamma: float,
    alpha_pos: float = 3.0,
    alpha_neg: float = 1.0,
    consensus_min: float = 0.75,
    consensus_strength: float = 1.0,
    residual_min: float = 0.05,
    residual_clip: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine layer-local experts with exact layer-0 ONLY as the fallback."""
    if expert_logits.ndim != 3 or expert_logits.shape[1] < 1:
        raise ValueError(f"Expected [batch, experts, vocab] expert logits, got {expert_logits.shape}")

    normal_prob = F.softmax(normal_logits, dim=-1)
    expert_prob = F.softmax(expert_logits, dim=-1)
    distances = (normal_prob.unsqueeze(1) - expert_prob).abs().sum(dim=-1)

    layer0_logits = expert_logits[:, 0, :]
    collaborative = normal_logits + alpha_pos * layer0_logits
    contrastive = (1 + alpha_neg) * normal_logits - alpha_neg * layer0_logits
    layer0_is_contrastive = distances[:, 0] >= gamma
    layer0_only = torch.where(layer0_is_contrastive.unsqueeze(-1), contrastive, collaborative)

    # Expert 0 is exact original ONLY. Consensus is formed only from the
    # independent soft layer-local experts that follow it.
    consensus_experts = expert_logits[:, 1:, :] if expert_logits.shape[1] > 1 else expert_logits
    consensus_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
    residuals = (
        F.log_softmax(normal_logits, dim=-1).unsqueeze(1)
        - F.log_softmax(consensus_experts, dim=-1)
    )
    median_residual = residuals.median(dim=1).values
    sign_agreement = (
        torch.sign(residuals) == torch.sign(median_residual).unsqueeze(1)
    ).float().mean(dim=1)
    agreed = (sign_agreement >= consensus_min) & (median_residual.abs() >= residual_min)

    step_gate = (
        consensus_distances.median(dim=1).values / max(float(gamma), 1e-6)
    ).clamp(0, 1).unsqueeze(-1)
    token_gate = agreed.to(normal_logits.dtype) * step_gate
    correction = median_residual.clamp(-residual_clip, residual_clip)
    consensus = normal_logits + consensus_strength * correction
    result = layer0_only + token_gate * (consensus - layer0_only)
    diagnostics = {
        "distances": distances,
        "layer0_is_contrastive": layer0_is_contrastive,
        "step_gate": step_gate.squeeze(-1),
        "activation_ratio": agreed.float().mean(dim=-1),
    }
    return result, diagnostics
