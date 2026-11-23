"""Self-supervised and downstream PHM loss functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def _masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def spectral_masked_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    spectral_weight: float = 0.5,
) -> Tensor:
    """Masked time MSE plus log-magnitude FFT MSE (SpecMAE)."""

    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape (batch, tokens)")
    selected = mask.to(dtype=torch.bool)
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    pred = prediction[selected]
    truth = target[selected]
    time_loss = F.mse_loss(pred, truth)
    pred_spectrum = torch.fft.rfft(pred.float(), dim=-1).abs().clamp_min(1e-8).log()
    truth_spectrum = torch.fft.rfft(truth.float(), dim=-1).abs().clamp_min(1e-8).log()
    frequency_loss = F.mse_loss(pred_spectrum, truth_spectrum)
    return time_loss + float(spectral_weight) * frequency_loss


def monotonicity_loss(
    current_health: Tensor,
    next_health: Tensor,
    current_token_mask: Tensor,
    next_token_mask: Tensor,
    pair_valid: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Penalize health increasing between ordered windows of one trajectory."""

    if current_health.ndim != 2 or next_health.ndim != 2:
        raise ValueError("health tensors must have shape (batch, tokens)")
    if current_token_mask.shape != current_health.shape or next_token_mask.shape != next_health.shape:
        raise ValueError("token masks must match their health tensors")
    if current_health.shape[0] != next_health.shape[0] or pair_valid.shape != current_health.shape[:1]:
        raise ValueError("paired health tensors and pair_valid must have the same batch size")

    current_weights = current_token_mask.to(current_health.dtype)
    next_weights = next_token_mask.to(next_health.dtype)
    current_hi = (current_health * current_weights).sum(dim=1) / current_weights.sum(dim=1).clamp_min(1)
    next_hi = (next_health * next_weights).sum(dim=1) / next_weights.sum(dim=1).clamp_min(1)
    valid = pair_valid.bool() & current_token_mask.any(dim=1) & next_token_mask.any(dim=1)
    penalty = F.relu(next_hi - current_hi + float(margin))
    return (penalty * valid.to(penalty.dtype)).sum() / valid.sum().clamp_min(1)


def info_nce_loss(anchor: Tensor, positive: Tensor, temperature: float = 0.1) -> Tensor:
    """Symmetric in-batch InfoNCE used for cross-condition augmentation."""

    if anchor.ndim != 2 or positive.ndim != 2 or anchor.shape != positive.shape:
        raise ValueError("anchor and positive must both have shape (batch, embedding)")
    batch = anchor.shape[0]
    if batch < 2:
        return anchor.sum() * 0.0
    temperature = max(float(temperature), 1e-5)
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.transpose(0, 1) / temperature
    labels = torch.arange(batch, device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


@dataclass
class PretrainingLossConfig:
    spectral_weight: float = 0.5
    monotonicity_weight: float = 0.1
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    monotonicity_margin: float = 0.0


def pretraining_loss(
    outputs: Dict[str, Tensor],
    positive_outputs: Optional[Dict[str, Tensor]] = None,
    config: Optional[PretrainingLossConfig] = None,
    *,
    next_outputs: Optional[Dict[str, Tensor]] = None,
    pair_valid: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Compute the three-term physics-guided pretraining objective."""

    config = config or PretrainingLossConfig()
    reconstruction = spectral_masked_reconstruction_loss(
        outputs["reconstruction"],
        outputs["target_patches"],
        outputs["reconstruction_mask"],
        spectral_weight=config.spectral_weight,
    )
    if next_outputs is not None and (pair_valid is None or positive_outputs is None):
        raise ValueError("next_outputs requires pair_valid and an unmasked positive_outputs view")
    if next_outputs is not None:
        monotonicity = monotonicity_loss(
            positive_outputs["health"],
            next_outputs["health"],
            positive_outputs["token_mask"],
            next_outputs["token_mask"],
            pair_valid,
            margin=config.monotonicity_margin,
        )
    else:
        monotonicity = outputs["health"].sum() * 0.0
    if positive_outputs is None:
        contrastive = outputs["pooled"].sum() * 0.0
    else:
        contrastive = info_nce_loss(
            outputs["pooled"], positive_outputs["pooled"], temperature=config.contrastive_temperature
        )
    total = (
        reconstruction
        + float(config.monotonicity_weight) * monotonicity
        + float(config.contrastive_weight) * contrastive
    )
    return {
        "loss": total,
        "specmae": reconstruction.detach(),
        "monotonicity": monotonicity.detach(),
        "contrastive": contrastive.detach(),
    }


def fd_loss(logits: Tensor, labels: Tensor, label_smoothing: float = 0.0) -> Tensor:
    return F.cross_entropy(logits, labels.long(), label_smoothing=float(label_smoothing))


def cross_group_contrastive_loss(features, labels, group_ids, temperature=0.1):
    """Same-label different-group positives; only other labels are negatives.

    Same-group same-class peers are excluded from the denominator, never negatives.
    Each valid anchor needs a positive and a negative; returns (loss, anchor count).
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if features.ndim != 2 or len(features) != len(labels) or len(group_ids) != len(labels):
        raise ValueError("Features, labels and group IDs must align")
    with torch.autocast(device_type=features.device.type, enabled=False):
        unit = F.normalize(features.float(), dim=-1)
        mapping = {g: i for i, g in enumerate(sorted(set(group_ids)))}
        groups = torch.tensor([mapping[g] for g in group_ids], device=features.device)
        same_label = labels[:, None] == labels[None, :]
        positive = same_label & (groups[:, None] != groups[None, :])
        negative = ~same_label
        valid = positive.any(1) & negative.any(1)
        count = int(valid.sum())
        if not count:
            return features.float().sum() * 0., 0
        logits = (unit @ unit.T) / temperature
        selected = logits[valid]
        allowed = (positive | negative)[valid]
        log_prob = selected - torch.logsumexp(selected.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True)
        mask = positive[valid]
        loss = -torch.where(mask, log_prob, 0.).sum(1) / mask.sum(1)
        return loss.mean(), count


def weibull_nll_loss(
    target: Tensor,
    shape: Tensor,
    scale: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Exact-observation Weibull NLL; zero RUL is evaluated at ``eps``.

    Targets and scale must use the same units.  A linear continuation of the
    cumulative hazard beyond 1e6 keeps extreme initial predictions finite
    and preserves a corrective gradient.
    """

    if target.shape != shape.shape or target.shape != scale.shape:
        raise ValueError("target, shape, and scale must have identical shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("Weibull targets must be finite and nonnegative")

    observation = target.float().clamp_min(eps)
    concentration = shape.float().clamp_min(eps)
    duration = scale.float().clamp_min(eps)
    log_ratio = observation.log() - duration.log()
    log_hazard = concentration * log_ratio
    cutoff = math.log(1e6)
    hazard = torch.exp(log_hazard.clamp(max=cutoff))
    hazard = torch.where(log_hazard > cutoff, hazard * (1.0 + log_hazard - cutoff), hazard)
    log_density = concentration.log() - duration.log() + (concentration - 1.0) * log_ratio - hazard
    return -log_density.mean()


def rul_loss(
    prediction: Tensor,
    target: Tensor,
    kind: str = "smooth_l1",
    *,
    weibull_shape: Optional[Tensor] = None,
    weibull_scale: Optional[Tensor] = None,
    weibull_weight: float = 0.0,
) -> Tensor:
    """Point regression plus an optional exact-target Weibull likelihood."""

    kind = kind.lower()
    if kind == "mse":
        point_loss = F.mse_loss(prediction.float(), target.float())
    elif kind in {"smooth_l1", "huber"}:
        point_loss = F.smooth_l1_loss(prediction.float(), target.float())
    elif kind == "mae":
        point_loss = F.l1_loss(prediction.float(), target.float())
    else:
        raise ValueError("RUL loss must be smooth_l1, mse, or mae")
    if not math.isfinite(weibull_weight) or weibull_weight < 0:
        raise ValueError("weibull_weight must be finite and nonnegative")
    if weibull_weight == 0:
        return point_loss
    if weibull_shape is None or weibull_scale is None:
        raise ValueError("weibull_shape and weibull_scale are required when weibull_weight > 0")
    return point_loss + weibull_weight * weibull_nll_loss(target, weibull_shape, weibull_scale)

