"""Detached, per-sample gating at the original two-branch logit sum only."""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from .power_spectrum import PowerFusionHead

MODE = "confidence_gate_v1"


class ConfidenceGatedPowerHead(PowerFusionHead):
    """Reuse both original branches and all their initialized parameter objects.

    ``parts`` already multiplies spectral logits by ``logit_scale``. The gate
    therefore applies a ratio of 1 or alpha_low/logit_scale, never a second 10x.
    There are no new trainable parameters or random initialization draws.
    """

    def __init__(self, original, threshold=.02, alpha_low=.01, enabled=True):
        nn.Module.__init__(self)
        high = float(original.logit_scale)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Gate threshold must be finite and nonnegative")
        if not math.isfinite(alpha_low) or not 0 <= alpha_low <= high:
            raise ValueError("Gate alpha-low must be finite and between 0 and power-logit-scale")
        self.backbone_head = original.backbone_head
        self.embed_dim = original.embed_dim
        self.weight = original.weight
        self.bias = original.bias
        self.register_buffer("logit_scale", original.logit_scale)
        self.threshold, self.alpha_low, self.alpha_high = threshold, alpha_low, high
        self.enabled = bool(enabled)
        # Extra state key makes legacy evaluators fail strict loading, preventing
        # them from silently reporting a constant-alpha result for this model.
        self.register_buffer("confidence_gate_settings", torch.tensor(
            [1., threshold, alpha_low, high, float(enabled)], dtype=torch.float64))
        self.reset_gate_stats()
        self.last_confidence = self.last_alpha = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "confidence_gate_settings"
        saved = state_dict.get(key)
        if saved is not None and not torch.equal(saved.cpu(), self.confidence_gate_settings.cpu()):
            error_msgs.append("Confidence gate settings differ from checkpoint; restore its saved gate configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def gate(self, logits_base):
        # Match torch.std's sample standard deviation (correction=1).
        # FP32 also keeps this calculation stable under BF16/FP16 autocast.
        with torch.autocast(device_type=logits_base.device.type, enabled=False):
            prob_base = torch.softmax(logits_base.detach().float(), dim=-1)
            confidence = prob_base.std(dim=-1, keepdim=True, unbiased=True)
            high = confidence >= self.threshold if self.enabled else torch.ones_like(confidence, dtype=torch.bool)
            alpha = torch.where(high, self.alpha_high, self.alpha_low)
        self.last_confidence, self.last_alpha = confidence, alpha
        self._record_stats(confidence, alpha)
        return alpha

    def forward(self, features):
        base, power = self.parts(features)  # Exactly one forward of each original branch.
        alpha = self.gate(base)
        # Gradients still reach both branches; only the alpha calculation is detached.
        return base + (alpha / self.alpha_high) * power

    def reset_gate_stats(self):
        self._gate_totals = self._gate_extrema = None

    @torch.no_grad()
    def _record_stats(self, confidence, alpha):
        c = confidence.flatten().double()
        if not c.numel():
            return
        totals = torch.stack((c.new_tensor(c.numel()), c.sum(), c.square().sum(), alpha.double().sum()))
        extrema = torch.stack((c.min(), -c.max()))
        if self._gate_totals is None:
            self._gate_totals, self._gate_extrema = totals, extrema
        else:
            # Diagnostics may run in inference_mode, then a caller may reuse the
            # same head in ordinary/no_grad mode. Out-of-place accumulation also
            # accepts a tensor created by the earlier inference context.
            self._gate_totals = self._gate_totals + totals
            self._gate_extrema = torch.minimum(self._gate_extrema, extrema)

    @torch.no_grad()
    def gate_stats(self, context=None):
        # A rank with an empty evaluation shard must still join reductions.
        device = self.logit_scale.device
        totals = self._gate_totals.clone() if self._gate_totals is not None else torch.zeros(4, dtype=torch.float64, device=device)
        extrema = self._gate_extrema.clone() if self._gate_extrema is not None else torch.full((2,), float("inf"), dtype=torch.float64, device=device)
        if context is not None and context.is_distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(extrema, op=dist.ReduceOp.MIN)
        n, total, square, alpha_sum = totals.cpu().tolist()
        lo, negative_hi = extrema.cpu().tolist()
        mean = total / n if n else None
        return {
            "gate_confidence_mean": mean,
            "gate_confidence_std": math.sqrt(max(0., square / n - mean * mean)) if n else None,
            "gate_confidence_min": lo if n else None,
            "gate_confidence_max": -negative_hi if n else None,
            "gate_alpha_mean": alpha_sum / n if n else None,
            "gate_threshold": self.threshold, "gate_alpha_high": self.alpha_high,
            "gate_alpha_low": self.alpha_low, "gate_enabled": self.enabled,
        }


def gate_head(model):
    model = getattr(model, "module", model)
    head = getattr(model, "head", None)
    return head if isinstance(head, ConfidenceGatedPowerHead) else None
