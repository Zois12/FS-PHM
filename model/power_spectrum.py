"""Support-fitted per-channel spectral features and an additive FD logit branch."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_power_checkpoint(saved_args):
    """Accept legacy additive and confidence-gated heads without mixing them."""
    mode = saved_args.get("power_fusion", "logits")
    if mode not in {"logits", "confidence_gate_v1"}:
        raise ValueError("This checkpoint uses a removed experimental power-spectrum mode. "
                         "Use an original power-bypass checkpoint, or start a new run from the foundation checkpoint.")
    if mode == "confidence_gate_v1":
        if not saved_args.get("fd_power_spectrum") or any(
                key not in saved_args for key in ("confidence_gate", "gate_threshold", "gate_alpha_low")):
            raise ValueError("Confidence-gated checkpoint is missing its saved power/gate settings")
        if not isinstance(saved_args["confidence_gate"], bool):
            raise ValueError("Confidence-gated checkpoint must record whether its gate is enabled")


def checkpoint_gate_options(saved_args):
    """Legacy heads have no gate buffer; preserve their original state layout."""
    validate_power_checkpoint(saved_args)
    if saved_args.get("power_fusion", "logits") == "confidence_gate_v1":
        return {key: saved_args[key] for key in ("confidence_gate", "gate_threshold", "gate_alpha_low")}
    return {"confidence_gate": None, "gate_threshold": .02, "gate_alpha_low": .01}


def ridge_coefficients(features, labels, classes, alpha=1.):
    """Fit a centered ridge classifier with intercept from labeled support only."""
    x, labels = features.double(), labels.long()
    if (not math.isfinite(alpha) or alpha <= 0 or x.ndim != 2 or not len(x)
            or not torch.isfinite(x).all() or labels.shape != (len(x),)
            or set(labels.cpu().tolist()) != set(range(classes))):
        raise ValueError("Power ridge initialization requires finite features, positive alpha and support for every class")
    y = F.one_hot(labels, classes).double()
    xm, ym = x.mean(0), y.mean(0)
    centered = x - xm
    if len(x) <= x.shape[1]:
        dual = torch.linalg.solve(centered @ centered.T + alpha * torch.eye(len(x), device=x.device), y - ym)
        weight = centered.T @ dual
    else:
        weight = torch.linalg.solve(centered.T @ centered + alpha * torch.eye(x.shape[1], device=x.device),
                                    centered.T @ (y - ym))
    return weight.T, ym - xm @ weight


class PowerSpectrumNormalizer(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("scale", torch.ones(dimensions))
        self.register_buffer("active", torch.ones(dimensions, dtype=torch.bool))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, support):
        x = support.double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not len(x) or not torch.isfinite(x).all():
            raise ValueError("Invalid power-spectrum support features")
        mean, scale = x.mean(0), x.std(0, unbiased=False)
        active = scale > 1e-10 * mean.abs().clamp_min(1.)
        self.mean.copy_(mean)
        self.scale.copy_(torch.where(active, scale, torch.ones_like(scale)))
        self.active.copy_(active)
        self.fitted.fill_(True)

    def forward(self, features):
        if not bool(self.fitted):
            raise ValueError("Power-spectrum normalization must first be fitted on training support")
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not torch.isfinite(features).all():
            raise ValueError("Invalid power-spectrum feature dimensions/values")
        # Keep support standardization out of AMP; it is not a batch statistic.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.normalize(((features.float() - self.mean) / self.scale) * self.active, dim=-1)


class PowerFusionHead(nn.Module):
    """Original model logits + scale * support-ridge-initialized spectral logits.

    Ridge initializes a trainable linear branch, not a query-fitted classifier.
    The caller concatenates original model features and normalized spectra so
    feature-comparison tools can evaluate this exact saved head.
    """
    def __init__(self, backbone_head, embed_dim, dimensions, classes, logit_scale=10.):
        super().__init__()
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError("Power logit scale must be finite and positive")
        self.backbone_head = backbone_head
        self.embed_dim = embed_dim
        # Zero allocation consumes no RNG, keeping existing foundation/head init.
        self.weight = nn.Parameter(torch.zeros(classes, dimensions))
        self.bias = nn.Parameter(torch.zeros(classes))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

    @torch.no_grad()
    def initialize_ridge(self, features, labels, alpha=1.):
        weight, bias = ridge_coefficients(features, labels, len(self.bias), alpha)
        self.weight.copy_(weight)
        self.bias.copy_(bias)

    def parts(self, features):
        base = self.backbone_head(features[:, :self.embed_dim])
        with torch.autocast(device_type=features.device.type, enabled=False):
            power = self.logit_scale * F.linear(features[:, self.embed_dim:].float(), self.weight, self.bias)
        return base, power

    def forward(self, features):
        base, power = self.parts(features)
        return base + power
