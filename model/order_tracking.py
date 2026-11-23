"""Order-domain spectrum from aligned shaft-angle or speed measurements."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def order_spectrum(
    signals: Tensor,
    lengths: Tensor,
    channel_mask: Tensor,
    window_seconds: Tensor,
    *,
    rpm_traces: Optional[Tensor] = None,
    rpm_trace_valid: Optional[Tensor] = None,
    shaft_angles: Optional[Tensor] = None,
    shaft_angle_valid: Optional[Tensor] = None,
    nominal_rpm: Optional[Tensor] = None,
    max_order: int = 32,
    points_per_revolution: int = 128,
    max_points: int = 8192,
) -> tuple[Tensor, Tensor]:
    """Return log-magnitude integer-order spectra and a per-sample validity mask.

    ``shaft_angles`` are radians. ``rpm_traces`` are point-aligned RPM values.
    A scalar ``nominal_rpm`` is used only as a constant-speed approximation.
    Missing or invalid speed metadata produces zero features and a false mask.
    """

    if signals.ndim != 3 or lengths.shape != signals.shape[:1]:
        raise ValueError("signals must be (batch, channels, time) with one length per sample")
    batch, _, padded_length = signals.shape
    features = torch.zeros((batch, max_order), device=signals.device, dtype=torch.float32)
    valid = torch.zeros((batch,), device=signals.device, dtype=torch.bool)
    orders = torch.arange(1, max_order + 1, device=signals.device, dtype=torch.float32)

    for row in range(batch):
        length = int(lengths[row].item())
        if length < 2 or length > padded_length:
            continue
        active_channels = channel_mask[row].bool()
        if not bool(active_channels.any()):
            continue
        seconds = float(window_seconds[row].item())
        if not math.isfinite(seconds) or seconds <= 0:
            continue

        phase: Optional[Tensor] = None
        if shaft_angles is not None and shaft_angle_valid is not None and bool(shaft_angle_valid[row]):
            candidate = shaft_angles[row, :length].float()
            if bool(torch.isfinite(candidate).all()):
                steps = candidate[1:] - candidate[:-1]
                steps = torch.where(steps < -math.pi, steps + 2.0 * math.pi, steps)
                steps = torch.where(steps > math.pi, steps - 2.0 * math.pi, steps)
                phase = torch.cat((candidate.new_zeros(1), steps.cumsum(dim=0))) / (2.0 * math.pi)
        if phase is None and rpm_traces is not None and rpm_trace_valid is not None and bool(rpm_trace_valid[row]):
            speed = rpm_traces[row, :length].float()
            if bool(torch.isfinite(speed).all()) and bool((speed >= 0).all()):
                steps = (speed[:-1] + speed[1:]) * (0.5 * seconds / ((length - 1) * 60.0))
                phase = torch.cat((speed.new_zeros(1), steps.cumsum(dim=0)))
        if phase is None and nominal_rpm is not None:
            rpm = float(nominal_rpm[row].item())
            if math.isfinite(rpm) and rpm > 0:
                phase = torch.linspace(0.0, rpm * seconds / 60.0, length, device=signals.device)
        if phase is None or not bool(torch.isfinite(phase).all()):
            continue

        phase = torch.cummax(phase, dim=0).values
        revolutions = float((phase[-1] - phase[0]).item())
        point_count = int(round(revolutions * points_per_revolution))
        if revolutions <= 0.25 or point_count < 32 or point_count > max_points:
            continue
        query = torch.linspace(phase[0], phase[-1], point_count, device=signals.device)
        right = torch.searchsorted(phase.contiguous(), query.contiguous()).clamp(1, length - 1)
        left = right - 1
        fraction = (query - phase[left]) / (phase[right] - phase[left]).clamp_min(1e-8)
        source = signals[row, active_channels, :length].float()
        angular_signal = source[:, left] + (source[:, right] - source[:, left]) * fraction
        spectrum = torch.fft.rfft(angular_signal, dim=-1).abs() / point_count
        bin_position = orders * revolutions
        if bool((bin_position >= spectrum.shape[-1] - 1).any()):
            continue
        lower = bin_position.floor().long()
        upper = lower + 1
        weight = bin_position - lower
        amplitudes = spectrum[:, lower] * (1.0 - weight) + spectrum[:, upper] * weight
        features[row] = amplitudes.mean(dim=0).log1p()
        valid[row] = True
    return features, valid
