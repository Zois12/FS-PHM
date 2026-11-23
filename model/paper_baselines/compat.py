"""Numerically equivalent utilities used by the supplied external source."""
import torch
from torch import nn
from torch.nn import functional as F


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def selective_scan_reference(u, delta, A, B, C, D=None, z=None,
                             delta_bias=None, delta_softplus=False, return_last_state=False):
    """Real-valued selective SSM recurrence; CPU verification, not a model substitute."""
    dtype = u.dtype
    u, delta, A, B, C = [v.float() for v in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    if B.ndim == 3:
        B, C = B[:, None], C[:, None]
    groups = B.shape[1]
    channels = u.shape[1]
    state = u.new_zeros(u.shape[0], channels, A.shape[1])
    out = []
    for i in range(u.shape[-1]):
        bi = B[..., i].repeat_interleave(channels // groups, dim=1)
        ci = C[..., i].repeat_interleave(channels // groups, dim=1)
        dt = delta[..., i, None]
        state = torch.exp(dt * A[None]) * state + dt * bi * u[..., i, None]
        out.append((state * ci).sum(-1))
    result = torch.stack(out, dim=-1)
    if D is not None:
        result = result + D[None, :, None] * u
    if z is not None:
        result = result * F.silu(z)
    result = result.to(dtype)
    return (result, state) if return_last_state else result
