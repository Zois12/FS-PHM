"""Label-free signal objectives for foundation and generic comparison encoders.

This is an explicit PHM-corpus adaptation, not a claim that the supplied
UniFault repository published this complete pretraining recipe.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .adapters import build_model
from .registry import GENERIC_MODELS

SSL_MODELS = ("unifault", "moment_community", *GENERIC_MODELS)


def patch_mask(x, patch, ratio, policy="random"):
    count = x.shape[-1] // patch
    if count < 2:
        raise ValueError("Pretraining requires at least two patches")
    number = min(count - 1, max(1, round(count * ratio)))
    masked = torch.zeros(len(x), count, dtype=torch.bool, device=x.device)
    if policy == "last":
        masked[:, -number:] = True
    else:
        order = torch.rand(len(x), count, device=x.device).argsort(-1)[:, :number]
        masked.scatter_(1, order, True)
    return masked.repeat_interleave(patch, -1)[:, None]


class MaskedPretrainer(nn.Module):
    def __init__(self, name, hf_backbone="google/flan-t5-small", hf_init="pretrained"):
        super().__init__()
        if name not in SSL_MODELS:
            raise ValueError(f"No label-free signal objective configured for {name}")
        self.name = name
        self.backbone = build_model(name, "fd", 2, hf_backbone=hf_backbone, hf_init=hf_init)
        if name == "moment_community":
            from .vendor.moment_community.src.mymoment.model.moment import PretrainHead
            self.ssl_decoder = PretrainHead(self.backbone.network.d_model, 256)

    @property
    def patch_size(self):
        return 256 if self.name == "moment_community" else 32 if self.name in GENERIC_MODELS else 64

    def reconstruct(self, x, masked):
        """Mask actual encoder inputs; hidden targets never enter the encoder."""
        network = self.backbone.network
        if self.name == "moment_community":
            visible = (~masked[:, 0]).to(x.dtype)
            # Preserve supplied RevIN, patch embedding, T5 and reconstruction head.
            target = network.normalizer(x, mask=visible, mode="norm")
            hidden = network.patch_embedding(network.tokenizer(target.masked_fill(masked, 0)))
            batch, channels, patches, width = hidden.shape
            output = network.encoder(inputs_embeds=hidden.reshape(batch * channels, patches, width),
                                     attention_mask=torch.ones(batch * channels, patches, device=x.device, dtype=torch.long))
            hidden = output.last_hidden_state.reshape(batch, channels, patches, width)
            return self.ssl_decoder(hidden), target, hidden.mean((1, 2)), masked
        hidden_input = x.masked_fill(masked, 0)
        if self.name == "unifault":
            hidden = network(hidden_input)
            prediction = network.pretrain_head(hidden).flatten(1)[:, None]
            return prediction, x, hidden.mean(1), masked
        # Existing generic baseline has a 64-bin reconstruction decoder.
        hidden = network.encoder(hidden_input)
        target = F.adaptive_avg_pool1d(x, 64)
        coarse_mask = F.adaptive_max_pool1d(masked.float(), 64).bool()
        return network.decoder(hidden)[:, None], target, hidden, coarse_mask

    def forward(self, x, mask_ratio=0.4, mask_policy="random", spectral_weight=0.5,
                contrastive_weight=0.1, temperature=0.1):
        masked = patch_mask(x, self.patch_size, mask_ratio, mask_policy)
        prediction, target, feature, loss_mask = self.reconstruct(x, masked)
        error = (prediction.float() - target.float()).square()
        time_loss = error.masked_select(loss_mask.expand_as(error)).mean()
        # Visible points come from the target, so spectral loss constrains only
        # missing regions rather than asking the model to copy visible context.
        combined = torch.where(loss_mask, prediction.float(), target.float())
        spectral_loss = F.mse_loss(torch.fft.rfft(combined).abs().log1p(),
                                   torch.fft.rfft(target.float()).abs().log1p())
        contrastive = feature.float().sum() * 0
        if contrastive_weight and len(x) > 1:
            view = x * (1 + 0.1 * torch.randn(len(x), 1, 1, device=x.device)) + 0.02 * torch.randn_like(x)
            other_mask = patch_mask(view, self.patch_size, mask_ratio, mask_policy)
            _, _, other, _ = self.reconstruct(view, other_mask)
            z = F.normalize(torch.cat((feature.float(), other.float())), dim=-1)
            scores = z @ z.T / temperature
            scores.fill_diagonal_(-torch.inf)
            labels = (torch.arange(2 * len(x), device=x.device) + len(x)) % (2 * len(x))
            contrastive = F.cross_entropy(scores, labels)
        loss = time_loss + spectral_weight * spectral_loss + contrastive_weight * contrastive
        return loss, {"time_loss": time_loss.detach(), "spectral_loss": spectral_loss.detach(),
                      "contrastive_loss": contrastive.detach()}
