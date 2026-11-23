"""External model adapters. Architecture changes are recorded in the guide."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .registry import GENERIC_MODELS


class GenericBaseline(nn.Module):
    """Reuse the existing baseline model and its channel-averaging interface."""
    def __init__(self, name, task, classes):
        super().__init__()
        from ..comparisons.architectures import BaselineModel
        self.network = BaselineModel(name, task, num_classes=classes, points=2048, width=64, layers=2)
        self.task = task

    def forward(self, x):
        lengths = torch.full((len(x),), x.shape[-1], device=x.device, dtype=torch.long)
        mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        output = self.network(x, lengths, mask)
        return output["logits"] if self.task == "fd" else output["prediction"]


class WDCNN(nn.Module):
    """Layer-for-layer PyTorch port of the supplied main.py, input length 2048."""
    def __init__(self, classes):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = [1, 16, 32, 64, 64, 64]
        for i in range(5):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels[i], channels[i + 1], 64 if i == 0 else 3,
                          stride=16 if i == 0 else 1, padding=1 if i in (1, 2, 3) else 0),
                nn.BatchNorm1d(channels[i + 1], eps=1e-3, momentum=0.01),
                nn.ReLU(), nn.MaxPool1d(2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(192, 100), nn.ReLU(), nn.Linear(100, classes))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)  # Keras default Glorot uniform.
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.interpolate(x, size=2048, mode="linear", align_corners=False)
        x = F.pad(x, (24, 24))  # Keras SAME for k=64, stride=16, length=2048.
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def regularization(self):
        # Supplied Keras code regularizes all kernels, including output Dense.
        return 1e-4 * sum(m.weight.square().sum() for m in self.modules() if isinstance(m, (nn.Conv1d, nn.Linear)))


class UniFault(nn.Module):
    def __init__(self, classes):
        super().__init__()
        from .vendor.unifault.model.model import Transformer_bkbone
        self.network = Transformer_bkbone(SimpleNamespace(seq_len=2048, patch_size=64,
            embed_dim=128, dropout=0.3, heads=4, depth=4, num_channels=1, num_classes=classes))

    def forward(self, x):
        x = F.interpolate(x, size=2048, mode="linear", align_corners=False)
        return self.network.head(self.network(x).mean(1))


class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        from .vendor.cnn_bilstm.rul_model import LSTM
        self.network = LSTM(160, 64, 2, "relu")

    def forward(self, x):
        # Original convolutions expect (1, channels=2, time=256) as a 2D image.
        batch, history, channels, points = x.shape
        feature = self.network.get_feature(x.reshape(batch * history, 1, channels, points))
        feature = feature.reshape(batch, history, 160)
        # Explicit history dimension prevents the original accidental recurrence
        # across unrelated minibatch samples.
        hidden, _ = self.network.get_time_feature(feature)
        return self.network.fc(hidden[:, -1, :].reshape(batch, 1, 128)).reshape(batch)


class DARUL(nn.Module):
    def __init__(self, history, features=32):
        super().__init__()
        from .vendor.da_rul.Experiment.DAT_Experiment import DAT_Experiment
        self.network = DAT_Experiment(cell_type="slstm", sequence_len=history, feature_num=features,
            hidden_dim=32, lstm_dim=16, fc_layer_dim=32, rnn_num_layers=1, output_dim=1,
            fc_activation="gelu", attention_type="deg_attention", feature_head_num=2,
            sequence_head_num=2, fc_dropout=0.25, bidirectional=False)

    def forward(self, x):
        # Exponential sLSTM gates and cdist run in FP32 for numerical stability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.network(x.float()).reshape(-1)


class DualMixer(nn.Module):
    def __init__(self, history, features=32):
        super().__init__()
        from .vendor.dual_mixer.models.RULPrediction.MLPMixer import DualMLPMixer
        from .vendor.dual_mixer.models.RULPrediction.ContrastiveModules import MSEContrastiveLoss
        self.network = DualMLPMixer(window_size=history, in_features=features, hidden_dim=32,
                                    num_layers=6, dropout=0, device="cpu", filter_size=0)
        self.criterion = MSEContrastiveLoss("InfoNCE")

    def forward(self, x):
        return self.network(x).reshape(-1)

    def training_loss(self, x, y):
        # Keep the supplied noise-positive, weighted negative and joint MSE loss.
        prediction, anchor, positive, negative, weights = self.network(x, y)
        loss = self.criterion(prediction, y, anchor, positive, negative, weights)
        return loss, prediction[:, 0]


class CommunityMoment(nn.Module):
    def __init__(self, task, classes, hf_backbone="google/flan-t5-small", hf_init="pretrained"):
        super().__init__()
        from .vendor.moment_community.src.mymoment.model.moment import MOMENT
        from transformers import T5Config, T5EncoderModel
        # FLAN and original T5 of the same size have different FFNs, heads and
        # layer counts. Preserve the exact public config when resuming offline.
        config_path = Path(__file__).parent / "hf_configs" / (hf_backbone.replace("/", "__") + ".json")
        encoder_config = T5Config.from_json_file(str(config_path))

        class ConfiguredMOMENT(MOMENT):
            def _build_hf_encoder(self, name):
                if hf_init == "random":
                    return T5EncoderModel(encoder_config)
                return super()._build_hf_encoder(name)

        self.network = ConfiguredMOMENT(dict(task_name="classification", n_class=classes if task == "fd" else 1,
            seq_len=2048, patch_len=256, patch_stride_len=256, n_channels=1 if task == "fd" else 2,
            transformer_backbone=hf_backbone, d_model=encoder_config.d_model, dropout=0.1))

    def forward(self, x):
        x = F.interpolate(x, size=2048, mode="linear", align_corners=False)
        return self.network(x).logits


def spectrogram(x):
    """Torch equivalent of supplied librosa STFT/power_to_db + 64x64 resize."""
    x = x[:, 0].float()
    spectrum = torch.stft(x, n_fft=512, hop_length=512,
                          window=torch.hann_window(512, device=x.device), return_complex=True).abs().square()
    db = 10 * torch.log10(spectrum.clamp_min(1e-10))
    db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80)
    return F.interpolate(db[:, None], size=(64, 64), mode="bilinear", align_corners=False)


class SCMamba(nn.Module):
    def __init__(self, classes, scan_backend="cuda"):
        super().__init__()
        from .vendor.sc_mamba.net import mamba
        if scan_backend == "reference":
            from .compat import selective_scan_reference
            mamba.selective_scan_fn = selective_scan_reference
        elif scan_backend == "cuda":
            # Do not silently replace an unavailable Mamba kernel with a proxy model.
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            mamba.selective_scan_fn = selective_scan_fn
        from .vendor.sc_mamba.net.new_proposed import MainNet
        self.network = MainNet(num_classes=classes)
        self.classes = classes
        self.register_buffer("covariances", torch.empty(0), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        # The supplied SB explicitly initializes this BN in eval mode; retain
        # that intent for the pooled 1x1 singleton supports in a 1-shot run.
        self.network.SB.fc[1].eval()
        return self

    def encode(self, waves):
        image = spectrogram(waves)
        with torch.autocast(device_type=waves.device.type, enabled=False):
            upper = self.network.upper(self.network.features1(image))
            lower = self.network.lower(self.network.features2(image))
            return self.network.SB(upper, lower)[0]

    @staticmethod
    def covariance(features):
        channels = features.shape[1]
        flattened = features.permute(1, 0, 2, 3).reshape(channels, -1)
        centered = flattened - flattened.mean(-1, keepdim=True)
        return (centered @ centered.T) / max(1, flattened.shape[-1] - 1)

    def predict_features(self, features, covariances):
        q = F.normalize(features.flatten(2), dim=2, eps=1e-8)
        similarity = torch.einsum("bcp,kcd,bdp->bkp", q, covariances, q)
        return self.network.classifier1(similarity.reshape(len(q), 1, -1)).squeeze(1)

    def forward(self, x):
        if not self.covariances.numel():
            raise RuntimeError("Build training-only support covariance before SC-Mamba inference")
        return self.predict_features(self.encode(x), self.covariances)

    def support_loss(self, x, y, support_x, support_y):
        support = self.encode(support_x)
        covariance = torch.stack([self.covariance(support[support_y == k]) for k in range(self.classes)])
        prediction = self.predict_features(self.encode(x), covariance)
        return F.cross_entropy(prediction, y), prediction

    @torch.no_grad()
    def build_support(self, loader, device):
        """Exact covariance sufficient statistics over ALL labeled training supports."""
        self.eval()
        sums, products, counts = {}, {}, {}
        for x, labels, _ in loader:
            features = self.encode(x.to(device)).double()
            for k in labels.unique().tolist():
                f = features[labels.to(device) == k].permute(1, 0, 2, 3).flatten(1)
                sums[k] = sums.get(k, 0) + f.sum(-1)
                products[k] = products.get(k, 0) + f @ f.T
                counts[k] = counts.get(k, 0) + f.shape[-1]
        self.covariances = torch.stack([(products[k] - sums[k][:, None] * sums[k][None, :] / counts[k]) /
                                        max(1, counts[k] - 1) for k in range(self.classes)]).float()


def build_model(name, task, classes=1, history=30, scan_backend="cuda", hf_backbone="google/flan-t5-small", hf_init="pretrained"):
    if name in GENERIC_MODELS:
        return GenericBaseline(name, task, classes)
    if name == "wdcnn":
        return WDCNN(classes)
    if name == "unifault":
        return UniFault(classes)
    if name == "cnn_bilstm":
        return CNNBiLSTM()
    if name == "da_rul":
        return DARUL(history)
    if name == "dual_mixer":
        return DualMixer(history)
    if name == "sc_mamba":
        return SCMamba(classes, scan_backend)
    if name == "moment_community":
        return CommunityMoment(task, classes, hf_backbone, hf_init)
    raise ValueError(name)
