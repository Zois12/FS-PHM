"""Explicit task scope and provenance for the supplied comparison models."""
from dataclasses import dataclass

REGIMES = ("1shot", "4shot", "8shot", "16shot", "10percent", "full")
GENERIC_MODELS = ("cnn_lstm", "transformer", "resnet18")


@dataclass(frozen=True)
class ModelSpec:
    tasks: tuple
    label: str
    source: str
    lr: float
    note: str


MODELS = {
    "sc_mamba": ModelSpec(("fd",), "SC-MambaFew (fixed-support adaptation)", "sc_mamba", 1e-3,
        "Original feature/SSM/fusion/covariance modules; fixed labeled support replaces original episodic label budget."),
    "unifault": ModelSpec(("fd",), "UniFault-Tiny", "unifault", 3e-4,
        "Original 4-layer, 128-wide backbone; scratch unless an explicit external checkpoint is supplied."),
    "wdcnn": ModelSpec(("fd",), "WDCNN (PyTorch port)", "wdcnn", 1e-3,
        "Layer-for-layer port of supplied Keras five-convolution architecture, including dense-100 and kernel L2."),
    "moment_community": ModelSpec(("fd", "rul"), "MOMENT/T5 community implementation (not original RmGPT)", "moment_community", 1e-4,
        "Supplied phm_RmGPT archive actually implements MOMENT; RUL adds a scalar downstream head."),
    "da_rul": ModelSpec(("rul",), "DA-RUL (XJTU feature-history adaptation)", "da_rul", 1e-3,
        "Original degradation attention + sLSTM; sensor histories adapted to train-standardized vibration descriptors."),
    "cnn_bilstm": ModelSpec(("rul",), "CNN-BiLSTM supervised backbone", "cnn_bilstm", 1e-3,
        "Original convolution and recurrent modules; explicit within-bearing history, no original transductive domain stage."),
    "dual_mixer": ModelSpec(("rul",), "Dual-Mixer + FSGRI (XJTU adaptation)", "dual_mixer", 1e-3,
        "Original progressive dual mixer and weighted contrastive loss; negatives come only from training trajectories."),
    "cnn_lstm": ModelSpec(("fd", "rul"), "CNN-LSTM (generic 1D baseline)", "local_generic", 1e-3,
        "Existing CNN front end + bidirectional LSTM, width 64, 2048 points; RUL averages the two train-standardized vibration channels."),
    "transformer": ModelSpec(("fd", "rul"), "Transformer (generic 1D baseline)", "local_generic", 1e-3,
        "Existing 2-layer, width-64, 4-head Transformer with convolutional patches and sinusoidal positions; 2048 points."),
    "resnet18": ModelSpec(("fd", "rul"), "ResNet18 (generic 1D baseline)", "local_generic", 1e-3,
        "Existing 1D ResNet18, width 64, eight residual blocks; 2048 points, channel averaging as in existing baseline interface."),
}


def normalize_regime(value):
    value = value.lower().replace(" ", "").replace("_", "")
    return {"10pct": "10percent", "10%": "10percent", "10%data": "10percent",
            "fulldata": "full"}.get(value, value)


def trials(models, datasets, regimes):
    for name in models:
        for dataset in datasets:
            task = "rul" if dataset == "XJTU-SY" else "fd"
            if task in MODELS[name].tasks:
                for regime in (regimes if task == "fd" else ("full",)):
                    yield {"model": name, "task": task, "dataset": dataset, "regime": regime}
