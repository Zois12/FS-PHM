# FS-PHM

**Physical-time-aligned pretraining for rotating machinery fault diagnosis and remaining useful life prediction.**

[简体中文](README-zh.md) · [Complete results](docs/results.md) · [Baseline implementations](docs/baselines.md)

FS-PHM converts vibration windows with different sampling rates and channel counts into physically aligned tokens. A shared fast/slow backbone learns representations for few-shot fault diagnosis (FD) and single-window remaining useful life (RUL) prediction.

![FS-PHM architecture and training stages](docs/assets/architecture.png)

*Architecture figure extracted from the project manuscript. FD uses LoRA and an optional power-spectrum branch. The recommended RUL pipeline freezes the backbone and trains a token-statistics/amplitude fusion regressor. Adaptation is configured separately for the two tasks.*

## Features

- **Physical-time alignment:** typically 0.2 s for FD and 0.1 s for RUL; deformable temporal encoding, FFT and Morlet CWT, with masks for variable lengths and channel counts.
- **Mixed self-supervised pretraining:** masked time/frequency reconstruction, view contrast, and a monotonicity objective on valid RUL trajectory pairs; optional task/dataset-balanced sampling.
- **Few-shot diagnosis:** a fixed test partition, K labeled windows per class or a fraction of the training pool, LoRA, linear/cosine heads, and a support-fitted power-spectrum branch.
- **Single-window RUL:** a frozen foundation encoder; token mean/std/max and same-window amplitude descriptors; bounded regression trained with MAE.
- **Baselines:** CNN-LSTM, Transformer, ResNet18, WDCNN, UniFault, SC-MambaFew, MOMENT/T5, DA-RUL, CNN-BiLSTM, Dual-Mixer, and a separate paper-derived RmGPT implementation.

## Layout

```text
FS-PHM/
├── README.md                       # Chinese
├── README.en.md                    # English
├── requirements.txt
├── model/
│   ├── architecture.py             # Tokenizer, fast/slow backbone, task wrappers
│   ├── data.py                     # H5 indexing and variable-length batches
│   ├── train_pretrain.py           # Single-GPU / DDP pretraining
│   ├── train_finetune.py           # FD few-shot / fractional / full-pool adaptation
│   ├── train_rul_frozen.py         # Recommended frozen-foundation RUL training
│   ├── rul_frozen.py               # RUL normalization and fusion head
│   ├── train_rul_full.py           # General RUL LoRA / linear / full adaptation
│   ├── train_fd_full.py            # General full-pool FD training
│   ├── make_fd_split.py            # Fixed train-pool/test partitions
│   ├── make_few_shot.py            # K-shot / fractional supports
│   ├── comparisons/architectures.py # Three generic baseline networks
│   └── paper_baselines/
│       ├── pretrain.py            # Label-free baseline pretraining
│       ├── runner.py              # Baseline training/evaluation
│       ├── rmgpt/                 # Architecture, pretraining, prompt/full adaptation
│       ├── vendor/                # Required upstream modules and provenance
│       └── hf_configs/            # MOMENT/T5 architecture configs
├── docs/                           # Figures, results and source notes
└── tests/                          # Core CPU checks
```

The export retains required normalization, splitting, logging, losses and metrics. Datasets, weights, historical outputs, parameter sweeps and diagnostic scripts are not distributed. `docs/source_manifest.json` records the copied source files and checksums.

## Installation

Use Python 3.10–3.12 and preferably PyTorch 2.5 or newer. Install a PyTorch build compatible with the server's GPU driver/CUDA environment, then run from the repository root:

```bash
pip install -r requirements.txt
```

For FS-PHM alone: `pip install -r model/requirements.txt`. SC-MambaFew CUDA training additionally requires a matching `mamba-ssm` installation; `--scan-backend reference` supports small CPU checks. MOMENT uses Hugging Face pretrained weights by default, requiring internet access or a local cache. `--hf-init random` constructs it from the included architecture configuration.

Examples use Bash continuations. On PowerShell, join each command into one line or replace the trailing backslash with a backtick.

## H5 data

Place prepared H5 datasets under `splitData/`, or supply an external location with `--root`. Raw-data exporters and datasets are not included.

```text
splitData/
├── JNU_h5/              # FD pretraining
├── PU_h5/               # FD pretraining
├── ims_h5/              # RUL pretraining
├── PHM2012_h5/          # RUL pretraining
├── CWRU_h5/             # Target FD
├── UO_h5/               # Target FD; 
└── XJTU-SY_h5_01/       # Target RUL, 0.1 s
```

Each row represents a vibration window. The loader supports signal keys such as `DE/FE`, `Horizontal/Vertical`, `vibration` and `signal`. FD labels normally use `label`; the recommended RUL target is `rul_norm` in [0, 1]. H5 files must provide sampling/window and source/device metadata sufficient to establish physical duration and acquisition/trajectory identities.

The pretraining index does not return supervised values such as `label`, `RUL`, `rul_min`, `rul_sec` or `rul_norm`. Exclude entire target datasets from pretraining, not only their support windows.

### Fixed FD partitions and support sets

Create one acquisition-group split, then draw supports only from its training pool. A group is a source acquisition/file, not necessarily an independent physical bearing.

```bash
python -m model.make_fd_split \
  --root splitData/CWRU_h5_12kdrive --datasets CWRU \
  --unit group --test-fraction 0.2 --seed 2026 \
  --output runs/manifests/cwru_split.json

python -m model.make_few_shot \
  --task fd --root splitData/CWRU_h5_12kdrive --datasets CWRU \
  --split-manifest runs/manifests/cwru_split.json \
  --shots 4 --seed 2026 --output runs/manifests/cwru_4shot.json
```

Change `--shots` for 1/2/4/8/16-shot. For 10% or the entire training pool, replace `--shots 4` with `--train-fraction 0.1` or `--train-fraction 1.0`, using a distinct output file. For UO use `splitData/UO_h5` and `--datasets UO`.

All budgets must share the same split manifest. Test windows never enter support selection. Exact historical reproduction requires the original manifests; regenerating 1-shot does not preserve a manually replaced normal support example.

## Pretraining

Example using four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m model.train_pretrain \
  --task all --data-root splitData \
  --root splitData/JNU_h5 --root splitData/PU_h5 \
  --root splitData/ims_h5 --root splitData/PHM2012_h5 \
  --exclude-dataset CWRU --exclude-dataset UO --exclude-dataset XJTU-SY \
  --sampling task-dataset-balanced --fd-sampling-fraction 0.5 \
  --epochs 10 --batch-size 512 --workers 16 --seed 2026 \
  --output runs/fs_phm_pretrain
```

Batch size is per process; effective DDP batch size also includes the number of GPUs and gradient accumulation. For one GPU use `python -m model.train_pretrain ... --device cuda:0`. Progress, configurations, metrics and checkpoints are saved in the output directory. Resume with `--resume runs/fs_phm_pretrain/last.pt`. Sampling defaults to `records`; the example explicitly enables balancing.

## FD adaptation

```bash
python -m model.train_finetune \
  --task fd --root splitData/CWRU_h5_12kdrive --datasets CWRU \
  --split-manifest runs/manifests/cwru_split.json \
  --few-shot-manifest runs/manifests/cwru_4shot.json \
  --checkpoint runs/fs_phm_pretrain/epoch_0002.pt \
  --adaptation lora --fd-head linear --fd-pooling mean \
  --epochs 50 --batch-size 64 --eval-batch-size 512 \
  --lr 0.002 --backbone-lr 0.00008 --workers 8 --seed 2026 \
  --device cuda:0 --max-eval -1 --output runs/cwru_4shot
```

Add `--fd-power-spectrum` for the spectral branch. Its standardization and ridge initialization use only the training supports; see `--help` for confidence-gate settings. Use `--fd-head cosine` for a cosine classifier. Switching the support manifest selects other shot counts, 10%, or the complete training pool.

The fixed-split protocol trains for the requested number of epochs and then evaluates the full fixed test partition. Metrics are Accuracy, macro F1, macro AUROC and macro Recall. Use a new output for a new setting, or `--resume <downstream checkpoint>` for continuation.

## Frozen-foundation RUL

`model.train_rul_frozen` packages the selected RUL pipeline without historical debug-script or run-directory dependencies.

```bash
python -m model.train_rul_frozen \
  --root splitData/XJTU-SY_h5_01 \
  --checkpoint runs/fs_phm_pretrain/epoch_0002.pt \
  --split-seed 2026 --seed 2026 --sampling-seed 2026 \
  --epochs 50 --batch-size 64 --workers 8 --device cuda:0 \
  --output runs/xjtu_frozen/seed_2026
```

The input is one two-channel vibration window. Fixed channel scales and amplitude standardization are fitted on training bearings only. The entire foundation is frozen; the head uses MAE and AdamW with an initial learning rate of 2e-4. Condition/bearing-balanced sampling draws N windows with replacement per epoch from the entire N-window training pool; it does not visit each row exactly once. Extracted features reside in RAM, with no feature cache written to disk.

Defaults preserve the adjusted full-pool learning budget: 20 warmup updates, a cosine schedule with a 600-update horizon, then a learning rate of 1e-8 after update 325, completing 50 epochs in total. Thus, 50 epochs does not mean 50 epochs at the initial learning rate. These settings are exposed as `--warmup-steps`, `--scheduler-steps`, `--phase-steps` and `--floor-lr`.

The split holds out complete bearing trajectories. The original data version yields 12 training bearings and 3 test bearings, with 206,304/14,880 windows. Counts depend on the data version. Keep `--split-seed 2026` fixed and change `--seed` to 2027/2028 with distinct output directories for repeated training. TEST is evaluated after fitting and is not used to select the best epoch.

Resume and evaluate:

```bash
# Retain the original training arguments when resuming.
python -m model.train_rul_frozen \
  --root splitData/XJTU-SY_h5_01 \
  --checkpoint runs/fs_phm_pretrain/epoch_0002.pt \
  --workers 8 --device cuda:0 --resume --output runs/xjtu_frozen/seed_2026

# final.pt includes the complete foundation, head and normalization buffers.
python -m model.train_rul_frozen \
  --root splitData/XJTU-SY_h5_01 \
  --evaluate runs/xjtu_frozen/seed_2026/final.pt \
  --workers 8 --device cuda:0 --output runs/xjtu_evaluation
```

Outputs include `config.json`, `metrics.jsonl`, `last.pt`, `final.pt`, `test_metrics.json`, per-window predictions and per-bearing metrics. `--evaluate` accepts checkpoints produced by this clean entry; historical debug checkpoint wrappers require conversion first.

The general `model.train_rul_full` entry also retains LoRA/linear/full adaptation. Its default head and losses differ from the fusion regressor described here.

## Baselines

The shared runner exposes generic networks and supplied paper-model adapters. Reuse the same FD split/support identities:

```bash
python -m model.paper_baselines \
  --models transformer resnet18 cnn_lstm wdcnn unifault \
  --datasets CWRU --regimes 4shot \
  --cwru-root splitData/CWRU_h5_12kdrive \
  --cwru-split runs/manifests/cwru_split.json \
  --cwru-support-template 'runs/manifests/cwru_{regime}.json' \
  --epochs 50 --batch-size 64 --seed 2026 --device cuda:1 \
  --output runs/comparisons_fd

python -m model.paper_baselines \
  --models da_rul cnn_bilstm dual_mixer transformer resnet18 cnn_lstm \
  --datasets XJTU-SY --regimes full --history 1 \
  --xjtu-root splitData/XJTU-SY_h5_01 \
  --epochs 50 --batch-size 64 --seed 2026 --device cuda:1 \
  --output runs/comparisons_rul
```

Baseline pretraining:

```bash
python -m model.paper_baselines.pretrain \
  --models unifault moment_community \
  --root splitData/JNU_h5 --root splitData/PU_h5 \
  --root splitData/ims_h5 --root splitData/PHM2012_h5 \
  --holdout-datasets CWRU UO XJTU-SY \
  --epochs 10 --batch-size 64 --device cuda:1 --output runs/comparison_pretrain
```

Pass `--pretrain-root runs/comparison_pretrain` downstream to use those backbones. Models without pretrained weights train from random initialization. UniFault is FD-only. `--history 1` adapts historical-sequence models to one window and does not reproduce their original full-history protocol. Baseline runners can write waveform caches under `--cache-dir`; FS-PHM training does not use that disk cache.


RmGPT is located under `model/paper_baselines/rmgpt/`; the former `model.rmgpt_paper.*` module paths are replaced by:

```bash
python -m model.paper_baselines.rmgpt.pretrain \
  --root splitData/JNU_h5 --root splitData/PU_h5 \
  --root splitData/ims_h5 --root splitData/PHM2012_h5 \
  --holdout-datasets CWRU UO XJTU-SY \
  --epochs 10 --batch-size 64 --device cuda:1 --output runs/rmgpt/pretrain

python -m model.paper_baselines.rmgpt.finetune \
  --datasets CWRU --regimes 4shot \
  --cwru-split runs/manifests/cwru_split.json \
  --cwru-support-template 'runs/manifests/cwru_{regime}.json' \
  --checkpoint runs/rmgpt/pretrain/last.pt --adaptation prompt \
  --epochs 50 --batch-size 64 --device cuda:1 --output runs/rmgpt/fd_prompt
```

Use `--adaptation full` for full-parameter adaptation. The paper-derived RmGPT implementation and `moment_community` are separate models with incompatible checkpoints. See [baseline notes](docs/baselines.md) for task scope and implementation differences.

## Reference results

These are existing experiment records, not new training runs of this export. FD Accuracy (%) is taken from the supplied manuscript and reported as mean ± standard deviation across three training seeds:

| Dataset / model | 1-shot | 4-shot | 8-shot | 16-shot | 10% | Full |
|---|---:|---:|---:|---:|---:|---:|
| CWRU · FS-PHM | 87.77 ± 0.25 | 93.29 ± 0.21 | 94.57 ± 0.13 | 95.29 ± 0.29 | 97.21 ± 0.05 | 99.02 ± 0.08 |
| CWRU · + Power Spectrum | 94.15 ± 1.96 | 99.72 ± 0.39 | 99.98 ± 0.03 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| UO · FS-PHM | 79.09 ± 1.31 | 93.74 ± 1.14 | 96.33 ± 0.76 | 97.64 ± 0.62 | 99.43 ± 0.15 | 99.97 ± 0.06 |
| UO · + Power Spectrum | 89.12 ± 4.25 | 99.93 ± 0.06 | 99.93 ± 0.06 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

| XJTU-SY full training pool | MAE ↓ | MSE ↓ | RMSE ↓ |
|---|---:|---:|---:|
| FS-PHM frozen fusion regressor | 0.13158 ± 0.00163 | 0.02624 ± 0.00087 | 0.16196 ± 0.00269 |

RUL metrics are computed on the complete fixed TEST for each training seed, then summarized across 2026/2027/2028 with sample standard deviation. The deviation is not the spread of individual window errors. Exact reproduction requires the original weights, data version and split/support manifests, which are not included.

![CWRU confusion matrix comparison](docs/assets/cwru_confusion.png)

*Manuscript example: no pretraining, pretraining, and pretraining plus power spectrum. A single-run visualization does not replace the multi-seed aggregate.*

![RUL error across lifetime stages](docs/assets/rul_lifetime_stage_error.png)

*RUL error analysis generated this month. Stages use true normalized RUL intervals and do not identify fault onset.*

More tables, figures and provenance: [results](docs/results.md).

## Verification and execution

```bash
python -m unittest discover -s tests -v
python -m model.train_pretrain --help
python -m model.train_finetune --help
python -m model.train_rul_frozen --help
python -m model.paper_baselines --models transformer --datasets CWRU --plan-only
```

CPU checks cover variable-length masking, pretraining backpropagation, a frozen RUL foundation, head parameter counts, and complete-weight round trips. Full GPU training requires the server and the corresponding H5 datasets.

FS-PHM pretraining supports DDP. Baseline training and frozen RUL run one process per experiment; distribute models or seeds across GPUs. `CUDA_VISIBLE_DEVICES=3 ... --device cuda:0` selects physical GPU 3, whereas `--device cuda:3` addresses visible GPU 3.
