# Comparison model scope / 对比模型范围

| Command name | Tasks | Implementation |
|---|---|---|
| `transformer` | FD, RUL | Local 1D Transformer baseline |
| `resnet18` | FD, RUL | Local 1D ResNet18 baseline |
| `cnn_lstm` | FD, RUL | Local CNN + bidirectional LSTM baseline |
| `wdcnn` | FD | PyTorch port of the supplied five-convolution Keras architecture |
| `unifault` | FD | Supplied UniFault Transformer backbone |
| `sc_mamba` | FD | Supplied feature/SSM/fusion/covariance modules, fixed labeled supports |
| `moment_community` | FD, RUL | Supplied MOMENT/T5 community implementation; not RmGPT |
| `da_rul` | RUL | Supplied degradation attention + sLSTM, adapted vibration descriptors |
| `cnn_bilstm` | RUL | Supplied convolution/recurrent backbone, explicit within-bearing history |
| `dual_mixer` | RUL | Supplied dual mixer and FSGRI loss, TRAIN-only negative samples |
| `paper_baselines.rmgpt` | FD, RUL | Independent paper-derived RmGPT architecture; next-patch pretraining, prompt/full adaptation |

## Protocol

FD comparisons can consume the same split/support manifests as FS-PHM. Generated manifests are stored under each output's `protocols/` directory. `--regimes all` runs 1/4/8/16-shot, 10% and full training pool; it does not include 2-shot by default. FS-PHM itself supports arbitrary positive K through `make_few_shot`.

RUL comparisons split complete bearing trajectories. Use `--history 1` for the single-window comparison; the runner's generic default is 30, so this option must be explicit. Adapted historical-sequence models are not a reproduction of their complete original-paper evaluation protocol. Input transforms and optimizer settings differ by implementation and are recorded in configuration files.

FD 对比可复用主模型的 split/support；RUL 使用完整轴承轨迹隔离。`--history 1` 需要显式指定。模型的原始研究背景与当前适配实验应分别说明，不能将适配结果直接标为原论文指标。

## Sources and dependencies

`vendor/sources.json` retains upstream archive hashes and recorded import/compatibility changes. This distribution includes only the transitive upstream modules used by the adapters, plus available attribution/license files; original data loaders, parameter-search scripts and unused upstream training scripts were removed. `vendor_pruning.json` records that selection.

The RmGPT code has been relocated from `model.rmgpt_paper` to `model.paper_baselines.rmgpt`; imports and source-identity paths were updated. It is a local implementation derived from the paper, not a claim of official released author code. It is distinct from the MOMENT implementation found in an earlier archive named RmGPT.

Install `mamba-ssm` separately for SC-Mamba CUDA execution. For MOMENT, use the root requirements' Transformers constraint and the matching Hugging Face cache, or select random initialization explicitly. Included upstream files retain their original ownership; consult their available license/attribution before redistribution. No new blanket license for third-party code is asserted here.

## Checkpoints and caches

Comparison pretraining saves a backbone checkpoint per model, accepted downstream through `--pretrain-root` or repeated `--checkpoint MODEL=PATH` arguments. RmGPT uses its own `--checkpoint` and accepts `--adaptation prompt` or `full`.

The comparison data adapters retain their original disk-backed cache implementation under `--cache-dir` (default `runs/paper_cache`). FS-PHM pretraining/FD load H5 lazily; the packaged frozen RUL entry keeps extracted statistics in RAM. No pre-existing cache or model weight is shipped.
