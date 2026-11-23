# Export verification / 副本验证

Verified on 2026-11-23 using Python 3.10, PyTorch 2.5.1 and CPU execution.

- Seven unit/integration tests passed: amplitude scaling, masked variable-length pretraining/backpropagation, RUL masked pooling and head size, frozen foundation/checkpoint round trip, synthetic-H5 pretraining plus FD/power training, RUL interrupted/resumed training with exact weight equality and independent evaluation, and relocated RmGPT pretraining/prompt adaptation.
- Ten comparison adapters produced finite forward outputs: Transformer, ResNet18, CNN-LSTM, WDCNN, UniFault, DA-RUL, CNN-BiLSTM, Dual-Mixer, randomly initialized MOMENT/T5, and SC-MambaFew using the reference scan backend. See `baseline_verification.json`.
- Eleven training/data entry points returned successfully for `--help`. See `cli_verification.json`.
- All retained Python files passed syntax inspection and local-import dependency inspection.
- Local links in the two READMEs and project documentation resolved successfully.
- Copied source-file hashes were rechecked against the source project; no source changes were detected. The export contains ordinary independent files, not hard links to the source.

These checks establish packaging and small-run functionality. They do not constitute a full GPU training run or a remeasurement of the manuscript results. CUDA/DDP execution, Mamba CUDA kernels and large-dataset throughput require server validation. Existing checkpoints and exact historical split/support manifests are not included.

本次验证针对代码整理、依赖完整性与小规模训练连通性；未在本机重跑论文完整实验，也未重新测量 README 中的精度。
