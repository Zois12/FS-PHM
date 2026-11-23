# Encoder configuration provenance

These six JSON files were retrieved on 2026-10-15 from the public Hugging Face
`https://huggingface.co/{model_id}/resolve/main/config.json` endpoint. The basename
is the model ID with `/` replaced by `__`.

They contain configuration only, not pretrained weights. They allow offline
construction of the exact encoder architecture for random initialization,
checkpoint loading and resume. FLAN-T5 uses different gated feedforward layers
and, for the small variant, different layer/head counts from original T5.

The selected JSON's SHA256 is part of each MOMENT experiment's configuration
identity. Pretrained weights still load from the requested model ID through the
supplied community implementation.
