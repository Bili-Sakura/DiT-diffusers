DiT Diffusers integration
=========================

This repository uses native [Hugging Face Diffusers](https://github.com/huggingface/diffusers) components instead of the legacy `models.py` and `diffusion/` implementations.

Package layout (mirrors [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers)):

- `src/dit_diffusers/` — checkpoint conversion, training loss helpers, and pipeline loading utilities
- `scripts/convert_dit_to_diffusers.py` — convert legacy `.pt` checkpoints into a Diffusers pipeline directory
- `scripts/sample_dit.py` — sample images with `DiTPipeline`
- `sample.py`, `sample_ddp.py`, `train.py` — updated entry points that call Diffusers APIs directly

Core Diffusers classes used:

- `DiTTransformer2DModel` — transformer backbone with adaLN-Zero conditioning
- `DiTPipeline` — class-conditional latent diffusion sampling
- `DDPMScheduler` / `DDIMScheduler` — noise schedules matching the original linear 1000-step training setup

Convert a legacy checkpoint
---------------------------

```bash
pip install -e .
python scripts/convert_dit_to_diffusers.py \
  --checkpoint pretrained_models/DiT-XL-2-256x256.pt \
  --output dit-xl-256-diffusers \
  --model DiT-XL/2 \
  --image-size 256 \
  --check-load
```

The output directory contains:

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae_pretrained_model_name_or_path.txt
```

Sample from Hub or a converted checkpoint
-----------------------------------------

Pre-trained DiT-XL/2 weights are available on the Hub (`facebook/DiT-XL-2-256`, `facebook/DiT-XL-2-512`):

```bash
python scripts/sample_dit.py \
  --class-label 207 \
  --class-label 360 \
  --image-size 256 \
  --cfg-scale 4.0 \
  --num-inference-steps 250
```

For a converted local pipeline:

```bash
python scripts/sample_dit.py \
  --diffusers-dir dit-xl-256-diffusers \
  --class-label 207
```

Train with Diffusers components
-------------------------------

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model DiT-XL/2 \
  --data-path /path/to/imagenet/train
```

New checkpoints store Diffusers-compatible `DiTTransformer2DModel` weights. Convert older training outputs with `scripts/convert_dit_to_diffusers.py` if needed.

Upstreaming
-----------

To contribute these APIs upstream, the conversion logic in `src/dit_diffusers/conversion.py` matches Hugging Face's official `scripts/convert_dit_to_diffusers.py`. The training helper in `src/dit_diffusers/training.py` can be reused in Diffusers training examples.
