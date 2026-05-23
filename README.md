## Scalable Diffusion Models with Transformers (DiT)<br><sub>Custom Diffusers pipeline implementation</sub>

### [Paper](http://arxiv.org/abs/2212.09748) | [Project Page](https://www.wpeebles.com/DiT) | [Diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/dit)

![DiT samples](visuals/sample_grid_0.png)

This repository provides class-conditional DiT training, sampling, and checkpoint conversion using a project-owned custom `DiTPipeline` implementation on top of [Hugging Face Diffusers](https://github.com/huggingface/diffusers) components. The layout follows [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers).

> [**Scalable Diffusion Models with Transformers**](https://www.wpeebles.com/DiT)<br>
> [William Peebles](https://www.wpeebles.com), [Saining Xie](https://www.sainingxie.com)

## Setup

```bash
git clone https://github.com/Bili-Sakura/DiT-diffusers.git
cd DiT-diffusers
pip install -e .
```

See [`README_DIFFUSERS.md`](README_DIFFUSERS.md) for full API and conversion details.

## Sampling

Pre-trained DiT-XL/2 weights can be loaded from custom checkpoint folders (`BiliSakura/DiT-diffusers/...`) with `custom_pipeline`:

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("models/BiliSakura/DiT-diffusers/DiT-XL-2-512")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    torch_dtype=torch.float16,
).to("cuda")

images = pipe(class_labels=[207], num_inference_steps=250, guidance_scale=4.0).images
```

CLI sampling is still available:

```bash
python scripts/sample_dit.py --class-label 207 --image-size 256 --cfg-scale 4.0
```

Legacy `.pt` checkpoints can still be converted:

```bash
python scripts/convert_dit_to_diffusers.py \
  --checkpoint pretrained_models/DiT-XL-2-256x256.pt \
  --output dit-xl-256-diffusers \
  --model DiT-XL/2 --check-load

python scripts/sample_dit.py --diffusers-dir dit-xl-256-diffusers --class-label 207
```

## Training

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_dit.py \
  --model DiT-XL/2 \
  --data-path /path/to/imagenet/train
```

## FID evaluation

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/sample_dit_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 50000
```

## Pre-trained weights

| DiT Model | Resolution | Checkpoint |
|-----------|------------|------------|
| XL/2 | 256×256 | [DiT-XL-2-256x256.pt](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) |
| XL/2 | 512×512 | [DiT-XL-2-512x512.pt](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) |

## BibTeX

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```
