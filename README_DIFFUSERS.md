DiT Diffusers integration (custom pipeline)
===========================================

This repository mirrors the [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers) layout for a project-owned custom `DiTPipeline`. It does not proxy to Diffusers built-in `DiTPipeline`.

Layout (NiT-style, installable as `diffusers`):

```text
src/diffusers/
  models/transformers/transformer_dit.py   # DiTTransformer2DModel
  pipelines/dit/pipeline_dit.py            # custom DiTPipeline implementation
  schedulers/                              # DDIMScheduler, DDPMScheduler
  utils/                                   # conversion, training, loading, sampling
scripts/
  convert_dit_to_diffusers.py
  sample_dit.py
  sample_dit_ddp.py
  train_dit.py
```

Install
-------

```bash
pip install -e .
```

Convert a legacy checkpoint
---------------------------

```bash
python scripts/convert_dit_to_diffusers.py \
  --checkpoint pretrained_models/DiT-XL-2-256x256.pt \
  --output dit-xl-256-diffusers \
  --model DiT-XL/2 \
  --check-load
```

Sample (custom pipeline)
------------------------

```bash
python scripts/sample_dit.py --class-label 207 --image-size 256
```

Model-repo loading pattern:

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("models/BiliSakura/DiT-diffusers/DiT-XL-2-512")
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
```

Train / FID
-----------

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train_dit.py --data-path /path/to/imagenet/train
torchrun --nnodes=1 --nproc_per_node=8 scripts/sample_dit_ddp.py --num-fid-samples 50000
```

When upstreaming to `huggingface/diffusers`, copy modules into `src/diffusers/models`, `pipelines`, and `schedulers` and register them in Diffusers' lazy import tables.
