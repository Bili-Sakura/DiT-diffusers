# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT using Diffusers-native components.
"""
import argparse
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dit_diffusers import DIT_MODEL_PRESETS, load_dit_pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None and args.diffusers_dir is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    pipe = load_dit_pipeline(
        model_name=args.model,
        image_size=args.image_size,
        vae=args.vae,
        checkpoint_path=args.ckpt,
        diffusers_dir=args.diffusers_dir,
        torch_dtype=torch.float32,
    ).to(device)

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        class_labels=class_labels,
        guidance_scale=args.cfg_scale,
        num_inference_steps=args.num_sampling_steps,
        generator=generator,
    )
    images = torch.stack([pil_to_tensor(image).float() / 255.0 for image in output.images])
    save_image(images, "sample.png", nrow=4, normalize=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DIT_MODEL_PRESETS.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Legacy .pt checkpoint or converted Diffusers directory (default: Hub download for DiT-XL/2).",
    )
    parser.add_argument("--diffusers-dir", type=str, default=None, help="Converted Diffusers pipeline directory.")
    args = parser.parse_args()
    main(args)
