#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))  # noqa: E402

from diffusers import load_dit_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with a Diffusers-native DiT pipeline.")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--class-label", type=int, action="append", default=[207, 360, 387, 974, 88, 979, 417, 279])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--ckpt", type=str, default=None, help="Legacy .pt checkpoint or converted Diffusers directory.")
    parser.add_argument("--diffusers-dir", type=str, default=None, help="Converted Diffusers pipeline directory.")
    parser.add_argument("--output", type=str, default="sample.png")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    generator = torch.Generator(device=args.device if args.device != "cpu" else "cpu")
    if args.seed is not None:
        generator.manual_seed(args.seed)

    pipe = load_dit_pipeline(
        model_name=args.model,
        image_size=args.image_size,
        vae=args.vae,
        checkpoint_path=args.ckpt,
        diffusers_dir=args.diffusers_dir,
        torch_dtype=dtype,
    ).to(args.device)

    output = pipe(
        class_labels=args.class_label,
        guidance_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    images = torch.stack([pil_to_tensor(image).float() / 255.0 for image in output.images])
    save_image(images, args.output, nrow=4, normalize=False)


if __name__ == "__main__":
    main()
