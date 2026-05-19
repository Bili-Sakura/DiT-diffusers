#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))  # noqa: E402

from dit_diffusers import load_dit_pipeline
from dit_diffusers.utils.sampling import sample_dit_latents

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        samples.append(np.asarray(sample_pil).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    latent_size = args.image_size // 8
    pipe = load_dit_pipeline(
        model_name=args.model,
        image_size=args.image_size,
        vae=args.vae,
        checkpoint_path=args.ckpt,
        diffusers_dir=args.diffusers_dir,
        torch_dtype=torch.float32,
    ).to(device)

    using_cfg = args.cfg_scale > 1.0
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    sample_folder_dir = (
        f"{args.sample_dir}/{model_string_name}-{ckpt_string_name}-size-{args.image_size}-"
        f"vae-{args.vae}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    )
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0

    for _ in pbar:
        generator = torch.Generator(device=device).manual_seed(seed + total)
        class_labels = torch.randint(0, args.num_classes, (n,), device=device)
        latents = sample_dit_latents(
            transformer=pipe.transformer,
            scheduler=pipe.scheduler,
            class_labels=class_labels,
            latent_size=latent_size,
            in_channels=pipe.transformer.config.in_channels,
            guidance_scale=args.cfg_scale if using_cfg else 1.0,
            num_inference_steps=args.num_sampling_steps,
            generator=generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        latents = latents / pipe.vae.config.scaling_factor
        samples = pipe.vae.decode(latents).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--diffusers-dir", type=str, default=None)
    main(parser.parse_args())
