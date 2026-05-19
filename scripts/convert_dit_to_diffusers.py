#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))  # noqa: E402

from diffusers._hf import get_hf_attr
from diffusers.utils import DIT_MODEL_PRESETS, convert_original_state_dict, get_transformer_config

try:
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_save_file = None


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _save_config(output_dir: Path, config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(
            state_dict,
            str(output_dir / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, vae: str) -> None:
    model_index = {
        "_class_name": "DiTPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "DDIMScheduler"],
        "transformer": ["diffusers", "DiTTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as handle:
        handle.write(vae + os.linesep)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy DiT .pt checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a legacy DiT .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model", choices=sorted(DIT_MODEL_PRESETS), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", choices=["ema", "mse"], default="mse")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    DiTPipeline = get_hf_attr("diffusers.pipelines.dit.pipeline_dit.DiTPipeline")
    DiTTransformer2DModel = get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    DDIMScheduler = get_hf_attr("diffusers.schedulers.DDIMScheduler")
    AutoencoderKL = get_hf_attr("diffusers.models.autoencoder_kl.AutoencoderKL")

    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    converted = convert_original_state_dict(state_dict, args.model)
    transformer_config = get_transformer_config(args.model, args.image_size, num_classes=args.num_classes)

    if args.check_load:
        model = DiTTransformer2DModel(**transformer_config)
        missing, unexpected = model.load_state_dict(converted, strict=True)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "DiTTransformer2DModel", **transformer_config})
    _save_weights(transformer_dir, converted, args.safe_serialization)
    _save_config(
        scheduler_dir,
        {
            "_class_name": "DDIMScheduler",
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
            "clip_sample": False,
        },
    )

    vae_id = f"stabilityai/sd-vae-ft-{args.vae}"
    _write_model_index(output_dir, vae_id)

    if args.check_load:
        pipeline = DiTPipeline(
            transformer=DiTTransformer2DModel(**transformer_config),
            scheduler=DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=False,
            ),
            vae=AutoencoderKL.from_pretrained(vae_id),
        )
        pipeline.transformer.load_state_dict(converted, strict=True)

    print(f"Saved Diffusers-style DiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
