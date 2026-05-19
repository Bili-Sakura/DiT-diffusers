#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DiTTransformer2DModel


MODEL_CONFIGS = {
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 2},
    "DiT-XL/4": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 4},
    "DiT-XL/8": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 8},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "patch_size": 2},
    "DiT-L/4": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "patch_size": 4},
    "DiT-L/8": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "patch_size": 8},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 2},
    "DiT-B/4": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 4},
    "DiT-B/8": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 8},
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "num_heads": 6, "patch_size": 2},
    "DiT-S/4": {"depth": 12, "hidden_size": 384, "num_heads": 6, "patch_size": 4},
    "DiT-S/8": {"depth": 12, "hidden_size": 384, "num_heads": 6, "patch_size": 8},
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("ema", "state_dict", "model", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return dict(state_dict)


def _convert_state_dict(state_dict: Dict[str, torch.Tensor], depth: int) -> Dict[str, torch.Tensor]:
    state_dict = dict(state_dict)
    state_dict.pop("pos_embed", None)

    state_dict["pos_embed.proj.weight"] = state_dict.pop("x_embedder.proj.weight")
    state_dict["pos_embed.proj.bias"] = state_dict.pop("x_embedder.proj.bias")

    for layer_idx in range(depth):
        state_dict[f"transformer_blocks.{layer_idx}.norm1.emb.timestep_embedder.linear_1.weight"] = state_dict[
            "t_embedder.mlp.0.weight"
        ]
        state_dict[f"transformer_blocks.{layer_idx}.norm1.emb.timestep_embedder.linear_1.bias"] = state_dict[
            "t_embedder.mlp.0.bias"
        ]
        state_dict[f"transformer_blocks.{layer_idx}.norm1.emb.timestep_embedder.linear_2.weight"] = state_dict[
            "t_embedder.mlp.2.weight"
        ]
        state_dict[f"transformer_blocks.{layer_idx}.norm1.emb.timestep_embedder.linear_2.bias"] = state_dict[
            "t_embedder.mlp.2.bias"
        ]
        state_dict[f"transformer_blocks.{layer_idx}.norm1.emb.class_embedder.embedding_table.weight"] = state_dict[
            "y_embedder.embedding_table.weight"
        ]

        state_dict[f"transformer_blocks.{layer_idx}.norm1.linear.weight"] = state_dict[
            f"blocks.{layer_idx}.adaLN_modulation.1.weight"
        ]
        state_dict[f"transformer_blocks.{layer_idx}.norm1.linear.bias"] = state_dict[
            f"blocks.{layer_idx}.adaLN_modulation.1.bias"
        ]

        qkv_weight = state_dict.pop(f"blocks.{layer_idx}.attn.qkv.weight")
        qkv_bias = state_dict.pop(f"blocks.{layer_idx}.attn.qkv.bias")
        q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)

        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_q.weight"] = q_weight
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_q.bias"] = q_bias
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_k.weight"] = k_weight
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_k.bias"] = k_bias
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_v.weight"] = v_weight
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_v.bias"] = v_bias

        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_out.0.weight"] = state_dict.pop(
            f"blocks.{layer_idx}.attn.proj.weight"
        )
        state_dict[f"transformer_blocks.{layer_idx}.attn1.to_out.0.bias"] = state_dict.pop(
            f"blocks.{layer_idx}.attn.proj.bias"
        )

        state_dict[f"transformer_blocks.{layer_idx}.ff.net.0.proj.weight"] = state_dict.pop(
            f"blocks.{layer_idx}.mlp.fc1.weight"
        )
        state_dict[f"transformer_blocks.{layer_idx}.ff.net.0.proj.bias"] = state_dict.pop(
            f"blocks.{layer_idx}.mlp.fc1.bias"
        )
        state_dict[f"transformer_blocks.{layer_idx}.ff.net.2.weight"] = state_dict.pop(
            f"blocks.{layer_idx}.mlp.fc2.weight"
        )
        state_dict[f"transformer_blocks.{layer_idx}.ff.net.2.bias"] = state_dict.pop(
            f"blocks.{layer_idx}.mlp.fc2.bias"
        )

        state_dict.pop(f"blocks.{layer_idx}.adaLN_modulation.1.weight")
        state_dict.pop(f"blocks.{layer_idx}.adaLN_modulation.1.bias")

    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.2.weight")
    state_dict.pop("t_embedder.mlp.2.bias")
    state_dict.pop("y_embedder.embedding_table.weight")

    state_dict["proj_out_1.weight"] = state_dict.pop("final_layer.adaLN_modulation.1.weight")
    state_dict["proj_out_1.bias"] = state_dict.pop("final_layer.adaLN_modulation.1.bias")
    state_dict["proj_out_2.weight"] = state_dict.pop("final_layer.linear.weight")
    state_dict["proj_out_2.bias"] = state_dict.pop("final_layer.linear.bias")

    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a DiT checkpoint into a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to a DiT .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Output directory for the converted pipeline.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    attention_head_dim = model_config["hidden_size"] // model_config["num_heads"]
    latent_size = args.image_size // 8

    state_dict = _load_state_dict(args.checkpoint)
    converted = _convert_state_dict(state_dict, model_config["depth"])

    transformer = DiTTransformer2DModel(
        sample_size=latent_size,
        patch_size=model_config["patch_size"],
        in_channels=4,
        out_channels=8,
        num_layers=model_config["depth"],
        num_attention_heads=model_config["num_heads"],
        attention_head_dim=attention_head_dim,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=args.num_classes,
        attention_bias=True,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
    )
    missing_keys, unexpected_keys = transformer.load_state_dict(converted, strict=False)
    if missing_keys or unexpected_keys:
        raise ValueError(f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    vae = AutoencoderKL.from_pretrained(args.vae)
    pipe = DiTPipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    print(f"Saved Diffusers DiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
