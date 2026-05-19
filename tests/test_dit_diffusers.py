import pytest

torch = pytest.importorskip("torch")

from diffusers import (
    DIT_MODEL_PRESETS,
    compute_dit_training_loss,
    convert_original_state_dict,
    create_training_scheduler,
    get_transformer_config,
)
from diffusers._hf import get_hf_attr


def test_transformer_forward():
    DiTTransformer2DModel = get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    model = DiTTransformer2DModel(
        sample_size=4,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        num_embeds_ada_norm=10,
    )
    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([10, 20])
    class_labels = torch.tensor([1, 2])
    output = model(latents, timestep=timesteps, class_labels=class_labels)
    assert output.sample.shape == (2, 8, 8, 8)


def test_training_loss():
    DiTTransformer2DModel = get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    model = DiTTransformer2DModel(
        sample_size=4,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        num_embeds_ada_norm=10,
    )
    scheduler = create_training_scheduler()
    latents = torch.randn(2, 4, 8, 8)
    class_labels = torch.tensor([1, 2])
    loss = compute_dit_training_loss(model, scheduler, latents, class_labels)
    assert loss.ndim == 0


def test_convert_state_dict_shapes():
    preset = DIT_MODEL_PRESETS["DiT-S/2"]
    depth = preset["num_layers"]
    hidden = preset["num_attention_heads"] * preset["attention_head_dim"]
    state = {
        "x_embedder.proj.weight": torch.randn(hidden, 4, preset["patch_size"], preset["patch_size"]),
        "x_embedder.proj.bias": torch.randn(hidden),
        "pos_embed": torch.randn(1, 16, hidden),
        "t_embedder.mlp.0.weight": torch.randn(hidden, 256),
        "t_embedder.mlp.0.bias": torch.randn(hidden),
        "t_embedder.mlp.2.weight": torch.randn(hidden, hidden),
        "t_embedder.mlp.2.bias": torch.randn(hidden),
        "y_embedder.embedding_table.weight": torch.randn(1001, hidden),
        "final_layer.adaLN_modulation.1.weight": torch.randn(2 * hidden, hidden),
        "final_layer.adaLN_modulation.1.bias": torch.randn(2 * hidden),
        "final_layer.linear.weight": torch.randn(preset["patch_size"] ** 2 * 8, hidden),
        "final_layer.linear.bias": torch.randn(preset["patch_size"] ** 2 * 8),
    }
    for idx in range(depth):
        state[f"blocks.{idx}.adaLN_modulation.1.weight"] = torch.randn(6 * hidden, hidden)
        state[f"blocks.{idx}.adaLN_modulation.1.bias"] = torch.randn(6 * hidden)
        state[f"blocks.{idx}.attn.qkv.weight"] = torch.randn(3 * hidden, hidden)
        state[f"blocks.{idx}.attn.qkv.bias"] = torch.randn(3 * hidden)
        state[f"blocks.{idx}.attn.proj.weight"] = torch.randn(hidden, hidden)
        state[f"blocks.{idx}.attn.proj.bias"] = torch.randn(hidden)
        state[f"blocks.{idx}.mlp.fc1.weight"] = torch.randn(4 * hidden, hidden)
        state[f"blocks.{idx}.mlp.fc1.bias"] = torch.randn(4 * hidden)
        state[f"blocks.{idx}.mlp.fc2.weight"] = torch.randn(hidden, 4 * hidden)
        state[f"blocks.{idx}.mlp.fc2.bias"] = torch.randn(hidden)

    converted = convert_original_state_dict(state, "DiT-S/2")
    DiTTransformer2DModel = get_hf_attr("diffusers.models.transformers.dit_transformer_2d.DiTTransformer2DModel")
    model = DiTTransformer2DModel(**get_transformer_config("DiT-S/2", image_size=32))
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing
    assert not unexpected
