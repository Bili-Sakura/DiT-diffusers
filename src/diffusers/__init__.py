from .utils import (
    DIT_MODEL_PRESETS,
    compute_dit_training_loss,
    convert_original_state_dict,
    create_training_scheduler,
    get_transformer_config,
    load_dit_pipeline,
)

__all__ = [
    "DIT_MODEL_PRESETS",
    "DiTPipeline",
    "DiTTransformer2DModel",
    "compute_dit_training_loss",
    "convert_original_state_dict",
    "create_training_scheduler",
    "get_transformer_config",
    "load_dit_pipeline",
]


def __getattr__(name: str):
    if name == "DiTTransformer2DModel":
        from .models.transformers.transformer_dit import DiTTransformer2DModel

        return DiTTransformer2DModel
    if name == "DiTPipeline":
        from .pipelines.dit.pipeline_dit import DiTPipeline

        return DiTPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
