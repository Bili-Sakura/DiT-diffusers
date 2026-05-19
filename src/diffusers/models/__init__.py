__all__ = ["DiTTransformer2DModel"]


def __getattr__(name: str):
    if name == "DiTTransformer2DModel":
        from .transformers.transformer_dit import DiTTransformer2DModel

        return DiTTransformer2DModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
