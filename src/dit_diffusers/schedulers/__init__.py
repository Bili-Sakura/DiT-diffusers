from .._hf import get_hf_diffusers

__all__ = ["DDIMScheduler", "DDPMScheduler"]


def __getattr__(name: str):
    if name == "DDIMScheduler":
        return get_hf_diffusers().DDIMScheduler
    if name == "DDPMScheduler":
        return get_hf_diffusers().DDPMScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
