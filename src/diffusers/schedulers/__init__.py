from .._hf import get_hf_attr

__all__ = ["DDIMScheduler", "DDPMScheduler"]


def __getattr__(name: str):
    if name == "DDIMScheduler":
        return get_hf_attr("diffusers.schedulers.DDIMScheduler")
    if name == "DDPMScheduler":
        return get_hf_attr("diffusers.schedulers.DDPMScheduler")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
