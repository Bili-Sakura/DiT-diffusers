# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# SPDX-License-Identifier: Apache-2.0

from ..._hf import get_hf_attr

__all__ = ["DiTPipeline"]


def __getattr__(name: str):
    if name == "DiTPipeline":
        return get_hf_attr("diffusers.pipelines.dit.pipeline_dit.DiTPipeline")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
