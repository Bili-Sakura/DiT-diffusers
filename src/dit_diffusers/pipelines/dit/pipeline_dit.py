# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# SPDX-License-Identifier: Apache-2.0

from ..._hf import get_hf_diffusers

DiTPipeline = get_hf_diffusers().DiTPipeline

__all__ = ["DiTPipeline"]
