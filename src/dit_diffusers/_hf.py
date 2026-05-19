"""Access the Hugging Face diffusers package installed from PyPI."""

from __future__ import annotations

from typing import Optional

_CACHED: Optional[object] = None


def get_hf_diffusers():
    global _CACHED
    if _CACHED is None:
        import diffusers as hf_diffusers

        _CACHED = hf_diffusers
    return _CACHED
