"""Load the Hugging Face ``diffusers`` PyPI package while this repository's ``src`` tree is on ``sys.path``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

_LOCAL_ROOT = Path(__file__).resolve().parent
_LOCAL_SRC = _LOCAL_ROOT.parent
_CACHED: Optional[object] = None


def _is_local_diffusers_module(module_name: str) -> bool:
    module = sys.modules.get(module_name)
    if module is None:
        return False
    module_file = getattr(module, "__file__", "") or ""
    if str(_LOCAL_ROOT) in module_file or str(_LOCAL_SRC) in module_file:
        return True
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        return any(str(_LOCAL_ROOT) in str(path) for path in module_paths)
    return module_name == "diffusers" and module_file == ""


def _clear_local_diffusers_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            if _is_local_diffusers_module(module_name):
                sys.modules.pop(module_name, None)


def get_hf_diffusers():
    """Return the installed Hugging Face diffusers distribution (not this repository's package)."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        _clear_local_diffusers_modules()
        _CACHED = importlib.import_module("diffusers")
    finally:
        sys.path = original_path

    return _CACHED


def get_hf_attr(dotted_path: str):
    """Import a symbol from the Hugging Face diffusers package by dotted path."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    original_path = sys.path[:]
    try:
        sys.path = [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]
        _clear_local_diffusers_modules()
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    finally:
        sys.path = original_path
