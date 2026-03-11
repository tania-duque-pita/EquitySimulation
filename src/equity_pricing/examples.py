"""Compatibility wrapper for the top-level examples module."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_examples_module() -> ModuleType:
    package_root = Path(__file__).resolve().parents[2]
    module_path = package_root / "examples" / "examples.py"
    if not module_path.exists():
        raise ModuleNotFoundError(f"Missing examples module at {module_path}.")

    spec = spec_from_file_location("equity_pricing_external_examples", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load examples module from {module_path}.")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_examples_module = _load_examples_module()
run_end_to_end_example = _examples_module.run_end_to_end_example
main = _examples_module.main

__all__ = ["run_end_to_end_example", "main"]
