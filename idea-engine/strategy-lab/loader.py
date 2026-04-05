"""
loader.py
---------
Bootstrap loader: makes the strategy-lab package importable as 'strategy_lab'
from anywhere in the SRFM lab environment.

Usage
-----
    # At the top of any script that needs strategy-lab:
    import importlib, pathlib, sys
    exec(open(pathlib.Path(__file__).parent / 'strategy-lab/loader.py').read())

Or simply:
    sys.path.insert(0, '/path/to/idea-engine/strategy-lab')
    # Then import via package names: versioning, experiments, champion, etc.
    from versioning.strategy_version import StrategyVersion

Or install in development mode:
    pip install -e idea-engine/strategy-lab  (requires pyproject.toml)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_STRATEGY_LAB_DIR = Path(__file__).parent


def register() -> None:
    """
    Register strategy-lab as 'strategy_lab' in sys.modules so that
    relative imports within the package work correctly.
    """
    if "strategy_lab" in sys.modules:
        return

    # Build a module object for 'strategy_lab'
    init_path = _STRATEGY_LAB_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "strategy_lab",
        init_path,
        submodule_search_locations=[str(_STRATEGY_LAB_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy_lab from {init_path}")

    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(_STRATEGY_LAB_DIR)]  # type: ignore[attr-defined]
    module.__package__ = "strategy_lab"
    sys.modules["strategy_lab"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Pre-register sub-packages so relative imports resolve correctly
    for subpkg in ("versioning", "experiments", "champion", "simulation", "reporting"):
        subpkg_dir = _STRATEGY_LAB_DIR / subpkg
        subpkg_init = subpkg_dir / "__init__.py"
        if not subpkg_init.exists():
            continue
        full_name = f"strategy_lab.{subpkg}"
        if full_name in sys.modules:
            continue
        subspec = importlib.util.spec_from_file_location(
            full_name,
            subpkg_init,
            submodule_search_locations=[str(subpkg_dir)],
        )
        if subspec is None or subspec.loader is None:
            continue
        submod = importlib.util.module_from_spec(subspec)
        submod.__path__ = [str(subpkg_dir)]  # type: ignore[attr-defined]
        submod.__package__ = full_name
        sys.modules[full_name] = submod


# Auto-register when imported
register()
