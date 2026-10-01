"""Render the Extended Data Figure 5 within-metric BH3 update."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO / "scripts/render_ed05_example_place_maps_assembly.py"
OUTPUT = REPO / "docs/figure_refinement_2026-08-16/ed05_round16_bh3"
ROUND = 16


def _load_base():
    spec = importlib.util.spec_from_file_location("ed05_assembly_for_round16", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    renderer = _load_base()
    renderer.ROUND = ROUND
    renderer.main(["--output-dir", str(OUTPUT)])


if __name__ == "__main__":
    main()
