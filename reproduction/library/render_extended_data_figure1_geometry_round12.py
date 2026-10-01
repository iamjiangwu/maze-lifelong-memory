"""Render the non-promoted ED1 round-12 geometry refinement candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO / "scripts" / "render_extended_data_figure1_round4.py"
OUTPUT = REPO / "docs" / "figure_resolution_repair_2026-08-12" / "ed01"
ROUND = 12


def _load_base():
    spec = importlib.util.spec_from_file_location("ed01_round4_for_round12", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    renderer = _load_base()
    renderer.OUTPUT = OUTPUT
    renderer.ROUND = ROUND
    renderer.STEM = OUTPUT / f"extended_data_figure1_source_rerender_round{ROUND}"
    renderer.SOURCE_AUDIT = OUTPUT / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv"
    renderer.LAYOUT_AUDIT = OUTPUT / f"LAYOUT_AUDIT_ROUND{ROUND}.csv"
    renderer.CAPTION_NOTES = OUTPUT / f"CAPTION_NOTES_ROUND{ROUND}.md"
    renderer.RENDER_VALIDATION = OUTPUT / f"RENDER_VALIDATION_ROUND{ROUND}.json"
    renderer.MANIFEST = OUTPUT / f"MANIFEST_ROUND{ROUND}.json"
    return renderer.main()


if __name__ == "__main__":
    raise SystemExit(main())
