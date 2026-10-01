"""Export the ED2 round-21 refinement panels as one candidate figure."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BASE_EXPORTER = REPO / "scripts" / "export_ed02_round11_vector_figure.py"
DEFAULT_SOURCE_ROOT = (
    REPO
    / "docs"
    / "figure_refinement_round2_2026-08-11"
    / "extended_data_1_5"
    / "ed02"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT
COMPOSITE_STEM = "ed02_round21_refinement_candidate"
MANIFEST_NAME = "ED02_ROUND21_COMPOSITE_MANIFEST.json"


def _load_exporter() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round11_exporter_for_round21", BASE_EXPORTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_EXPORTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(source_root: Path, output_root: Path) -> dict[str, Any]:
    return _load_exporter().export(
        source_root,
        output_root,
        composite_stem=COMPOSITE_STEM,
        review_round=21,
        manifest_name=MANIFEST_NAME,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(export(args.source_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
