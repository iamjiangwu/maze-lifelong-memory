"""Export the ED2 round-18 upright-policy candidate as vector/raster masters."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_EXPORTER = REPO_ROOT / "scripts" / "export_ed02_round11_vector_figure.py"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tmp" / "ed02_round18_upright_policy_2026-08-11" / "panels"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round18_upright_policy_2026-08-11" / "composite"
COMPOSITE_STEM = "ed02_round18_upright_policy_version_record"
MANIFEST_NAME = "ED02_ROUND18_COMPOSITE_MANIFEST.json"


def _load_exporter() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round11_exporter_for_round18", BASE_EXPORTER)
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
        review_round=18,
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
