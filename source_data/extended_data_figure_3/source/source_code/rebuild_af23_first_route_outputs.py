"""Rebuild only the AF23-dependent route tables and panels.

This focused wrapper intentionally leaves the ED2 and non-route ED3 products
untouched while refreshing the central ED04 source family consumed by Main
Figure 1 and the merged Extended Data Figure 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from render_ed02_ed04_current_panels import (
    DEFAULT_REFRESH_ROOT,
    FORMATS,
    ROUTE_ROOT,
    configure_style,
    json_ready,
    plot_ed04,
    prepare_ed04,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-root", type=Path, default=DEFAULT_REFRESH_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_style()
    refresh_root = args.refresh_root.resolve()
    source_tables = refresh_root / "source_tables"
    output_dir = refresh_root / "source_rendered_new" / "extended_data_fig_04"
    source_tables.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables, table_paths, audit = prepare_ed04(source_tables)
    outputs = plot_ed04(tables, output_dir)

    frozen_analysis_sources: dict[str, Path] = {}
    for name in (
        "reward_path_first_turns.csv",
        "session_route_summary.csv",
        "entry_path_reconstruction_audit.csv",
        "entry_path_choice_reconstruction_audit.csv",
        "entry_path_frame_audit.csv",
        "run_manifest.json",
        "report.md",
    ):
        source = ROUTE_ROOT / name
        destination = source_tables / f"ed04_af23_{name}"
        shutil.copy2(source, destination)
        frozen_analysis_sources[name] = destination

    all_files = [*table_paths.values(), *outputs, *frozen_analysis_sources.values()]
    payload: dict[str, Any] = {
        "status": "af23_direct_entry_reconstruction_complete",
        "core_conclusion": (
            "The unique home-to-node-16 entry path reconstructs two pre-recording "
            "choices in 489-3day1; its chronologically resolved route score is 4/6."
        ),
        "source_analysis_root": str(ROUTE_ROOT),
        "scope_audit": audit,
        "source_tables": {key: str(path) for key, path in table_paths.items()},
        "frozen_analysis_sources": {
            key: str(path) for key, path in frozen_analysis_sources.items()
        },
        "outputs": [str(path) for path in outputs],
        "style": {
            "backend": "Python/matplotlib only",
            "raster_dpi": 600,
            "svg_text_editable": True,
            "pdf_fonttype": 42,
            "formats": list(FORMATS),
        },
        "sha256": {
            str(path.relative_to(refresh_root)).replace("\\", "/"): _sha256(path)
            for path in all_files
        },
    }
    manifest_path = source_tables / "ed04_current_render_manifest.json"
    manifest_path.write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_ready({"manifest": manifest_path, **payload}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
