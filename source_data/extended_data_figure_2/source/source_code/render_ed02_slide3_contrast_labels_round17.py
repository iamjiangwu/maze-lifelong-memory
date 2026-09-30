"""Render ED2 round 17 with the panel-g contrast key raised by 14 pixels.

The accepted round-13 renderer remains the source and layout authority.  This
append-only wrapper changes only the lower panel-g contrast legend anchor; all
scientific artists, axes, panel frames, and the upper legend are inherited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_panel_g_labels_round13.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-10_round16_publication"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round17_contrast_labels_2026-08-10" / "panels"
PANEL_G_CONTRAST_ANCHOR = (0.58, 1.06513460)
PANEL_G_CONTRAST_RAISE_PX_600 = 14.0


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round13_for_round17", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
PARENT_BUILD_PANEL_G = BASE.build_panel_g


def build_panel_g(renderer, policy_rows, contrast_frame):
    fig, tables = PARENT_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    if len(fig.axes) < 2:
        raise RuntimeError("Panel g no longer exposes its two maintained axes")
    lower_axis = fig.axes[1]
    legend = lower_axis.get_legend()
    if legend is None:
        raise RuntimeError("Panel-g contrast legend is missing")
    legend.set_bbox_to_anchor(PANEL_G_CONTRAST_ANCHOR, transform=lower_axis.transAxes)
    return fig, tables


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    original = BASE.build_panel_g
    try:
        BASE.build_panel_g = build_panel_g
        result = BASE.render(record_root, output_root, dpi)
    finally:
        BASE.build_panel_g = original

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 17
    manifest["status"] = "review_candidate_not_current_version_record"
    manifest["panel_g_contrast_key_round17"] = {
        "source_anchor_axes": [0.58, 1.02],
        "target_anchor_axes": list(PANEL_G_CONTRAST_ANCHOR),
        "target_raise_px_at_600dpi": PANEL_G_CONTRAST_RAISE_PX_600,
        "changed_artists": [
            "Return day − Naive day",
            "Second day − Return day",
        ],
        "upper_legend_changed": False,
        "axes_or_scientific_artists_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result["review_round"] = 17
    result["panel_g_contrast_key_round17"] = manifest["panel_g_contrast_key_round17"]
    (output_root / "RENDER_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args.record_root, args.output_root, args.dpi), indent=2))


if __name__ == "__main__":
    main()
