#!/usr/bin/env python
"""Validate the frozen semantic-color authority for rebuilt ED7-ED9.

This validator checks the current accepted main-figure records, the shared palette,
and the rebuild-local color crosswalk. Panel renderers add their own SVG-level
required/allowed/forbidden color checks on top of this authority check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
REBUILD = REPO / "docs" / "extended_data_supplement_v2_rebuild_2026-08-05"
DEFAULT_OUT = REBUILD / "ED07_ED09_COLOR_AUTHORITY_VALIDATION.json"

AUTHORITIES = {
    "main3_manifest": (
        REPO
        / "docs/figure_version_records/main_figure3_2026-08-08_full_audit/figure3_source_manifest.json",
        "0038713863766a12aa986aea711dad2c28be9e4bc3f41177be8f84f1cf4b968a",
    ),
    "main4_manifest": (
        REPO
        / "docs/figure_version_records/main_figure4_2026-08-08_full_audit/figure4_source_manifest.json",
        "b2bc2aa95b81d8c104887cfd4260d8770148da4823d76bb5a19a341c7f91ad6c",
    ),
    "main5_manifest": (
        REPO
        / "docs/figure_version_records/main_figure5_2026-08-08_full_audit/figure5_source_manifest.json",
        "c69ef80e826542938088e57a5dab373eb2c99d02f85a986bf7c0a4330158e5cc",
    ),
    "main1_schematic_source": (
        REPO
        / "docs/figure_version_records/main_figure1_2026-08-08_full_audit/source_code/preview_figure1_assembly.py",
        "050865ecd4f1bdea8e4457bc6ea072c37b57a5eca19dc31c0b013a97d52edf17",
    ),
    "shared_palette": (
        REPO / "src/ca_analysis/figure_colors.py",
        "1c8997b36cd9fc7db0a46d94d502a48b15ac365aa9ca39aaadc52cfc97173d53",
    ),
    "extended_data_color_audit": (
        REPO
        / "docs/extended_data_figure_panel_refresh_2026-07-12/COLOR_CONSISTENCY_AUDIT_2026-07-16.md",
        "fcd9110baf620cc2474a698be01bca62c469b3527a9e4ce638bbf26a441fe8ab",
    ),
    "tree_depth_control_renderer": (
        REPO / "scripts/render_ed12_control_focused.py",
        "03e04e796902352968cdf94ca15c2aec4b4340ffd9cb9c82558b821e16e66185",
    ),
}

BASE_COLORS = {
    "place_map_correlation": "#b95a58",
    "pv_correlation": "#e29957",
    "ensemble_rate_correlation": "#4a5e65",
    "level_map_correlation": "#7976a2",
}

EXPERIENCE_COLORS = {
    "place_map_correlation": {
        "naive": "#d89c98",
        "3mo": "#cb817d",
        "6mo": "#bf6965",
        "9mo": "#a84e4b",
        "12mo": "#78312f",
    },
    "pv_correlation": {
        "naive": "#edbd8f",
        "3mo": "#e8a76f",
        "6mo": "#e29957",
        "9mo": "#bd7137",
        "12mo": "#8a4a1f",
    },
    "ensemble_rate_correlation": {
        "naive": "#9fb0b5",
        "3mo": "#7d9198",
        "6mo": "#637980",
        "9mo": "#4a5e65",
        "12mo": "#2c3a3f",
    },
    "level_map_correlation": {
        "naive": "#bdb9d6",
        "3mo": "#a19cc3",
        "6mo": "#8a85b2",
        "9mo": "#6f6b99",
        "12mo": "#4e4a75",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lower_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): lower_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [lower_nested(item) for item in value]
    if isinstance(value, str) and value.startswith("#"):
        return value.lower()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    for name, (path, expected_hash) in AUTHORITIES.items():
        observed = sha256(path) if path.is_file() else None
        record(f"authority_hash_{name}", observed == expected_hash, observed, expected_hash)

    main3 = json.loads(AUTHORITIES["main3_manifest"][0].read_text(encoding="utf-8"))
    observed_base = lower_nested(main3["style"]["metric_colors"])
    observed_experience = lower_nested(main3["style"]["day2_group_colors"])
    record("main3_base_colors", observed_base == BASE_COLORS, observed_base, BASE_COLORS)
    record(
        "main3_experience_colors",
        observed_experience == EXPERIENCE_COLORS,
        observed_experience,
        EXPERIENCE_COLORS,
    )

    main4 = json.loads(AUTHORITIES["main4_manifest"][0].read_text(encoding="utf-8"))
    decoder_color = str(main4["color_policy"]["decoder_color"]).lower()
    record("main4_decoder_orange", decoder_color == "#e29957", decoder_color, "#e29957")
    main4_rule = str(main4["color_policy"]["rule"])
    record(
        "main4_no_blue_d1_rule",
        "do not encode D1 with blue" in main4_rule,
        main4_rule,
        "rule explicitly forbids blue D1 encoding",
    )

    main5 = json.loads(AUTHORITIES["main5_manifest"][0].read_text(encoding="utf-8"))
    record(
        "main5_level_color",
        str(main5["style"]["level_color"]).lower() == "#7976a2",
        str(main5["style"]["level_color"]).lower(),
        "#7976a2",
    )
    record(
        "main5_level_cmap",
        str(main5["style"]["heatmap_cmap"]).lower() == "magma",
        str(main5["style"]["heatmap_cmap"]).lower(),
        "magma",
    )

    tree_text = AUTHORITIES["tree_depth_control_renderer"][0].read_text(encoding="utf-8")
    record(
        "tree_depth_shuffle_gray",
        'SHUFFLE_GRAY = "#7f7f7f"' in tree_text,
        "#7f7f7f" if 'SHUFFLE_GRAY = "#7f7f7f"' in tree_text else "not found",
        "#7f7f7f",
    )
    record(
        "tree_depth_black_magma_schematic",
        'HEATMAP_CMAP = "magma"' in tree_text and '"maze_outline": "black"' in tree_text,
        {
            "magma": 'HEATMAP_CMAP = "magma"' in tree_text,
            "black_outline": '"maze_outline": "black"' in tree_text,
        },
        {"magma": True, "black_outline": True},
    )

    crosswalk_path = REBUILD / "ED07_ED09_SEMANTIC_COLOR_CROSSWALK.csv"
    with crosswalk_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_role = {(row["scope"], row["semantic_role"]): row for row in rows}
    expected_rows = {
        ("shared", "place_map"): EXPERIENCE_COLORS["place_map_correlation"]
        | {"base_or_single": BASE_COLORS["place_map_correlation"]},
        ("shared", "population_vector_or_decoder"): EXPERIENCE_COLORS["pv_correlation"]
        | {"base_or_single": BASE_COLORS["pv_correlation"]},
        ("shared", "ensemble_rate"): EXPERIENCE_COLORS["ensemble_rate_correlation"]
        | {"base_or_single": BASE_COLORS["ensemble_rate_correlation"]},
        ("shared", "level_map"): EXPERIENCE_COLORS["level_map_correlation"]
        | {"base_or_single": BASE_COLORS["level_map_correlation"]},
        ("ED8", "reward_proximal_decoder"): {"base_or_single": "#e29957"},
        ("ED8", "main_figure_1_path_or_endpoint"): {"base_or_single": "#d82929"},
        ("ED9", "shuffle_or_null"): {"base_or_single": "#7f7f7f"},
        ("ED9", "level_schematic"): {"base_or_single": "magma"},
    }
    for key, expected in expected_rows.items():
        row = by_role.get(key)
        observed = None if row is None else {
            field: str(row.get(field, "")).lower() for field in expected
        }
        record(f"crosswalk_{key[0]}_{key[1]}", observed == expected, observed, expected)

    authority_path = REBUILD / "ED07_ED09_COLOR_AUTHORITY.md"
    authority_text = authority_path.read_text(encoding="utf-8").lower()
    missing_hashes = [
        expected_hash
        for _, expected_hash in AUTHORITIES.values()
        if expected_hash not in authority_text
    ]
    record("authority_markdown_lists_all_hashes", not missing_hashes, missing_hashes, [])

    current_main = (REPO / "docs/figure_version_records/CURRENT_MAIN_FIGURES.md").read_text(
        encoding="utf-8"
    )
    expected_records = [
        "main_figure1_2026-08-08_full_audit",
        "main_figure3_2026-08-08_full_audit",
        "main_figure4_2026-08-08_full_audit",
        "main_figure5_2026-08-08_full_audit",
    ]
    missing_records = [record_name for record_name in expected_records if record_name not in current_main]
    record("current_main_pointer_records", not missing_records, missing_records, [])

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    payload = {
        "schema": "ed07_ed09_color_authority_validation/v1",
        "status": status,
        "checks": checks,
        "authority_markdown": str(authority_path.relative_to(REPO)).replace("\\", "/"),
        "crosswalk_csv": str(crosswalk_path.relative_to(REPO)).replace("\\", "/"),
        "note": "Panel-level validators must additionally inspect final SVG semantic colors.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if status != "PASS":
        failed = [row["check"] for row in checks if row["status"] != "PASS"]
        raise SystemExit(f"Color authority validation failed: {failed}")
    print(f"PASS: {len(checks)} color-authority checks -> {args.out}")


if __name__ == "__main__":
    main()
