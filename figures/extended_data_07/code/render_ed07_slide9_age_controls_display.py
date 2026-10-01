"""Render the ED7 slide-9 display from frozen scientific tables only.

This is intentionally a thin, display-only layer.  It never recomputes pair
metrics, cohort membership, or model statistics.  The scientific renderer and
its 29 no-exclusion cache JSONs remain pinned separately; this script reads the
four frozen plot/model tables and applies the current Main Figure 3 semantic
experience gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from PIL import Image

import render_ed07_slide9_age_controls as science
from ca_analysis.figure_colors import (
    CORRELATION_EXPERIENCE_COLORS,
    CORRELATION_MEASURE_COLORS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed07_age_controls"
)
SHARED_ROOT = DEFAULT_ROOT.parent
COLOR_AUTHORITY = SHARED_ROOT / "ED07_ED09_COLOR_AUTHORITY.md"
SHARED_COLOR_CROSSWALK = SHARED_ROOT / "ED07_ED09_SEMANTIC_COLOR_CROSSWALK.csv"
COLOR_AUTHORITY_VALIDATOR = REPO_ROOT / "scripts" / "validate_ed07_ed09_color_authority.py"
COLOR_AUTHORITY_VALIDATION = SHARED_ROOT / "ED07_ED09_COLOR_AUTHORITY_VALIDATION.json"
FIG3_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "figure_version_records_archive_2026-08-09"
    / "main_figure3_2026-08-08_full_audit"
    / "figure3_source_manifest.json"
)
FIGURE_COLORS = REPO_ROOT / "src" / "ca_analysis" / "figure_colors.py"
SCIENTIFIC_RENDERER = REPO_ROOT / "scripts" / "render_ed07_slide9_age_controls.py"

EXPECTED_SCIENTIFIC_RENDERER_SHA256 = (
    "524bd11f8d064c90396e9bc314e96491b31bf415a061211fa11404c242840358"
)
EXPECTED_FIG3_MANIFEST_SHA256 = (
    "0038713863766a12aa986aea711dad2c28be9e4bc3f41177be8f84f1cf4b968a"
)
EXPECTED_FIGURE_COLORS_SHA256 = (
    "1c8997b36cd9fc7db0a46d94d502a48b15ac365aa9ca39aaadc52cfc97173d53"
)

# The exact root companions were produced by the frozen scientific renderer
# before the Main Figure 3 palette was applied.  Pin both the superseded bytes
# and the already accepted corrected display bytes so synchronization fails
# closed if either source has drifted for an unrelated reason.
ROOT_COMPANION_PROVENANCE = {
    ".png": {
        "legacy_sha256": "e8b98284a8631feaede98f136b5b626f39cc5c9552695baed13badb9cb074108",
        "legacy_bytes": 829238,
        "corrected_sha256": "81599f8bd19aa2cef5cb1278d90f6fec9afe74aafa6b296d2e0589b737eac8c7",
        "corrected_bytes": 886436,
    },
    ".tiff": {
        "legacy_sha256": "0d0facb054f8d6ebe80a82cea33c202022d1f69bc61118545b8a7b835df928ef",
        "legacy_bytes": 1598922,
        "corrected_sha256": "bc3637a665758be81baca8d88c8b82bb7920343d498dc4a404924878c41552db",
        "corrected_bytes": 1690334,
    },
}

SOURCE_TABLE_NAMES = {
    "same_plot": "ed07_same_stage_plot_ready_maximal64_no_manual_mask_round1.csv",
    "same_model": "ed07_same_stage_models_maximal64_displayed_round1.csv",
    "long_plot": "ed07_long_gap_plot_ready_maximal38_round1.csv",
    "long_model": "ed07_long_gap_models_maximal38_displayed_round1.csv",
}

METRICS = (
    {
        "key": "place",
        "label": "Place map",
        "same_metric": "same_stage_place_map_corr_support_matched",
        "long_metric": "place_map_correlation",
        "semantic_role": "place_map",
        "figure3_metric": "place_map_correlation",
    },
    {
        "key": "pv",
        "label": "Population vector",
        "same_metric": "same_stage_pv_corr_support_matched",
        "long_metric": "pv_correlation",
        "semantic_role": "population_vector_or_decoder",
        "figure3_metric": "pv_correlation",
    },
    {
        "key": "rate",
        "label": "Ensemble rate",
        "same_metric": "same_stage_ensemble_rate_corr_support_matched",
        "long_metric": "ensemble_rate_correlation",
        "semantic_role": "ensemble_rate",
        "figure3_metric": "ensemble_rate_correlation",
    },
)

STAGE_KEYS = {0: "naive", 1: "3mo", 2: "6mo", 3: "9mo", 4: "12mo"}
STAGE_LABELS = {0: "naive", 1: "3 mo", 2: "6 mo", 3: "9 mo", 4: "12 mo"}
TRANSITION_LABELS = {1: "0-3 mo", 2: "3-6 mo", 3: "6-9 mo", 4: "9-12 mo"}

FORBIDDEN_DATA_COLORS = {
    "#777777",
    "#aaa9c7",
    "#8885b2",
    "#67629a",
    "#443f7a",
    "#63666a",
    "#3b82f6",
    "#f97316",
    "#8b5cf6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def file_record(path: Path, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "role": role,
        "path": rel(path),
        "sha256": sha256(path),
        "bytes": stat.st_size,
    }


def normalize(color: str) -> str:
    value = str(color).strip().lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", value):
        raise AssertionError(f"Expected six-digit hex color, got {color!r}")
    return value


def load_palette_authorities() -> tuple[dict[str, dict[int, str]], dict[str, str], dict[str, Any]]:
    assert sha256(FIG3_MANIFEST) == EXPECTED_FIG3_MANIFEST_SHA256
    assert sha256(FIGURE_COLORS) == EXPECTED_FIGURE_COLORS_SHA256
    shared_validation = json.loads(COLOR_AUTHORITY_VALIDATION.read_text(encoding="utf-8"))
    assert shared_validation["status"] == "PASS"
    assert len(shared_validation["checks"]) == 28
    assert all(row["status"] == "PASS" for row in shared_validation["checks"])

    manifest = json.loads(FIG3_MANIFEST.read_text(encoding="utf-8"))
    manifest_gradients = manifest["style"]["day2_group_colors"]
    manifest_bases = manifest["style"]["metric_colors"]
    shared = pd.read_csv(SHARED_COLOR_CROSSWALK).fillna("")
    shared = shared.loc[shared["scope"].eq("shared")].set_index("semantic_role")

    palettes: dict[str, dict[int, str]] = {}
    bases: dict[str, str] = {}
    for spec in METRICS:
        key = spec["key"]
        figure3_metric = spec["figure3_metric"]
        semantic_role = spec["semantic_role"]
        shared_row = shared.loc[semantic_role]
        palette = {
            stage: normalize(shared_row[stage_key])
            for stage, stage_key in STAGE_KEYS.items()
        }
        expected_from_manifest = {
            stage: normalize(manifest_gradients[figure3_metric][stage_key])
            for stage, stage_key in STAGE_KEYS.items()
        }
        expected_from_module = {
            stage: normalize(CORRELATION_EXPERIENCE_COLORS[figure3_metric][stage_key])
            for stage, stage_key in STAGE_KEYS.items()
        }
        assert palette == expected_from_manifest == expected_from_module

        base = normalize(shared_row["base_or_single"])
        assert base == normalize(manifest_bases[figure3_metric])
        assert base == normalize(CORRELATION_MEASURE_COLORS[figure3_metric])
        palettes[key] = palette
        bases[key] = base

    authority = {
        "shared_color_authority": file_record(COLOR_AUTHORITY, "active ED7-ED9 hard-gate contract"),
        "shared_color_crosswalk": file_record(
            SHARED_COLOR_CROSSWALK, "machine-readable shared semantic palette"
        ),
        "main_figure3_manifest": file_record(
            FIG3_MANIFEST, "accepted Main Figure 3 gradient and representation-color authority"
        ),
        "figure_colors_module": file_record(
            FIGURE_COLORS, "maintained shared semantic palette implementation"
        ),
        "shared_color_authority_validator": file_record(
            COLOR_AUTHORITY_VALIDATOR, "shared ED7-ED9 color-authority validator"
        ),
        "shared_color_authority_validation": {
            **file_record(
                COLOR_AUTHORITY_VALIDATION, "shared ED7-ED9 color-authority PASS gate"
            ),
            "status": shared_validation["status"],
            "checks_passed": len(shared_validation["checks"]),
        },
    }
    return palettes, bases, authority


def build_palette_crosswalk(
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    authority: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fig3_sha = authority["main_figure3_manifest"]["sha256"]
    shared_sha = authority["shared_color_crosswalk"]["sha256"]
    for spec in METRICS:
        for stage in range(5):
            previous_color = normalize(science.STAGE_COLORS[stage]).upper()
            corrected_color = palettes[spec["key"]][stage].upper()
            rows.append(
                {
                    "panel": "a",
                    "panel_role": "same-stage age control",
                    "metric_key": spec["key"],
                    "metric_label": spec["label"],
                    "semantic_role": spec["semantic_role"],
                    "encoded_variable": "same-stage experience",
                    "source_stage_index": stage,
                    "destination_stage_index": stage,
                    "display_label": STAGE_LABELS[stage],
                    "previous_source_hex_color": previous_color,
                    "corrected_hex_color": corrected_color,
                    "hex_color": corrected_color,
                    "color_changed": previous_color != corrected_color,
                    "previous_palette_source": "render_ed07_slide9_age_controls.py::STAGE_COLORS",
                    "previous_palette_source_sha256": EXPECTED_SCIENTIFIC_RENDERER_SHA256,
                    "palette_defect": "generic same-stage source palette replaced by metric-specific experience gradient",
                    "base_title_color": bases[spec["key"]].upper(),
                    "marker_shape_encoding": "circle=sorted; diamond=original recomputed",
                    "line_semantics": "common age slope rendered within the stage age range",
                    "figure3_manifest_json_key": (
                        f"style.day2_group_colors.{spec['figure3_metric']}.{STAGE_KEYS[stage]}"
                    ),
                    "authority_path": authority["main_figure3_manifest"]["path"],
                    "authority_sha256": fig3_sha,
                    "shared_crosswalk_path": authority["shared_color_crosswalk"]["path"],
                    "shared_crosswalk_sha256": shared_sha,
                    "required_in_svg": True,
                }
            )
        for experience in range(1, 5):
            previous_color = normalize(science.TRANSITION_COLORS[experience]).upper()
            corrected_color = palettes[spec["key"]][experience].upper()
            rows.append(
                {
                    "panel": "b",
                    "panel_role": "long-gap age control",
                    "metric_key": spec["key"],
                    "metric_label": spec["label"],
                    "semantic_role": spec["semantic_role"],
                    "encoded_variable": "destination experience",
                    "source_stage_index": experience - 1,
                    "destination_stage_index": experience,
                    "display_label": TRANSITION_LABELS[experience],
                    "previous_source_hex_color": previous_color,
                    "corrected_hex_color": corrected_color,
                    "hex_color": corrected_color,
                    "color_changed": previous_color != corrected_color,
                    "previous_palette_source": "render_ed07_slide9_age_controls.py::TRANSITION_COLORS",
                    "previous_palette_source_sha256": EXPECTED_SCIENTIFIC_RENDERER_SHA256,
                    "palette_defect": "generic transition source palette replaced by metric-specific destination-experience gradient",
                    "base_title_color": bases[spec["key"]].upper(),
                    "marker_shape_encoding": "circle=maintained parity36; diamond=recovered original",
                    "line_semantics": "common age slope rendered within the transition age range",
                    "figure3_manifest_json_key": (
                        f"style.day2_group_colors.{spec['figure3_metric']}.{STAGE_KEYS[experience]}"
                    ),
                    "authority_path": authority["main_figure3_manifest"]["path"],
                    "authority_sha256": fig3_sha,
                    "shared_crosswalk_path": authority["shared_color_crosswalk"]["path"],
                    "shared_crosswalk_sha256": shared_sha,
                    "required_in_svg": True,
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 27
    assert frame["color_changed"].all()
    return frame


def tuple_handle(colors: list[str]) -> tuple[Line2D, ...]:
    return tuple(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=3.3,
        )
        for color in colors
    )


def same_stage_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    result: pd.Series,
    spec: dict[str, str],
    palette: dict[int, str],
    base_color: str,
    *,
    show_ylabel: bool,
) -> None:
    metric = spec["same_metric"]
    beta = float(result["beta_per_30d"])
    for stage, group in frame.groupby("experience_index", sort=True):
        stage = int(stage)
        color = palette[stage]
        original = group["source_scope"].astype(str).str.startswith("original") | group[
            "neural_id_space"
        ].astype(str).eq("presort")
        for is_original, source_group in group.groupby(original, sort=True):
            marker = "D" if bool(is_original) else "o"
            ax.scatter(
                source_group["age_day1_days"],
                source_group[metric],
                s=15 if marker == "D" else 11,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        x = group["age_day1_days"].to_numpy(dtype=float)
        y = group[metric].to_numpy(dtype=float)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
        line_y = float(np.mean(y)) + beta * (line_x - float(np.mean(x))) / 30.0
        ax.plot(line_x, line_y, color=color, linewidth=1.0, zorder=2)
    ax.set_title(spec["label"], loc="left", fontweight="bold", color=base_color, pad=1.6)
    ax.set_xlabel("Age on Day 1 (days)")
    if show_ylabel:
        ax.set_ylabel("Same-stage Day 1-Day 2\ncorrelation (r)")
    ax.set_xlim(80, 680)
    ax.set_xticks([120, 300, 480, 660])
    ax.set_ylim(-0.04, 1.0)


def value_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    low, high = float(np.min(finite)), float(np.max(finite))
    padding = max((high - low) * 0.12, 0.025)
    return low - padding, high + padding


def long_gap_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    result: pd.Series,
    spec: dict[str, str],
    palette: dict[int, str],
    base_color: str,
    *,
    show_ylabel: bool,
) -> None:
    metric = spec["long_metric"]
    subset = frame.loc[frame["metric"].eq(metric)].copy()
    beta = float(result["beta_per_30d"])
    for experience, group in subset.groupby("experience", sort=True):
        experience = int(experience)
        color = palette[experience]
        x = group["age_dst_days"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        for original_source, source_group in group.groupby(
            "original_source_indicator", sort=True
        ):
            is_original = bool(int(original_source))
            ax.scatter(
                source_group["age_dst_days"],
                source_group["value"],
                s=15 if is_original else 11,
                marker="D" if is_original else "o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        if len(x) >= 3 and np.std(x) > 0:
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
            line_y = float(np.mean(y)) + beta * (line_x - float(np.mean(x))) / 30.0
            ax.plot(line_x, line_y, color=color, linewidth=1.0, zorder=2)
    ax.set_title(spec["label"], loc="left", fontweight="bold", color=base_color, pad=1.6)
    ax.set_xlabel("Destination-session age (days)")
    if show_ylabel:
        ax.set_ylabel("Long-gap cross-session\ncorrelation (r)")
    ax.set_xlim(175, 680)
    ax.set_xticks([200, 350, 500, 650])
    ax.set_ylim(*value_limits(subset["value"]))


def render_top_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
) -> list[Path]:
    lookup = models.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=science.TOP_ROW_SIZE_IN, sharex=True, sharey=True)
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        same_stage_axis(
            ax,
            frame,
            lookup.loc[spec["same_metric"]],
            spec,
            palettes[spec["key"]],
            bases[spec["key"]],
            show_ylabel=index == 0,
        )
    stage_handles = [
        tuple_handle([palettes[spec["key"]][stage] for spec in METRICS])
        for stage in range(5)
    ]
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", markersize=3.8),
        Line2D([0], [0], marker="D", linestyle="", color="#555555", markersize=3.8),
    ]
    handles: list[Any] = stage_handles + source_handles
    labels = [STAGE_LABELS[stage] for stage in range(5)] + ["sorted", "original recomputed"]
    fig.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        title="Experience (place / PV / rate)",
        title_fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.985),
        ncol=7,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.72,
    )
    fig.text(0.012, 0.965, "a", fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.16, top=0.83, wspace=0.23)
    return science.save_figure(fig, stem)


def render_bottom_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
) -> list[Path]:
    lookup = models.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=science.BOTTOM_ROW_SIZE_IN, sharex=True)
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        long_gap_axis(
            ax,
            frame,
            lookup.loc[spec["long_metric"]],
            spec,
            palettes[spec["key"]],
            bases[spec["key"]],
            show_ylabel=index == 0,
        )
    transition_handles = [
        tuple_handle([palettes[spec["key"]][experience] for spec in METRICS])
        for experience in range(1, 5)
    ]
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", markersize=3.8),
        Line2D([0], [0], marker="D", linestyle="", color="#555555", markersize=3.8),
    ]
    handles: list[Any] = transition_handles + source_handles
    labels = [TRANSITION_LABELS[experience] for experience in range(1, 5)] + [
        "maintained parity36",
        "recovered original",
    ]
    fig.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        title="Destination experience (place / PV / rate)",
        title_fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=6,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.82,
    )
    fig.text(0.012, 0.965, "b", fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.16, top=0.83, wspace=0.24)
    return science.save_figure(fig, stem)


def svg_text(svg_path: Path) -> str:
    root = ET.parse(svg_path).getroot()
    values: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "text":
            values.extend(text.strip() for text in element.itertext() if text.strip())
    return " ".join(values)


def svg_colors(svg_path: Path) -> set[str]:
    return {value.lower() for value in re.findall(r"#[0-9a-fA-F]{6}", svg_path.read_text(encoding="utf-8"))}


def validate_svg_palette(
    svg_path: Path,
    required: set[str],
    known_semantic_colors: set[str],
) -> dict[str, Any]:
    observed = svg_colors(svg_path)
    required = {normalize(value) for value in required}
    missing = sorted(required - observed)
    forbidden = sorted(FORBIDDEN_DATA_COLORS & observed)
    unexpected_semantic = sorted((observed & known_semantic_colors) - required)
    assert not missing, f"Missing required colors in {svg_path.name}: {missing}"
    assert not forbidden, f"Forbidden colors in {svg_path.name}: {forbidden}"
    assert not unexpected_semantic, (
        f"Unexpected semantic colors in {svg_path.name}: {unexpected_semantic}"
    )
    return {
        "path": rel(svg_path),
        "observed_hex_colors": sorted(observed),
        "required_hex_colors": sorted(required),
        "missing_required": missing,
        "forbidden_observed": forbidden,
        "unexpected_semantic": unexpected_semantic,
        "status": "PASS",
    }


def read_cache_gate(root: Path) -> dict[str, Any]:
    target_root = (
        root
        / "assembly"
        / "source_tables"
        / "same_stage_original_target164_no_exclusions_2026-08-08"
    )
    native117_root = (
        root
        / "assembly"
        / "source_tables"
        / "same_stage_original_native117_no_exclusions_2026-08-08"
    )
    cache_paths = sorted(target_root.rglob("*.json")) + sorted(native117_root.rglob("*.json"))
    assert len(cache_paths) == 29
    rows = []
    for path in cache_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["cache_provenance"]
        pair = payload["pair_row"]
        assert provenance["wrapper_renderer_sha256"] == EXPECTED_SCIENTIFIC_RENDERER_SHA256
        assert provenance["exclusion_profile"] == "none"
        assert bool(provenance["exclusion_map_empty"])
        assert int(pair["n_neurons_day1_raw"]) == int(pair["n_neurons_after_profile_exclusions"])
        rows.append(file_record(path, "frozen no-exclusion pair cache"))
    return {
        "cache_count": len(cache_paths),
        "wrapper_sha256": EXPECTED_SCIENTIFIC_RENDERER_SHA256,
        "exclusion_profile": "none",
        "profile_excluded_neurons": 0,
        "cache_files": rows,
    }


def image_record(path: Path) -> dict[str, Any]:
    record = file_record(path, "display output")
    if path.suffix.lower() in {".png", ".tiff"}:
        with Image.open(path) as image:
            record["pixel_dimensions"] = list(image.size)
            record["dpi"] = [float(item) for item in image.info.get("dpi", (0, 0))]
    return record


def add_check(checks: list[dict[str, str]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def write_notes(
    root: Path,
    display_dir: Path,
    visual_status: str,
    renderer_sha: str,
    authority: dict[str, Any],
    root_companions: dict[str, Path],
    root_records_before: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    readme = root / "README_DISPLAY_ROUND1.md"
    visual = root / "VISUAL_QA_DISPLAY_ROUND1.md"
    readme.write_text(
        "# ED7 slide 9 final display layer (round 1)\n\n"
        "This display layer reads the frozen maximal64 same-stage and maximal38 long-gap "
        "plot-ready/model tables. It does not recompute scientific values or modify the 29 "
        "no-exclusion caches.\n\n"
        f"- Scientific renderer SHA256: `{EXPECTED_SCIENTIFIC_RENDERER_SHA256}`\n"
        f"- Display renderer SHA256: `{renderer_sha}`\n"
        f"- Main Figure 3 palette manifest SHA256: "
        f"`{authority['main_figure3_manifest']['sha256']}`\n"
        f"- Final display assets: `{rel(display_dir)}`\n"
        "- Visible inventory: exactly two stacked rows and three metrics per row.\n"
        "- Point/segment colors use the exact metric-specific Main Figure 3 experience "
        "gradients. Titles use the representation base colors.\n"
        f"- Exact root PNG companion: `{rel(root_companions['png'])}`; SHA256 "
        f"`{sha256(root_companions['png'])}`.\n"
        f"- Exact root TIFF companion: `{rel(root_companions['tiff'])}`; SHA256 "
        f"`{sha256(root_companions['tiff'])}`.\n"
        "- Both root companions are byte-identical to the corrected compact display assembly.\n"
        "- Superseded generic-palette root hashes: PNG "
        f"`{ROOT_COMPANION_PROVENANCE['.png']['legacy_sha256']}`; TIFF "
        f"`{ROOT_COMPANION_PROVENANCE['.tiff']['legacy_sha256']}`.\n"
        f"- This run's pre-synchronization hashes: PNG "
        f"`{root_records_before['png']['sha256']}`; TIFF "
        f"`{root_records_before['tiff']['sha256']}`.\n"
        "- Exact required/forbidden SVG-color checks are recorded in "
        "`assembly/ED07_DISPLAY_MANIFEST_ROUND1.json`.\n\n"
        "The earlier scientific-renderer artwork with generic/ad hoc point palettes is "
        "superseded as a display source. The frozen scientific tables, statistics, panel "
        "inventory, and PowerPoint file were not changed.\n",
        encoding="utf-8",
    )
    if visual_status == "pass":
        visual_detail = (
            "Full-resolution inspection of the two standalone rows, compact assembly, exact "
            "root PNG companion, and 7.5-inch slide-geometry mockup is complete.\n\n"
            "- The exact six PowerPoint roles are present in the original two-by-three layout.\n"
            "- Place map, population vector, and ensemble rate use their red, orange, and "
            "blue-grey experience gradients in both rows.\n"
            "- Titles, tuple legends, points, axes, labels, and panel letters are intact.\n"
            "- No clipping, overlap, stray generic-palette color, or unexplained visual object "
            "was observed.\n"
        )
    else:
        visual_detail = (
            "Inspect both standalone row PNG/SVG exports, the compact assembly, the exact root "
            "PNG companion, and the 7.5-inch slide-geometry mockup. Confirm all titles, tuple "
            "legends, points, axes, and panel letters are intact at full resolution before any "
            "future PowerPoint integration.\n"
        )
    visual.write_text(
        "# ED7 display visual QA (round 1)\n\n"
        f"Status: **{visual_status.upper()}**.\n\n"
        f"{visual_detail}",
        encoding="utf-8",
    )
    return readme, visual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--visual-qa-status", choices=("pending", "pass"), default="pending")
    args = parser.parse_args()

    root = args.out_dir.resolve()
    assert args.round == 1, "Root source-companion synchronization is intentionally round-1 only"
    source_dir = root / "assembly" / "source_tables"
    display_dir = root / "assembly" / f"display_round{args.round}"
    display_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "assembly" / f"ED07_DISPLAY_MANIFEST_ROUND{args.round}.json"
    validation_path = root / "assembly" / f"ED07_DISPLAY_VALIDATION_ROUND{args.round}.csv"
    crosswalk_path = root / "assembly" / f"ED07_PALETTE_CROSSWALK_ROUND{args.round}.csv"
    root_companions = {
        "png": root / "extended_data_figure7_source_rerender_round1.png",
        "tiff": root / "extended_data_figure7_source_rerender_round1.tiff",
    }
    for path in root_companions.values():
        assert path.is_file(), path
    root_records_before = {
        key: file_record(path, "pre-synchronization ED7 root source companion")
        for key, path in root_companions.items()
    }
    root_before_classification: dict[str, str] = {}
    for key, path in root_companions.items():
        provenance = ROOT_COMPANION_PROVENANCE[path.suffix.lower()]
        observed = root_records_before[key]
        if observed["sha256"] == provenance["legacy_sha256"]:
            assert observed["bytes"] == provenance["legacy_bytes"]
            root_before_classification[key] = "superseded_generic_palette"
        elif observed["sha256"] == provenance["corrected_sha256"]:
            assert observed["bytes"] == provenance["corrected_bytes"]
            root_before_classification[key] = "already_synchronized_corrected_palette"
        else:
            raise AssertionError(
                f"Unexpected pre-synchronization root companion drift: {path} "
                f"{observed['sha256']} ({observed['bytes']} bytes)"
            )

    assert sha256(SCIENTIFIC_RENDERER) == EXPECTED_SCIENTIFIC_RENDERER_SHA256
    input_paths = {key: source_dir / name for key, name in SOURCE_TABLE_NAMES.items()}
    for path in input_paths.values():
        assert path.is_file(), path
    input_hashes_before = {key: sha256(path) for key, path in input_paths.items()}

    same_plot = pd.read_csv(input_paths["same_plot"], dtype={"animal_id": str})
    same_model = pd.read_csv(input_paths["same_model"])
    long_plot = pd.read_csv(input_paths["long_plot"], dtype={"animal_id": str})
    long_model = pd.read_csv(input_paths["long_model"])
    assert len(same_plot) == 64 and same_plot["animal_id"].nunique() == 21
    assert len(same_model) == 3
    assert len(long_plot) == 114 and long_plot["animal_id"].nunique() == 14
    assert len(long_model) == 3
    assert set(long_plot.groupby("metric").size()) == {38}

    palettes, bases, authority = load_palette_authorities()
    crosswalk = build_palette_crosswalk(palettes, bases, authority)
    crosswalk.to_csv(crosswalk_path, index=False)

    science.configure_style()
    top_stem = display_dir / f"ed07_age_controls_panel_a_same_stage_mainfig3_palette_round{args.round}"
    bottom_stem = display_dir / f"ed07_age_controls_panel_b_long_gap_mainfig3_palette_round{args.round}"
    assembly_stem = display_dir / f"ed07_age_controls_assembly_mainfig3_palette_round{args.round}"
    mockup_path = display_dir / f"ed07_slide9_geometry_mockup_mainfig3_palette_round{args.round}.png"

    top_outputs = render_top_row(same_plot, same_model, palettes, bases, top_stem)
    bottom_outputs = render_bottom_row(long_plot, long_model, palettes, bases, bottom_stem)
    assembly_outputs = science.compose_compact_assembly(
        top_stem.with_suffix(".png"), bottom_stem.with_suffix(".png"), assembly_stem
    )
    science.compose_slide_mockup(
        top_stem.with_suffix(".png"), bottom_stem.with_suffix(".png"), mockup_path
    )

    assembly_companions = {
        "png": assembly_stem.with_suffix(".png"),
        "tiff": assembly_stem.with_suffix(".tiff"),
    }
    for key, assembly_path in assembly_companions.items():
        provenance = ROOT_COMPANION_PROVENANCE[assembly_path.suffix.lower()]
        assert sha256(assembly_path) == provenance["corrected_sha256"], assembly_path
        assert assembly_path.stat().st_size == provenance["corrected_bytes"], assembly_path
        shutil.copy2(assembly_path, root_companions[key])
        assert root_companions[key].read_bytes() == assembly_path.read_bytes()

    root_records_after = {
        key: file_record(path, "palette-corrected ED7 root source companion")
        for key, path in root_companions.items()
    }

    input_hashes_after = {key: sha256(path) for key, path in input_paths.items()}
    assert input_hashes_before == input_hashes_after
    cache_gate = read_cache_gate(root)
    scientific_bundle_paths = {
        "scientific_assembly_manifest": root / "assembly" / "ED07_ASSEMBLY_MANIFEST_ROUND1.json",
        "source_data_audit": root / "assembly" / "ED07_SOURCE_DATA_AUDIT_ROUND1.csv",
        "caption_statistics": root / "assembly" / "ED07_CAPTION_STATISTICS_ROUND1.csv",
        "caption_notes": root / "CAPTION_NOTES_ROUND1.md",
        "source_selection_validation": root / "AUDIT_VALIDATION.json",
    }
    for path in scientific_bundle_paths.values():
        assert path.is_file(), path
    scientific_manifest = json.loads(
        scientific_bundle_paths["scientific_assembly_manifest"].read_text(encoding="utf-8")
    )
    caption_statistics = pd.read_csv(scientific_bundle_paths["caption_statistics"])
    source_selection_validation = json.loads(
        scientific_bundle_paths["source_selection_validation"].read_text(encoding="utf-8")
    )
    assert scientific_manifest["renderer_sha256"] == EXPECTED_SCIENTIFIC_RENDERER_SHA256
    assert source_selection_validation["status"] == "PASS"
    assert len(caption_statistics) == 6

    known_semantic_colors = {
        normalize(value)
        for column in ("naive", "3mo", "6mo", "9mo", "12mo", "base_or_single")
        for value in pd.read_csv(SHARED_COLOR_CROSSWALK).fillna("")[column]
        if str(value).startswith("#")
    } | FORBIDDEN_DATA_COLORS
    top_required = {bases[spec["key"]] for spec in METRICS} | {
        palettes[spec["key"]][stage] for spec in METRICS for stage in range(5)
    }
    bottom_required = {bases[spec["key"]] for spec in METRICS} | {
        palettes[spec["key"]][experience]
        for spec in METRICS
        for experience in range(1, 5)
    }
    top_svg_gate = validate_svg_palette(
        top_stem.with_suffix(".svg"), top_required, known_semantic_colors
    )
    bottom_svg_gate = validate_svg_palette(
        bottom_stem.with_suffix(".svg"), bottom_required, known_semantic_colors
    )

    checks: list[dict[str, str]] = []
    add_check(
        checks,
        "scientific_renderer_sha_frozen",
        sha256(SCIENTIFIC_RENDERER) == EXPECTED_SCIENTIFIC_RENDERER_SHA256,
        EXPECTED_SCIENTIFIC_RENDERER_SHA256,
    )
    add_check(
        checks,
        "scientific_input_tables_unchanged_by_display",
        input_hashes_before == input_hashes_after,
        "four plot/model table SHA256 values identical before and after display render",
    )
    add_check(checks, "scientific_cache_gate_29", cache_gate["cache_count"] == 29, "29/29 caches")
    add_check(checks, "same_stage_maximal64", len(same_plot) == 64, "64 rows / 21 mice")
    add_check(checks, "long_gap_maximal38", len(long_plot) == 114, "38 rows x 3 displayed metrics")
    add_check(checks, "palette_crosswalk_exact_rows", len(crosswalk) == 27, "15 top + 12 bottom mappings")
    add_check(
        checks,
        "palette_crosswalk_all_27_corrected",
        bool(crosswalk["color_changed"].all()),
        "all 27 displayed panel/metric/stage mappings changed from generic to metric-specific",
    )
    add_check(
        checks,
        "root_companion_inputs_recognized",
        set(root_before_classification.values()).issubset(
            {"superseded_generic_palette", "already_synchronized_corrected_palette"}
        ),
        json.dumps(root_before_classification, sort_keys=True),
    )
    for key in ("png", "tiff"):
        add_check(
            checks,
            f"root_{key}_byte_equal_to_display_assembly",
            sha256(root_companions[key]) == sha256(assembly_companions[key])
            and root_companions[key].read_bytes() == assembly_companions[key].read_bytes(),
            f"{sha256(root_companions[key])}; exact byte equality",
        )
    add_check(
        checks,
        "shared_color_authority_25_of_25",
        authority["shared_color_authority_validation"]["status"] == "PASS"
        and authority["shared_color_authority_validation"]["checks_passed"] == 25,
        "shared validator and frozen-authority hashes PASS 25/25",
    )
    add_check(
        checks,
        "caption_statistics_source_locked_six_rows",
        len(caption_statistics) == 6,
        "three displayed metrics x two panels; inference remains caption-only",
    )
    add_check(checks, "top_svg_required_colors", top_svg_gate["status"] == "PASS", "all required; no forbidden")
    add_check(checks, "bottom_svg_required_colors", bottom_svg_gate["status"] == "PASS", "all required; no forbidden")

    for label, stem in (("top", top_stem), ("bottom", bottom_stem), ("assembly", assembly_stem)):
        outputs = [stem.with_suffix(f".{extension}") for extension in science.FORMATS]
        add_check(checks, f"{label}_formats_complete", all(path.is_file() and path.stat().st_size > 0 for path in outputs), ", ".join(path.suffix for path in outputs))
    for label, path, expected in (
        ("top", top_stem.with_suffix(".png"), (4290, 1950)),
        ("bottom", bottom_stem.with_suffix(".png"), (4290, 1950)),
        ("assembly", assembly_stem.with_suffix(".png"), (4500, 4092)),
        ("root_png", root_companions["png"], (4500, 4092)),
        ("root_tiff", root_companions["tiff"], (4500, 4092)),
        ("mockup", mockup_path, (4500, 4500)),
    ):
        with Image.open(path) as image:
            add_check(checks, f"{label}_pixel_dimensions", image.size == expected, f"{image.size}; expected {expected}")

    for label, path, letter in (
        ("top", top_stem.with_suffix(".svg"), "a"),
        ("bottom", bottom_stem.with_suffix(".svg"), "b"),
    ):
        text = svg_text(path)
        raw = path.read_text(encoding="utf-8")
        add_check(checks, f"{label}_svg_arial", "Arial" in raw, "editable Arial SVG text")
        add_check(checks, f"{label}_titles_complete", all(spec["label"] in text for spec in METRICS), "three representation titles")
        add_check(checks, f"{label}_panel_letter_once", text.split().count(letter) == 1, f"one embedded {letter}")
        no_stats = re.search(
            r"(?i)(?:\bbeta\b|\bp\s*=|\bq\s*=|\bn\s*=|\*{1,3})", text
        ) is None
        add_check(checks, f"{label}_no_inferential_annotations", no_stats, "statistics remain caption-only")

    hard_failures = [row for row in checks if row["status"] != "PASS"]
    assert not hard_failures
    visual_status = "pass" if args.visual_qa_status == "pass" else "pending_manual_visual_qa"
    checks.append(
        {
            "check": "manual_visual_qa",
            "status": "PASS" if args.visual_qa_status == "pass" else "PENDING",
            "detail": "fresh palette-corrected rows, assembly and exact mockup inspected"
            " together with the synchronized root companion"
            if args.visual_qa_status == "pass"
            else "fresh palette-corrected outputs require full-resolution inspection",
        }
    )
    pd.DataFrame(checks).to_csv(validation_path, index=False)

    renderer_sha = sha256(Path(__file__).resolve())
    readme_path, visual_path = write_notes(
        root,
        display_dir,
        visual_status,
        renderer_sha,
        authority,
        root_companions,
        root_records_before,
    )
    output_paths = (
        top_outputs
        + bottom_outputs
        + assembly_outputs
        + [mockup_path, root_companions["png"], root_companions["tiff"]]
    )
    manifest = {
        "figure": "Extended Data Figure 7; slide 9",
        "round": args.round,
        "status": visual_status,
        "final_display_authority": True,
        "display_renderer": file_record(Path(__file__).resolve(), "thin palette-locked display renderer"),
        "scientific_renderer": file_record(SCIENTIFIC_RENDERER, "frozen science/cache renderer"),
        "scientific_cache_recomputed_by_display_renderer": False,
        "scientific_values_changed_by_display_renderer": False,
        "powerpoint_image_assets_changed": False,
        "scientific_cache_gate": cache_gate,
        "scientific_inputs": [
            {**file_record(path, key), "rows": int(len(pd.read_csv(path)))}
            for key, path in input_paths.items()
        ],
        "scientific_bundle_records": {
            key: file_record(path, key.replace("_", " "))
            for key, path in scientific_bundle_paths.items()
        },
        "palette_authorities": authority,
        "palette": {
            "representation_base_colors": {key: value.upper() for key, value in bases.items()},
            "metric_specific_experience_gradients": {
                key: {STAGE_KEYS[stage]: value.upper() for stage, value in palette.items()}
                for key, palette in palettes.items()
            },
            "long_gap_destination_mapping": {
                TRANSITION_LABELS[experience]: STAGE_LABELS[experience]
                for experience in range(1, 5)
            },
            "forbidden_data_colors": sorted(color.upper() for color in FORBIDDEN_DATA_COLORS),
            "crosswalk": file_record(crosswalk_path, "ED7 panel/metric/stage palette crosswalk"),
            "top_svg_gate": top_svg_gate,
            "bottom_svg_gate": bottom_svg_gate,
        },
        "root_source_companion_synchronization": {
            "status": "PASS",
            "superseded_generic_palette_records": {
                key: {
                    "path": rel(path),
                    "sha256": ROOT_COMPANION_PROVENANCE[path.suffix.lower()][
                        "legacy_sha256"
                    ],
                    "bytes": ROOT_COMPANION_PROVENANCE[path.suffix.lower()][
                        "legacy_bytes"
                    ],
                }
                for key, path in root_companions.items()
            },
            "pre_render_records": root_records_before,
            "pre_render_classification": root_before_classification,
            "pinned_legacy_and_corrected_provenance": ROOT_COMPANION_PROVENANCE,
            "corrected_root_records": root_records_after,
            "corrected_assembly_records": {
                key: file_record(path, "palette-corrected compact display assembly")
                for key, path in assembly_companions.items()
            },
            "byte_equal_to_assembly": {
                key: root_companions[key].read_bytes() == assembly_companions[key].read_bytes()
                for key in root_companions
            },
        },
        "panel_inventory": {
            "a": {"role": "same-stage age control", "visible_metrics": [spec["key"] for spec in METRICS]},
            "b": {"role": "long-gap age control", "visible_metrics": [spec["key"] for spec in METRICS]},
        },
        "slide_geometry_inches": {
            "slide": list(science.SLIDE_SIZE_IN),
            "top_slot": list(science.TOP_SLOT_IN),
            "bottom_slot": list(science.BOTTOM_SLOT_IN),
            "source_row_media": list(science.ROW_SIZE_IN),
            "crop": None,
        },
        "outputs": [image_record(path) for path in output_paths],
        "validation": checks,
        "validation_file": file_record(validation_path, "display hard-gate validation"),
        "readme": file_record(readme_path, "display authority note"),
        "visual_qa": file_record(visual_path, "fresh palette-corrected visual QA status"),
        "powerpoint_edited": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
