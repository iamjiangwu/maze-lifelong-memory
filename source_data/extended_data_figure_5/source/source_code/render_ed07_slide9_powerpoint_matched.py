"""Render ED7 slide 9 from the PowerPoint-matched first-window estimand.

The scientific reconciliation is produced separately under
``ed07_age_controls/estimand_parity_audit``.  This script is display-only: it
reads the reconciled first-30-min tables and the maintained clean36 long gap
tables, applies the accepted Main Figure 3 semantic palettes, and
exports the two inherited three-metric rows plus a compact slide mockup.

The displayed primary cohort is locked to the maintained 58-session
first-window authority.  Six technically recoverable rows are deliberately
excluded: the five different-experiment 6-month pairs (1636, 1637, 1639, 325,
347) and the sick-animal 579-6 pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from PIL import Image

import render_ed07_slide9_age_controls as science
import render_ed07_slide9_age_controls_display as palette_display


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed07_age_controls"
)

METRICS = (
    {
        "key": "place",
        "label": "Place map",
        "metric": "place_map_correlation",
    },
    {
        "key": "pv",
        "label": "Population vector",
        "metric": "pv_correlation",
    },
    {
        "key": "rate",
        "label": "Ensemble rate",
        "metric": "ensemble_rate_correlation",
    },
)

STAGE_LABELS = {0: "Exp0 (naive)", 1: "Exp1", 2: "Exp2", 3: "Exp3", 4: "Exp4"}
TRANSITION_LABELS = {
    1: "Exp1 (0-3 mo)",
    2: "Exp2 (3-6 mo)",
    3: "Exp3 (6-9 mo)",
    4: "Exp4 (9-12 mo)",
}

MAINTAINED_FIRST30_SCOPE = "accepted_main_figure3_first30_fixed_reference"
EXCLUDED_FIRST30_SESSION_PAIRS = (
    "1636-6day1->1636-6day2",
    "1637-6day1->1637-6day2",
    "1639-6day1->1639-6day2",
    "325-6day1->325-6day2",
    "347-6day1->347-6day2",
    "579-6day1->579-6day2",
)

# Round 8 is the source authority for values, cohorts, colors, and axis ranges.
# Round 9 established the six-panel compact grammar, and round 10 removed the
# parenthetical correlation symbol. Round 11 makes the first-row day roles explicit.
ROW_SIZE_IN = (7.15, 2.55)
PRIOR_ASSEMBLY_SIZE_IN = (7.5, 5.15)
ASSEMBLY_SIZE_IN = (7.5, 4.92)
# Removing the two redundant legend-title rows releases 0.115 in per row.
# Keep the six axes at their accepted physical width/height: the lower row
# stays fixed, the inter-row gap loses one title slot, and the upper margin
# loses the other title slot.
_TITLE_SLOT_IN = (PRIOR_ASSEMBLY_SIZE_IN[1] - ASSEMBLY_SIZE_IN[1]) / 2.0
_PRIOR_AXES_HEIGHT_IN = (
    (0.86 - 0.095) * PRIOR_ASSEMBLY_SIZE_IN[1] / (2.0 + 0.72)
)
_COMPACT_BOTTOM = 0.095 * PRIOR_ASSEMBLY_SIZE_IN[1] / ASSEMBLY_SIZE_IN[1]
_COMPACT_HSPACE = (
    0.72 * _PRIOR_AXES_HEIGHT_IN - _TITLE_SLOT_IN
) / _PRIOR_AXES_HEIGHT_IN
_COMPACT_TOP = (
    0.095 * PRIOR_ASSEMBLY_SIZE_IN[1]
    + 2.0 * _PRIOR_AXES_HEIGHT_IN
    + _COMPACT_HSPACE * _PRIOR_AXES_HEIGHT_IN
) / ASSEMBLY_SIZE_IN[1]
SLIDE_SIZE_IN = (7.5, 7.5)
PANEL_LETTERS_TOP = ("a", "b", "c")
PANEL_LETTERS_BOTTOM = ("d", "e", "f")
TOP_YLABEL = "Day 1 early vs Day 2 correlation"
BOTTOM_YLABEL = "Long gap correlation"
X_LABEL = "Age (days)"


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


def pick(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    raise KeyError(f"None of {tuple(names)} is present and finite")


def normalize_first30_plot(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename = {}
    if "animal" in result.columns and "animal_id" not in result.columns:
        rename["animal"] = "animal_id"
    if "first_value" in result.columns and "value" not in result.columns:
        rename["first_value"] = "value"
    result = result.rename(columns=rename)
    required = {"animal_id", "experience_index", "age_day1_days", "metric", "value"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise AssertionError(f"First30 plot table is missing {missing}")
    for column in ("experience_index", "age_day1_days", "value"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(required)].isna().any().any():
        raise AssertionError("First30 plot table has missing primary fields")
    result["animal_id"] = result["animal_id"].astype(str)
    if "source_scope" not in result.columns:
        result["source_scope"] = "maintained_first30_authority"
    result["source_scope"] = result["source_scope"].astype(str)
    return result


def model_lookup(models: pd.DataFrame, specification: str) -> pd.DataFrame:
    subset = models.copy()
    if "specification" in subset.columns:
        subset = subset.loc[subset["specification"].astype(str).eq(specification)]
    if len(subset) != 3:
        raise AssertionError(
            f"Expected three {specification} displayed models, found {len(subset)}"
        )
    return subset.set_index("metric", drop=False)


def model_values(row: pd.Series, *, q_family: str) -> dict[str, float | int]:
    beta = float(pick(row, ("beta_per_30d", "beta_age_per_30d")))
    p_value = float(
        pick(
            row,
            (
                "p_cluster",
                "p_cluster_two_sided",
                "p_CR1_student_t_G_minus_1",
            ),
        )
    )
    q_candidates = (
        "q_bh3",
        "q_bh3_displayed",
        "q_BH3_age",
        "p_cluster_q_bh3",
    ) if q_family == "BH3" else (
        "p_cluster_q_bh4",
        "q_bh4",
        "q_BH4_age",
    )
    q_value = float(pick(row, q_candidates))
    return {
        "beta": beta,
        "p": p_value,
        "q": q_value,
        "n": int(pick(row, ("n_rows", "n_observations"))),
        "mice": int(pick(row, ("n_animals", "n_mice"))),
    }


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


def add_panel_letters(axes: Iterable[plt.Axes], letters: Iterable[str]) -> None:
    for ax, letter in zip(axes, letters):
        ax.text(
            -0.18,
            1.08,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="black",
            clip_on=False,
        )


def draw_first30_axes(
    axes: Iterable[plt.Axes],
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
) -> None:
    axes = list(axes)
    lookup = model_lookup(models, "maintained58_primary")
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        subset = frame.loc[frame["metric"].eq(spec["metric"])].copy()
        if len(subset) != 58 or subset["animal_id"].nunique() != 21:
            raise AssertionError(
                f"{spec['metric']} is not maintained58/21: {len(subset)} rows, "
                f"{subset['animal_id'].nunique()} mice"
            )
        model = model_values(lookup.loc[spec["metric"]], q_family="BH3")
        for stage, group in subset.groupby("experience_index", sort=True):
            stage = int(stage)
            color = palettes[spec["key"]][stage]
            ax.scatter(
                group["age_day1_days"],
                group["value"],
                s=12,
                marker="o",
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
                zorder=3,
            )
            x = group["age_day1_days"].to_numpy(dtype=float)
            y = group["value"].to_numpy(dtype=float)
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
            line_y = float(np.mean(y)) + float(model["beta"]) * (
                line_x - float(np.mean(x))
            ) / 30.0
            ax.plot(line_x, line_y, color=color, linewidth=1.05, zorder=2)
        ax.set_title(
            spec["label"],
            loc="left",
            fontweight="bold",
            color=bases[spec["key"]],
            pad=1.6,
        )
        ax.set_xlabel(X_LABEL)
        if index == 0:
            ax.set_ylabel(TOP_YLABEL)
        ax.set_xlim(80, 680)
        ax.set_xticks([120, 300, 480, 660])
        ax.set_ylim(-0.04, 1.0)
    add_panel_letters(axes, PANEL_LETTERS_TOP)


def draw_long_gap_axes(
    axes: Iterable[plt.Axes],
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    *,
    bottom_scope: str,
) -> None:
    axes = list(axes)
    expected_rows = 36 if bottom_scope == "clean36" else 38
    expected_specification = "clean36" if bottom_scope == "clean36" else "maximal38"
    lookup = model_lookup(models, expected_specification)
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        subset = frame.loc[frame["metric"].eq(spec["metric"])].copy()
        if len(subset) != expected_rows or subset["animal_id"].astype(str).nunique() != 14:
            raise AssertionError(
                f"{spec['metric']} is not {bottom_scope}/14: {len(subset)} rows, "
                f"{subset['animal_id'].astype(str).nunique()} mice"
            )
        model = model_values(lookup.loc[spec["metric"]], q_family="BH4")
        for experience, group in subset.groupby("experience", sort=True):
            experience = int(experience)
            color = palettes[spec["key"]][experience]
            ax.scatter(
                group["age_dst_days"],
                group["value"],
                s=12,
                marker="o",
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
                zorder=3,
            )
            x = group["age_dst_days"].to_numpy(dtype=float)
            y = group["value"].to_numpy(dtype=float)
            if len(x) >= 3 and np.std(x) > 0:
                line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
                line_y = float(np.mean(y)) + float(model["beta"]) * (
                    line_x - float(np.mean(x))
                ) / 30.0
                ax.plot(line_x, line_y, color=color, linewidth=1.05, zorder=2)
        ax.set_title(
            spec["label"],
            loc="left",
            fontweight="bold",
            color=bases[spec["key"]],
            pad=1.6,
        )
        ax.set_xlabel(X_LABEL)
        if index == 0:
            ax.set_ylabel(BOTTOM_YLABEL)
        ax.set_xlim(175, 680)
        ax.set_xticks([200, 350, 500, 650])
        ax.set_ylim(-0.08, 0.75)
    add_panel_letters(axes, PANEL_LETTERS_BOTTOM)


def add_stage_legend(
    fig: plt.Figure,
    palettes: dict[str, dict[int, str]],
    *,
    anchor_y: float,
) -> None:
    stage_handles = [
        tuple_handle([palettes[spec["key"]][stage] for spec in METRICS])
        for stage in range(5)
    ]
    fig.legend(
        stage_handles,
        [STAGE_LABELS[stage] for stage in range(5)],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        loc="upper center",
        bbox_to_anchor=(0.55, anchor_y),
        ncol=5,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.78,
    )


def add_transition_legend(
    fig: plt.Figure,
    palettes: dict[str, dict[int, str]],
    *,
    anchor_y: float,
) -> None:
    transition_handles = [
        tuple_handle([palettes[spec["key"]][experience] for spec in METRICS])
        for experience in range(1, 5)
    ]
    fig.legend(
        transition_handles,
        [TRANSITION_LABELS[experience] for experience in range(1, 5)],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        loc="upper center",
        bbox_to_anchor=(0.55, anchor_y),
        ncol=4,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.80,
    )


def render_first30_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=ROW_SIZE_IN, sharex=True, sharey=True)
    draw_first30_axes(axes, frame, models, palettes, bases)
    add_stage_legend(fig, palettes, anchor_y=0.995)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.205, top=0.72, wspace=0.23)
    return science.save_figure(fig, stem)


def render_long_gap_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
    *,
    bottom_scope: str,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=ROW_SIZE_IN, sharex=True, sharey=True)
    draw_long_gap_axes(
        axes,
        frame,
        models,
        palettes,
        bases,
        bottom_scope=bottom_scope,
    )
    add_transition_legend(fig, palettes, anchor_y=0.995)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.205, top=0.72, wspace=0.23)
    return science.save_figure(fig, stem)


def render_six_panel_assembly(
    first30: pd.DataFrame,
    first30_models: pd.DataFrame,
    long_gap: pd.DataFrame,
    long_models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
    *,
    bottom_scope: str,
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=ASSEMBLY_SIZE_IN,
        sharex="row",
        sharey="row",
    )
    draw_first30_axes(axes[0], first30, first30_models, palettes, bases)
    draw_long_gap_axes(
        axes[1],
        long_gap,
        long_models,
        palettes,
        bases,
        bottom_scope=bottom_scope,
    )
    add_stage_legend(fig, palettes, anchor_y=0.995)
    add_transition_legend(fig, palettes, anchor_y=0.505)
    fig.subplots_adjust(
        left=0.078,
        right=0.995,
        bottom=_COMPACT_BOTTOM,
        top=_COMPACT_TOP,
        wspace=0.23,
        hspace=_COMPACT_HSPACE,
    )
    return science.save_figure(fig, stem)


def compose_slide_mockup(figure_png: Path, output_path: Path) -> Path:
    dpi = 600
    canvas = Image.new(
        "RGB",
        (int(SLIDE_SIZE_IN[0] * dpi), int(SLIDE_SIZE_IN[1] * dpi)),
        "white",
    )
    image = Image.open(figure_png).convert("RGB")
    target_size = (
        int(round(ASSEMBLY_SIZE_IN[0] * dpi)),
        int(round(ASSEMBLY_SIZE_IN[1] * dpi)),
    )
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    y = int(round((SLIDE_SIZE_IN[1] - ASSEMBLY_SIZE_IN[1]) * dpi / 2.0))
    canvas.paste(image, (0, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))
    return output_path


def svg_colors(path: Path) -> set[str]:
    return {
        value.lower()
        for value in re.findall(r"#[0-9a-fA-F]{6}", path.read_text(encoding="utf-8"))
    }


def validate_svg(
    path: Path,
    required: set[str],
    forbidden: set[str],
) -> dict[str, Any]:
    observed = svg_colors(path)
    required = {value.lower() for value in required}
    missing = sorted(required - observed)
    forbidden_present = sorted({value.lower() for value in forbidden} & observed)
    text = path.read_text(encoding="utf-8", errors="replace")
    embedded_images = text.count("<image")
    if missing or forbidden_present or text.count("<text") == 0 or embedded_images:
        raise AssertionError(
            f"SVG gate failed for {path.name}: missing={missing}, "
            f"forbidden={forbidden_present}, embedded_images={embedded_images}"
        )
    return {
        "path": rel(path),
        "required_colors": sorted(required),
        "observed_colors": sorted(observed),
        "missing": missing,
        "forbidden_present": forbidden_present,
        "editable_text_nodes": text.count("<text"),
        "embedded_images": embedded_images,
    }


def image_check(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((256, 256)), dtype=float)
        if image.width < 2000 or image.height < 1000 or float(np.std(gray)) < 3.0:
            raise AssertionError(f"Raster output is blank or too small: {path}")
        return {
            **file_record(path, "rendered raster"),
            "width_px": image.width,
            "height_px": image.height,
            "grayscale_sd": float(np.std(gray)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional candidate-only output directory; scientific inputs still resolve from --root.",
    )
    parser.add_argument("--round", type=int, default=11)
    parser.add_argument(
        "--visual-qa-status", choices=("PENDING", "PASS"), default="PENDING"
    )
    parser.add_argument(
        "--sync-root",
        action="store_true",
        help="After all gates pass, replace the user-named root PNG/TIFF companions.",
    )
    parser.add_argument(
        "--bottom-scope",
        choices=("clean36",),
        default="clean36",
        help="Use the maintained clean36 primary; excluded recovered rows are not displayable.",
    )
    args = parser.parse_args()

    science.configure_style()
    root = args.root.resolve()
    audit_dir = root / "estimand_parity_audit"
    source_dir = root / "assembly" / "source_tables"
    output_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else root / "assembly" / f"display_round{args.round}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_long_root = (
        REPO_ROOT
        / "docs"
        / "extended_data_figure_panel_refresh_2026-07-12"
        / "source_tables"
    )
    if args.bottom_scope == "clean36":
        long_plot_path = clean_long_root / "ed09_long_gap_age_joined_corrected36.csv"
        long_model_path = clean_long_root / "ed09_long_gap_age_transition_fixed_models_corrected36.csv"
    else:
        long_plot_path = source_dir / "ed07_long_gap_plot_ready_maximal38_round1.csv"
        long_model_path = source_dir / "ed07_long_gap_models_maximal38_displayed_round1.csv"

    accepted_record_name = "main_figure3_2026-08-11_round2_timing_schematic_open_terms"
    accepted_record_root = next(
        (
            root
            for root in (
                REPO_ROOT / "docs" / "figure_version_records" / accepted_record_name,
                REPO_ROOT
                / "docs"
                / "figure_version_records_archive_2026-08-09"
                / accepted_record_name,
            )
            if root.is_dir()
        ),
        None,
    )
    if accepted_record_root is None:
        raise FileNotFoundError(accepted_record_name)
    inputs = {
        "first30_plot": audit_dir / "ED07_FIRST30_PLOT_READY_MAINTAINED58.csv",
        "first30_models": audit_dir / "ED07_FIRST30_AGE_MODELS_MAINTAINED58.csv",
        "maintained58_validation": audit_dir / "ED07_MAINTAINED58_AUTHORITY_VALIDATION.json",
        "accepted_first30_source": (
            accepted_record_root
            / "source_data"
            / "first30"
            / "combined_first30_points_all_metrics.csv"
        ),
        "long_gap_plot": long_plot_path,
        "long_gap_models": long_model_path,
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {label}: {path}")
    input_records = [file_record(path, label) for label, path in inputs.items()]
    round8_dir = root / "assembly" / "display_round8"
    round8_manifest_path = round8_dir / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json"
    round8_assembly_png = round8_dir / "ed07_age_controls_powerpoint_matched_round8.png"
    if not round8_manifest_path.is_file() or not round8_assembly_png.is_file():
        raise FileNotFoundError("Display round 8 is required as the revision authority")
    round8_manifest = json.loads(round8_manifest_path.read_text(encoding="utf-8"))
    # Immutable records may be archived under a new directory name.  Lock the
    # scientific payload by role, hash, and byte count rather than by its
    # administrative path; the current Main Fig. 3 record carries the same
    # accepted first-30-min table byte-for-byte.
    round8_input_signature = [
        (item["role"], item["sha256"], item["bytes"])
        for item in round8_manifest["inputs"]
    ]
    current_input_signature = [
        (item["role"], item["sha256"], item["bytes"]) for item in input_records
    ]
    if current_input_signature != round8_input_signature:
        raise AssertionError("Current scientific inputs drifted from display round 8")
    prior_dir = root / "assembly" / "display_round10"
    prior_manifest_path = prior_dir / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json"
    prior_assembly_png = prior_dir / "ed07_age_controls_six_panel_round10.png"
    if not prior_manifest_path.is_file() or not prior_assembly_png.is_file():
        raise FileNotFoundError("Display round 10 is required as the immediate layout authority")
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_input_signature = [
        (item["role"], item["sha256"], item["bytes"])
        for item in prior_manifest["inputs"]
    ]
    if current_input_signature != prior_input_signature:
        raise AssertionError("Round-11 scientific inputs drifted from display round 10")
    first30 = normalize_first30_plot(pd.read_csv(inputs["first30_plot"], dtype=str))
    maintained_validation = json.loads(
        inputs["maintained58_validation"].read_text(encoding="utf-8")
    )
    if maintained_validation.get("status") != "PASS":
        raise AssertionError("ED7 maintained58 authority gate is not PASS")
    first30 = first30.loc[first30["source_scope"].eq(MAINTAINED_FIRST30_SCOPE)].copy()
    if len(first30) != 174:
        raise AssertionError(
            f"Expected maintained58 x three metrics after exclusion, found {len(first30)}"
        )
    first30_models = pd.read_csv(inputs["first30_models"])
    accepted_first30 = pd.read_csv(
        inputs["accepted_first30_source"], dtype={"animal": str}
    )
    accepted_source = accepted_first30[
        ["animal", "experience_index", "metric", "source_bundles"]
    ].rename(columns={"animal": "animal_id"})
    accepted_source["experience_index"] = pd.to_numeric(
        accepted_source["experience_index"], errors="raise"
    )
    accepted_source["accepted_original_fallback"] = accepted_source[
        "source_bundles"
    ].astype(str).str.contains("original_fallback", regex=False)
    first30 = first30.merge(
        accepted_source.drop(columns="source_bundles"),
        on=["animal_id", "experience_index", "metric"],
        how="left",
        validate="one_to_one",
    )
    accepted_original = first30["accepted_original_fallback"].eq(True)
    first30["display_original_source"] = accepted_original
    if int(accepted_original.sum()) != 21:
        raise AssertionError("Expected seven age-eligible accepted original fallbacks x three metrics")
    if int(first30["display_original_source"].sum()) != 21:
        raise AssertionError("Expected seven accepted original-session fallbacks x three metrics")
    long_gap = pd.read_csv(inputs["long_gap_plot"], dtype={"animal_id": str})
    long_models = pd.read_csv(inputs["long_gap_models"])
    displayed_metrics = {spec["metric"] for spec in METRICS}
    long_gap = long_gap.loc[long_gap["metric"].isin(displayed_metrics)].copy()
    long_models = long_models.loc[long_models["metric"].isin(displayed_metrics)].copy()
    if "original_source_indicator" not in long_gap.columns:
        long_gap["original_source_indicator"] = 0
    palettes, bases, palette_authority = palette_display.load_palette_authorities()

    top_stem = output_dir / f"ed07_panels_abc_first30_to_day2_maintained58_round{args.round}"
    bottom_stem = output_dir / f"ed07_panels_def_long_gap_{args.bottom_scope}_round{args.round}"
    assembly_stem = output_dir / f"ed07_age_controls_six_panel_round{args.round}"
    top_outputs = render_first30_row(first30, first30_models, palettes, bases, top_stem)
    bottom_outputs = render_long_gap_row(
        long_gap,
        long_models,
        palettes,
        bases,
        bottom_stem,
        bottom_scope=args.bottom_scope,
    )
    assembly_outputs = render_six_panel_assembly(
        first30,
        first30_models,
        long_gap,
        long_models,
        palettes,
        bases,
        assembly_stem,
        bottom_scope=args.bottom_scope,
    )
    mockup = compose_slide_mockup(
        assembly_stem.with_suffix(".png"),
        output_dir / f"ed07_slide9_geometry_mockup_round{args.round}.png",
    )

    required_top = {
        palettes[spec["key"]][stage]
        for spec in METRICS
        for stage in range(5)
    } | set(bases.values())
    required_bottom = {
        palettes[spec["key"]][stage]
        for spec in METRICS
        for stage in range(1, 5)
    } | set(bases.values())
    top_svg_gate = validate_svg(
        top_stem.with_suffix(".svg"), required_top, palette_display.FORBIDDEN_DATA_COLORS
    )
    bottom_svg_gate = validate_svg(
        bottom_stem.with_suffix(".svg"), required_bottom, palette_display.FORBIDDEN_DATA_COLORS
    )
    assembly_svg_gate = validate_svg(
        assembly_stem.with_suffix(".svg"),
        required_top | required_bottom,
        palette_display.FORBIDDEN_DATA_COLORS,
    )

    top_text = top_stem.with_suffix(".svg").read_text(encoding="utf-8")
    bottom_text = bottom_stem.with_suffix(".svg").read_text(encoding="utf-8")
    assembly_text = assembly_stem.with_suffix(".svg").read_text(encoding="utf-8")
    forbidden_top_phrases = (
        "Same-stage Day 1-Day 2",
        "Same-stage Day 1–Day 2",
        "level map",
        "Level map",
        "Age on Day 1",
        "original-source",
        "sorted",
        "(r)",
    )
    if any(phrase in top_text for phrase in forbidden_top_phrases):
        raise AssertionError("Top row retains the withdrawn estimand or an undisplayed level role")
    for required_phrase in (
        TOP_YLABEL,
        X_LABEL,
        "Place map",
        "Population vector",
        "Ensemble rate",
    ):
        if required_phrase not in top_text:
            raise AssertionError(f"Top row is missing {required_phrase!r}")
    for forbidden_phrase in (
        "Destination-session age",
        "original-source",
        "sorted",
        "(r)",
    ):
        if forbidden_phrase in bottom_text:
            raise AssertionError(f"Bottom row retains forbidden label {forbidden_phrase!r}")
    for required_phrase in (BOTTOM_YLABEL, X_LABEL):
        if required_phrase not in bottom_text:
            raise AssertionError(f"Bottom row is missing {required_phrase!r}")
    for letter in (*PANEL_LETTERS_TOP, *PANEL_LETTERS_BOTTOM):
        if f">{letter}<" not in assembly_text:
            raise AssertionError(f"Six-panel assembly is missing panel letter {letter!r}")
    if "(r)" in assembly_text:
        raise AssertionError("Six-panel assembly retains the withdrawn '(r)' qualifier")
    forbidden_panel_statistics = ("age slope=", "; p=", "; q=", " mice")
    for svg_path in (top_stem.with_suffix(".svg"), bottom_stem.with_suffix(".svg")):
        svg_text = svg_path.read_text(encoding="utf-8")
        if any(token in svg_text for token in forbidden_panel_statistics):
            raise AssertionError(
                f"Caption-level statistics survived in displayed artwork: {svg_path.name}"
            )

    raster_checks = [
        image_check(top_stem.with_suffix(".png")),
        image_check(bottom_stem.with_suffix(".png")),
        image_check(assembly_stem.with_suffix(".png")),
        image_check(mockup),
    ]
    with Image.open(round8_assembly_png) as round8_image, Image.open(
        prior_assembly_png
    ) as prior_image, Image.open(assembly_stem.with_suffix(".png")) as current_image:
        if current_image.width != round8_image.width or current_image.height >= round8_image.height:
            raise AssertionError(
                "Current assembly must retain round-8 width and reduce vertical extent"
            )
        expected_height_reduction_px = int(
            round((PRIOR_ASSEMBLY_SIZE_IN[1] - ASSEMBLY_SIZE_IN[1]) * 600)
        )
        if (
            current_image.width != prior_image.width
            or prior_image.height - current_image.height != expected_height_reduction_px
        ):
            raise AssertionError(
                "Compact assembly did not preserve width and remove exactly the two legend-title slots"
            )
        round8_reference = {
            "manifest": file_record(round8_manifest_path, "round-8 display authority"),
            "assembly_png": file_record(round8_assembly_png, "round-8 assembly reference"),
            "scientific_input_identity": True,
            "round8_dimensions_px": [round8_image.width, round8_image.height],
            "current_dimensions_px": [current_image.width, current_image.height],
            "vertical_reduction_percent": round(
                100.0 * (1.0 - current_image.height / round8_image.height), 2
            ),
        }
        previous_display_reference = {
            "manifest": file_record(prior_manifest_path, "round-10 display authority"),
            "assembly_png": file_record(prior_assembly_png, "round-10 assembly reference"),
            "scientific_input_identity": True,
            "dimension_identity": False,
            "physical_axes_size_identity": True,
            "height_reduction_inches": round(
                PRIOR_ASSEMBLY_SIZE_IN[1] - ASSEMBLY_SIZE_IN[1], 4
            ),
            "requested_display_delta": (
                "use Day 1 early vs Day 2 correlation and Long gap correlation "
                "as the two row-level titles"
            ),
        }

    caption_path = output_dir / "ED07_SIX_PANEL_CAPTION_NOTES.md"
    caption_path.write_text(
        "# Extended Data Figure 7 caption notes\n\n"
        "**a-c,** Place-map, population-vector and ensemble-rate correlations "
        "between the first 30 min of maze time on Day 1 and the full Day 2 "
        "template, plotted against age. Colors denote experience; every "
        "observation is shown as a solid circle. Lines show the "
        "experience-adjusted age association. The primary cohort contains "
        "58 observations from 21 mice; BH correction spans the three "
        "displayed measures. Source provenance remains available in the "
        "plot-ready tables but is not visually encoded.\n\n"
        "**d-f,** Place map, population vector and ensemble rate long gap "
        "source-to-destination correlations plotted against age. Colors "
        "denote destination experience; every observation is shown as a "
        "solid circle. Lines show the transition-adjusted age association. "
        "The clean36 primary contains 36 session pairs from 14 mice; BH "
        "correction spans the four prespecified measures, including the "
        "undisplayed level-map measure.\n\n"
        "The six author-locked sessions remain excluded from all displayed "
        "values, models and multiplicity families.\n",
        encoding="utf-8",
    )

    contract_path = output_dir / "ED07_SIX_PANEL_FIGURE_CONTRACT.md"
    contract_path.write_text(
        "# Extended Data Figure 7 six-panel figure contract\n\n"
        "- Core conclusion: age covaries with first-window-to-Day-2 and long gap "
        "neural stability across three maintained representations.\n"
        "- Panels a-c: place-map, population-vector and ensemble-rate correlations "
        "between the first 30 min of maze time on Day 1 and the full Day 2 template "
        "(58 observations, 21 mice; BH3).\n"
        "- Panels d-f: the same three representations for the clean long gap "
        "source-to-destination analysis (36 pairs, 14 mice; BH4).\n"
        "- Display-only revision from round 10: the first-row y title is "
        "`Day 1 early vs Day 2 correlation` and the second-row title is "
        "`Long gap correlation`; the neutral `Age (days)` x labels, uniform "
        "solid-circle observations, six panel letters "
        "are retained. The redundant `Experience` and `Transition` legend titles "
        "are removed, and the assembly is compacted to 7.5 x 4.92 inches while "
        "preserving the six axes' physical dimensions.\n"
        "- Scientific lock: round-8 input paths and SHA-256 hashes, cohorts, "
        "exclusions, estimators, multiplicity families and semantic colors are unchanged.\n"
        "- Export contract: Python/matplotlib SVG and PDF with editable text and "
        "zero embedded raster images, plus 600-dpi PNG and LZW-TIFF companions.\n",
        encoding="utf-8",
    )

    root_png = root / "extended_data_figure7_source_rerender_round1.png"
    root_tiff = root / "extended_data_figure7_source_rerender_round1.tiff"
    root_caption = root / "ED07_CAPTION_NOTES.md"
    outputs = top_outputs + bottom_outputs + assembly_outputs + [
        mockup,
        caption_path,
        contract_path,
    ]
    root_identity: dict[str, Any] = {"synchronized": False}
    if args.sync_root:
        # Synchronize the exact companions named by the user only after every
        # scientific/source/color/output gate above has passed.
        shutil.copyfile(assembly_stem.with_suffix(".png"), root_png)
        shutil.copyfile(assembly_stem.with_suffix(".tiff"), root_tiff)
        shutil.copyfile(caption_path, root_caption)
        if sha256(root_png) != sha256(assembly_stem.with_suffix(".png")):
            raise AssertionError("Root PNG companion is not byte-identical to final assembly")
        if sha256(root_tiff) != sha256(assembly_stem.with_suffix(".tiff")):
            raise AssertionError("Root TIFF companion is not byte-identical to final assembly")
        outputs.extend([root_png, root_tiff, root_caption])
        root_identity = {
            "synchronized": True,
            "png": sha256(root_png),
            "tiff": sha256(root_tiff),
            "caption": sha256(root_caption),
        }
    manifest = {
        "schema": "ed07_slide9_six_panel_display/v2",
        "status": "PASS" if args.visual_qa_status == "PASS" else "PENDING_VISUAL_QA",
        "scientific_gate_status": maintained_validation["status"],
        "visual_qa_status": args.visual_qa_status,
        "panel_inventory": [
            "a/place_map",
            "b/population_vector",
            "c/ensemble_rate",
            "d/place_map",
            "e/population_vector",
            "f/ensemble_rate",
        ],
        "top_estimand": "Day 1 first 30 min of maze time versus full Day 2 template",
        "top_primary_scope": "58 maintained observations / 21 mice; BH3",
        "excluded_from_all_ed7_primary_analyses": list(EXCLUDED_FIRST30_SESSION_PAIRS),
        "exclusion_rationale": {
            "1636_1637_1639_325_347_6month": "different experiment",
            "579_6month": "sick animal",
        },
        "bottom_estimand": "long gap source-to-destination correlation with transition adjustment",
        "bottom_primary_scope": (
            "36 pairs / 14 mice; BH4; current clean single-vintage primary"
            if args.bottom_scope == "clean36"
            else "38 pairs / 14 mice; BH4; two-addition sensitivity"
        ),
        "representation_space": {
            "place": "15x15 grid-position tuning map",
            "pv": "grid-position population vector",
            "rate": "matched-neuron ensemble-rate vector",
        },
        "manual_mask_profile": "none for every displayed source row",
        "display_contract": {
            "derived_from": "display_round10",
            "row_size_inches": list(ROW_SIZE_IN),
            "assembly_size_inches": list(ASSEMBLY_SIZE_IN),
            "x_axis_label": X_LABEL,
            "top_y_axis_label": TOP_YLABEL,
            "bottom_y_axis_label": BOTTOM_YLABEL,
            "marker_policy": "uniform solid circle; source provenance is not visually encoded",
            "panel_letters": list(PANEL_LETTERS_TOP + PANEL_LETTERS_BOTTOM),
        },
        "caption_only_statistics": True,
        "previous_display_reference": previous_display_reference,
        "round8_reference": round8_reference,
        "inputs": input_records,
        "palette_authority": palette_authority,
        "top_svg_gate": top_svg_gate,
        "bottom_svg_gate": bottom_svg_gate,
        "assembly_svg_gate": assembly_svg_gate,
        "raster_checks": raster_checks,
        "outputs": [file_record(path, "display output") for path in outputs],
        "root_companion_identity": root_identity,
        "renderer": file_record(Path(__file__).resolve(), "display-only renderer"),
    }
    manifest_path = output_dir / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": rel(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
