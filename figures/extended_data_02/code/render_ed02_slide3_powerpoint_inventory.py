"""Render the nine PowerPoint-defined ED2 slide-3 panels from current data.

The supplied slide is the panel-inventory authority: a-i are retained in their
original scientific roles.  Panels a-d and f-i reuse the frozen current ED2
renderer.  Panel e preserves the displayed ``Shift from naive`` inset while
updating it from the same fixed-30-minute rows used by current panels b and c.

All panels are rendered at their intended physical PowerPoint size so text is
not made inconsistent by post-render image scaling.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_full_audit"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "tmp"
    / "ed02_slide3_inventory_correction_2026-08-08"
    / "panels"
)

PANEL_ORDER = tuple("abcdefghi")
FORMATS = ("svg", "pdf", "png", "tiff")

# Final physical sizes in the 7.5 x 7.5-inch source PowerPoint.  The renderer
# writes these exact sizes; the deck assembler inserts each asset 1:1.
PANEL_SIZE_IN = {
    "a": (1.10, 1.70),
    "b": (2.95, 1.60),
    "c": (2.95, 1.60),
    "d": (2.40, 1.55),
    "e": (1.15, 1.55),
    "f": (3.50, 1.65),
    "g": (2.10, 1.70),
    "h": (1.55, 1.70),
    "i": (3.50, 1.65),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_fixed_30min_slide",
    "b": "ed02b_naive_vs_return_policy_scatter_fixed_30min_slide",
    "c": "ed02c_return_vs_second_day_policy_scatter_fixed_30min_slide",
    "d": "ed02d_four_policy_contributions_fixed_30min_slide",
    "e": "ed02e_shift_from_naive_fixed_30min_slide",
    "f": "ed02f_four_policy_calibration_fixed_30min_slide",
    "g": "ed02g_depth_pbs_profile_fixed_30min_slide",
    "h": "ed02h_four_vs_depth_reconstruction_fixed_30min_slide",
    "i": "ed02i_depth_calibration_fixed_30min_slide",
}

PANEL_ROLE = {
    "a": "four branch-policy definitions",
    "b": "Naive versus Return policy probabilities",
    "c": "Return versus Second-day policy probabilities",
    "d": "one-policy-at-a-time contribution test",
    "e": "Return and Second-day shifts from the Naive policy means",
    "f": "four-policy walk calibration",
    "g": "depth-dependent P_BS profile and contrasts",
    "h": "four-policy versus depth-varying reconstruction",
    "i": "depth-varying walk calibration",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_renderer(record_root: Path) -> Any:
    path = record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ed02_frozen_full_story", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_slide_style(renderer: Any) -> None:
    renderer.configure_style()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    text = deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_TEXT)
    title_map = {
        "stem_to_branch": r"$P_{SA}$",
        "stem_forward": r"$P_{SF}$",
        "branch_to_stem": r"$P_{BS}$",
        "branch_forward": r"$P_{BF}$",
    }
    for row in text["rows"]:
        row["title"] = title_map[row["key"]]
        if row["key"] == "stem_to_branch":
            row["annotations"] = [
                {
                    "text": "preceding\nturn",
                    "xy": (2.92, 3.52),
                    "ha": "right",
                    "va": "center",
                }
            ]
    fig, _, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={"panel_size": PANEL_SIZE_IN["a"], "tight_layout": False},
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 0.9,
            "arrow_linewidth": 1.25,
            "previous_arrow_linewidth": 1.25,
            "arrow_mutation_scale": 7.5,
            "title_fontsize": 6.0,
            "title_fontweight": "normal",
            "or_fontsize": 5.6,
            "annotation_fontsize": 5.4,
        },
        text=text,
        save=False,
    )
    fig.set_size_inches(*PANEL_SIZE_IN["a"], forward=True)
    rows = pd.DataFrame(
        [
            {
                "window_minutes": 30,
                "helper": "render_grid_route_choice_schematic",
                "row_key": row["key"],
                "policy_label": row["title"],
            }
            for row in schematic["text"]["rows"]
        ]
    )
    return fig, {"schematic_rows": rows}


def build_panel_e(
    renderer: Any,
    natural_policy_rows: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    metrics = list(renderer.POLICY_ORDER)
    natural = natural_policy_rows.copy()
    natural[metrics] = natural[metrics].apply(pd.to_numeric, errors="coerce")

    means = {
        "naive": natural.loc[natural["group"].eq("naive"), metrics].mean(),
        "LTM": natural.loc[natural["group"].eq("LTM"), metrics].mean(),
        "experienced": natural.loc[
            natural["group"].eq("experienced"), metrics
        ].mean(),
    }
    shifts = pd.DataFrame(
        {
            "window_minutes": 30,
            "metric": metrics,
            "policy_label": [renderer.POLICY_LABEL[key] for key in metrics],
            "naive_mean": [means["naive"][key] for key in metrics],
            "return_mean": [means["LTM"][key] for key in metrics],
            "second_day_mean": [means["experienced"][key] for key in metrics],
            "return_minus_naive": [
                means["LTM"][key] - means["naive"][key] for key in metrics
            ],
            "second_day_minus_naive": [
                means["experienced"][key] - means["naive"][key]
                for key in metrics
            ],
            "n_naive_sessions": int(natural["group"].eq("naive").sum()),
            "n_return_sessions": int(natural["group"].eq("LTM").sum()),
            "n_second_day_sessions": int(
                natural["group"].eq("experienced").sum()
            ),
            "support": "natural primary-policy support",
            "point_weighting": "session-weighted group means",
        }
    )

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["e"])
    x = np.array([0.0, 1.0])
    return_color = renderer.GROUP_COLOR["LTM"]
    second_color = renderer.GROUP_COLOR["experienced"]
    label_offsets = {"Bf": -0.001, "Ba": 0.000, "Lf": -0.001, "Lo": 0.002}
    for row in shifts.itertuples(index=False):
        values = np.array([row.return_minus_naive, row.second_day_minus_naive])
        line_style = "--" if row.metric == "Bf" else "-"
        line_color = renderer.LIGHT if row.metric == "Bf" else renderer.REFERENCE
        ax.plot(x, values, color=line_color, linestyle=line_style, linewidth=0.9)
        ax.scatter(
            [0], [values[0]], s=18, color=return_color, edgecolors="white", linewidths=0.45, zorder=3
        )
        ax.scatter(
            [1], [values[1]], s=18, color=second_color, edgecolors="white", linewidths=0.45, zorder=3
        )
        ax.text(
            1.08,
            values[1] + label_offsets[row.metric],
            row.policy_label,
            ha="left",
            va="center",
            fontsize=5.2,
            color=line_color if row.metric == "Bf" else renderer.INK,
        )
    ax.axhline(0, color=renderer.REFERENCE, linestyle="--", linewidth=0.65)
    ax.set_xlim(-0.20, 1.48)
    ax.set_ylim(-0.025, 0.15)
    ax.set_xticks(x, ["Return", "Second\nday"])
    ax.set_ylabel("Mean - naive", labelpad=1.8)
    ax.set_title("Shift from naive", pad=2.0)
    ax.spines["bottom"].set_bounds(-0.02, 1.02)
    fig.subplots_adjust(left=0.36, right=0.80, bottom=0.18, top=0.87)
    return fig, {"shift_from_naive": shifts}


def normalize_figure_for_slide(fig: plt.Figure, panel: str) -> None:
    fig.set_size_inches(*PANEL_SIZE_IN[panel], forward=True)
    axes = fig.axes
    if panel in {"b", "c"}:
        footer = (
            "Dashed: equal-edge ref. | strict support | filled: session mean"
            if panel == "b"
            else "Dashed: equal-edge ref. | natural support | resemblance != equivalence"
        )
        for text in fig.texts:
            if "Equal-next-edge" in text.get_text():
                text.set_text(footer)
                text.set_fontsize(4.15)
                text.set_y(0.785)
        if fig.legends:
            fig.legends[0].set_bbox_to_anchor((0.5, 0.985))
        for ax in axes:
            ax.xaxis.label.set_fontsize(5.7)
            ax.xaxis.set_label_coords(0.5, -0.13)
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.735, wspace=0.32)
    elif panel == "d":
        axes[0].set_title("One-policy substitution", loc="left")
        axes[0].set_xlabel(
            "Return-own - one policy fixed to\nNaive mean (percentage points)",
            fontsize=5.2,
            labelpad=2.0,
        )
        fig.subplots_adjust(left=0.20, right=0.985, bottom=0.22, top=0.82)
    elif panel in {"f", "i"}:
        title = (
            "Four-policy calibration | gate: |bias| <= 0.05; rho >= 0.40"
            if panel == "f"
            else r"Depth-varying $P_{BS}$ calibration | gate: |bias| <= 0.05; rho >= 0.40"
        )
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text(title)
            fig._suptitle.set_fontsize(6.2)
            fig._suptitle.set_x(0.055)
            fig._suptitle.set_y(0.985)
        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.215, top=0.75, wspace=0.24)
    elif panel == "g":
        if fig.legends:
            legend = fig.legends[0]
            for text, label in zip(legend.get_texts(), ("Naive 23", "Return 58", "Day 2 74")):
                text.set_text(label)
                text.set_fontsize(4.25)
            if hasattr(legend, "set_ncols"):
                legend.set_ncols(3)
            legend.set_bbox_to_anchor((0.5, 0.985))
        for text in axes[0].texts:
            if "Lines = session means" in text.get_text():
                text.set_text("Lines: mean; bands: IQR")
                text.set_fontsize(4.3)
        lower_legend = axes[1].get_legend()
        if lower_legend is not None:
            for text, label in zip(
                lower_legend.get_texts(), ("Return - Naive", "Day 2 - Return")
            ):
                text.set_text(label)
                text.set_fontsize(4.15)
        for text in axes[1].texts:
            if "95% CI" in text.get_text():
                text.set_text("95% CI; BH3; root excluded")
                text.set_fontsize(4.2)
            elif text.get_text().startswith("q"):
                text.set_text(text.get_text().replace("0.", "."))
                text.set_fontsize(3.6)
                x, y = text.get_position()
                center = round(x)
                is_return_minus_naive = x > center
                text.set_rotation(0)
                text.set_x(center + (0.04 if is_return_minus_naive else -0.04))
                text.set_y(y + (0.025 if is_return_minus_naive else -0.015))
                text.set_ha("left" if is_return_minus_naive else "right")
        axes[1].set_xlim(-0.35, 2.40)
        axes[1].set_ylabel("Contrast\n(CR1)", fontsize=5.2, labelpad=0.5)
        fig.subplots_adjust(left=0.21, right=0.985, bottom=0.18, top=0.76, hspace=0.46)
    elif panel == "h":
        axes[0].set_title("Gap reconstructed", loc="left")
        axes[0].set_yticklabels(("Four policy", r"Depth $P_{BS}$"))
        axes[0].set_xlabel("Return - Naive efficiency gap", fontsize=5.4)
        axes[0].set_xlim(-0.005, 0.325)
        for index, text in enumerate(axes[0].texts):
            text.set_fontsize(4.65)
            if index == 1:
                text.set_ha("right")
                text.set_x(0.315)
        fig.subplots_adjust(left=0.32, right=0.985, bottom=0.20, top=0.80)
    fig.canvas.draw()


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": None, "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    record_root = record_root.resolve()
    output_root = output_root.resolve()
    renderer = load_frozen_renderer(record_root)
    configure_slide_style(renderer)
    frames, input_paths = renderer.load_sources(
        record_root / "analysis_inputs" / "strict",
        record_root / "analysis_inputs" / "natural",
    )

    b_points, b_contrasts = renderer.prepare_two_axis_scatter(
        frames["strict_policy_rows"],
        frames["strict_policy_contrasts"],
        ("naive", "LTM"),
        "return_vs_naive",
    )
    c_points, c_contrasts = renderer.prepare_two_axis_scatter(
        frames["natural_policy_rows"],
        frames["natural_policy_contrasts"],
        ("LTM", "experienced"),
        "second_vs_return",
    )

    builders: dict[str, Callable[[], tuple[plt.Figure, dict[str, pd.DataFrame]]]] = {
        "a": lambda: build_panel_a(renderer),
        "b": lambda: renderer.policy_scatter_panel(
            b_points,
            b_contrasts,
            ("naive", "LTM"),
            footer=(
                "Equal-next-edge references = dashed lines; strict common support; "
                "filled symbols = session-weighted means."
            ),
        ),
        "c": lambda: renderer.policy_scatter_panel(
            c_points,
            c_contrasts,
            ("LTM", "experienced"),
            footer=(
                "Equal-next-edge references = dashed lines; natural support; "
                "visual resemblance is not equality or equivalence."
            ),
        ),
        "d": lambda: renderer.panel_d(frames["strict_contributions"]),
        "e": lambda: build_panel_e(
            renderer, frames["natural_policy_rows"]
        ),
        "f": lambda: renderer.calibration_panel(
            frames["natural_simulation_rows"],
            frames["natural_calibration"],
            "four_bias",
        ),
        "g": lambda: renderer.panel_g(
            frames["natural_policy_rows"], frames["natural_depth_contrasts"]
        ),
        "h": lambda: renderer.panel_h(frames["strict_gap"]),
        "i": lambda: renderer.calibration_panel(
            frames["natural_simulation_rows"],
            frames["natural_calibration"],
            "depth_stratified",
        ),
    }

    panel_dir = output_root / "panels"
    table_dir = output_root / "plot_ready"
    panel_manifest: list[dict[str, Any]] = []
    table_manifest: list[dict[str, Any]] = []
    for panel in PANEL_ORDER:
        fig, tables = builders[panel]()
        normalize_figure_for_slide(fig, panel)
        paths = save_figure(fig, panel_dir / PANEL_STEM[panel], dpi)
        for path in paths:
            panel_manifest.append(
                {
                    "panel": panel,
                    "role": PANEL_ROLE[panel],
                    "format": path.suffix.lstrip("."),
                    "path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "width_in": PANEL_SIZE_IN[panel][0],
                    "height_in": PANEL_SIZE_IN[panel][1],
                }
            )
        for name, frame in tables.items():
            if panel == "g" and name == "group_profiles" and "window_minutes" not in frame:
                frame = frame.copy()
                frame.insert(0, "window_minutes", 30)
            path = write_table(frame, table_dir / f"panel_{panel}_{name}.csv")
            table_manifest.append(
                {
                    "panel": panel,
                    "table": name,
                    "path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "rows": len(frame),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    with (output_root / "PANEL_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(panel_manifest[0]))
        writer.writeheader()
        writer.writerows(panel_manifest)
    with (output_root / "TABLE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_manifest[0]))
        writer.writeheader()
        writer.writerows(table_manifest)

    source_manifest = {
        "panel_inventory_authority": "supplied PowerPoint slide 3; nine roles a-i",
        "source_record": str(record_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "frozen_renderer": {
            "path": str(
                (record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py").relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "sha256": sha256(
                record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
            ),
        },
        "inputs": [
            {
                "key": key,
                "path": str(path.relative_to(record_root)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for key, path in sorted(input_paths.items())
        ],
        "window_minutes": 30,
        "clock": "first 30 cumulative TimeInMaze minutes; all modes advance the clock",
        "policy_events": "Explore-mode junction decisions only",
        "manual_mask_applicability": "not applicable; behavior-only analysis",
        "panel_e_update": (
            "same displayed Shift-from-naive role as the supplied slide, recomputed from "
            "current fixed-30-minute natural-support session-weighted group means"
        ),
    }
    (output_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "status": "pass",
        "panel_count": len(PANEL_ORDER),
        "panel_order": list(PANEL_ORDER),
        "panel_artifacts": len(panel_manifest),
        "plot_ready_tables": len(table_manifest),
        "dpi": dpi,
        "output_root": str(output_root),
    }
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
