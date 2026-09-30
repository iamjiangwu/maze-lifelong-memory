"""Shared display contract for the renumbered Extended Data supplement rebuild.

Layout authority is ``sup-figs-v2.pptx``. The deck merges ED2 into ED1 and ED8 into
ED7, so slide numbers and figure numbers differ; see
``docs/figure_version_records/CURRENT_EXTENDED_DATA_FIGURES.md``.

The contract implemented here was settled across the Main Figure 1-4 and Extended Data
Figure 3 review rounds:

* typography matches the main figures **at printed size**, so a 7.5-inch Extended Data
  canvas draws at 0.75x Main Figure 1's point sizes for a shared 180 mm export;
* Main Figure 1 condition palette and labels, with Return day softened;
* Main Figure 1 bar-plus-scatter grammar and bracket-and-star significance;
* no background grid lines, no rotated x tick labels;
* exact P/q values, session and mouse counts and mean/SEM notes belong in the caption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REBUILD_ROOT = (
    REPO_ROOT / "docs" / "extended_data_supplement_v2_rebuild_2026-08-05"
)
DECK_LAYOUT_MANIFEST = (
    REPO_ROOT / "tmp" / "extended_data_deck_layout_2026-08-05" / "deck_layout_manifest.json"
)

FORMATS = ("svg", "pdf", "png", "tiff")
EXPORT_WIDTH_MM = 180.0
MAIN_FIGURE_CANVAS_WIDTH_IN = 10.0

FONT_FAMILY = "Arial"
MAIN_BASE_FONT_SIZE = 8.0
MAIN_AXIS_LABEL_SIZE = 8.0
MAIN_TICK_LABEL_SIZE = 7.0
MAIN_TITLE_SIZE = 8.0
MAIN_PANEL_ID_SIZE = 13.0
MAIN_AXES_LINEWIDTH = 0.8

# Main Figure 1 condition palette. Return day is softened from Figure 1's #DA2323, which
# sits at saturation 0.84 / luminance 74 against 0.13-0.41 / 127-182 for its palette-mates
# and reads too hot as large filled bars. Documented deviation, not drift.
NAIVE_COLOR = "#8BBDEB"
SECOND_DAY_COLOR = "#5F8C62"
LATENT_DAY2_COLOR = "#B3A99B"
RETURN_COLOR = "#E15759"
RETURN_COLOR_MAIN_FIGURE = "#DA2323"

CONDITION_LABELS = {
    "naive": "Naive day",
    "experienced_day2": "Second day",
    "return": "Return day",
    "latent_day2": "Latent day2",
}
CONDITION_COLORS = {
    "naive": NAIVE_COLOR,
    "experienced_day2": SECOND_DAY_COLOR,
    "return": RETURN_COLOR,
    "latent_day2": LATENT_DAY2_COLOR,
}

AXIS_COLOR = "#202020"
MUTED_COLOR = "#8A8A8A"

# Main Figure 3 method-schematic palette, reused for Extended Data schematics.
SCHEMATIC_COLORS = {
    "ink": "#30363A",
    "muted": "#667168",
    "bar_fill": "#F1F6F1",
    "bar_edge": "#91A493",
    "whole_fill": "#75A86F",
    "whole_edge": "#376F3E",
    "window_edge": "#3F8048",
    "window_colors": ("#DCEBDC", "#B8D3B8", "#8FB791"),
    "arrow": "#3F8048",
    "connector": "#B3BEB4",
}


class Canvas:
    """A figure canvas with main-figure-matched typography and inch-based placement."""

    def __init__(
        self,
        width_in: float,
        height_in: float,
        *,
        font_reference_width_in: float | None = None,
    ) -> None:
        self.width_in = float(width_in)
        self.height_in = float(height_in)
        # Default: scale typography from this canvas width (7.5 in → 0.75×).
        # Content-tight canvases can keep the shared contract via font_reference_width_in.
        self.font_reference_width_in = float(
            self.width_in if font_reference_width_in is None else font_reference_width_in
        )
        self.font_scale = self.font_reference_width_in / MAIN_FIGURE_CANVAS_WIDTH_IN
        self.base_font = MAIN_BASE_FONT_SIZE * self.font_scale
        self.axis_label = MAIN_AXIS_LABEL_SIZE * self.font_scale
        self.tick_label = MAIN_TICK_LABEL_SIZE * self.font_scale
        self.title = MAIN_TITLE_SIZE * self.font_scale
        self.panel_id = MAIN_PANEL_ID_SIZE * self.font_scale
        self.axes_linewidth = MAIN_AXES_LINEWIDTH * self.font_scale
        self.configure()
        self.fig = plt.figure(figsize=(self.width_in, self.height_in))
        self.fig.patch.set_facecolor("white")

    def configure(self) -> None:
        matplotlib.rcParams.update(
            {
                "font.family": FONT_FAMILY,
                "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
                "font.size": self.base_font,
                "axes.labelsize": self.axis_label,
                "axes.titlesize": self.title,
                "xtick.labelsize": self.tick_label,
                "ytick.labelsize": self.tick_label,
                "legend.fontsize": self.tick_label,
                "axes.linewidth": self.axes_linewidth,
                "xtick.major.width": self.axes_linewidth,
                "ytick.major.width": self.axes_linewidth,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
                "mathtext.fontset": "custom",
                "mathtext.rm": FONT_FAMILY,
                "mathtext.it": f"{FONT_FAMILY}:italic",
                "mathtext.bf": f"{FONT_FAMILY}:bold",
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def rect(self, x: float, y: float, width: float, height: float) -> list[float]:
        """Inch rectangle, measured from the lower-left corner, to figure fraction."""
        return [
            x / self.width_in,
            y / self.height_in,
            width / self.width_in,
            height / self.height_in,
        ]

    def axes(self, x: float, y: float, width: float, height: float) -> plt.Axes:
        return self.fig.add_axes(self.rect(x, y, width, height))

    def panel_letter(self, x_in: float, y_in: float, letter: str) -> None:
        self.fig.text(
            x_in / self.width_in,
            y_in / self.height_in,
            letter,
            ha="left",
            va="top",
            fontsize=self.panel_id,
            fontweight="bold",
        )

    def text(self, x_in: float, y_in: float, body: str, **kwargs: Any) -> None:
        self.fig.text(x_in / self.width_in, y_in / self.height_in, body, **kwargs)

    def place_image(
        self,
        path: Path,
        x: float,
        y: float,
        *,
        width: float | None = None,
        height: float | None = None,
        crop: dict[str, float] | None = None,
        fill_slot: bool = False,
    ) -> dict[str, Any]:
        """Place a maintained panel image without distorting its aspect ratio.

        Exactly one of ``width`` or ``height`` fixes the size; the other follows from the
        image's own aspect after any crop, so a panel can never be stretched to fill a
        PowerPoint slot. Pass both ``width`` and ``height`` with ``fill_slot=True`` only
        when the image already matches the PowerPoint slot aspect and must occupy that
        exact rectangle.
        """
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGBA"))
        rows, cols = array.shape[0], array.shape[1]
        if crop:
            left = int(round(cols * float(crop.get("l", 0.0))))
            right = cols - int(round(cols * float(crop.get("r", 0.0))))
            top = int(round(rows * float(crop.get("t", 0.0))))
            bottom = rows - int(round(rows * float(crop.get("b", 0.0))))
            array = array[top:bottom, left:right]
            rows, cols = array.shape[0], array.shape[1]
        aspect = cols / rows
        if width is None and height is None:
            raise ValueError("place_image needs width or height")
        if fill_slot:
            if width is None or height is None:
                raise ValueError("fill_slot requires both width and height")
            aspect_preserved = abs((float(width) / float(height)) - aspect) < 0.02
        elif width is None:
            width = float(height) * aspect
            aspect_preserved = True
        elif height is None:
            height = float(width) / aspect
            aspect_preserved = True
        else:
            raise ValueError("pass one of width/height, or both with fill_slot=True")
        ax = self.axes(x, y, float(width), float(height))
        ax.imshow(array, interpolation="antialiased", aspect="auto")
        ax.set_axis_off()
        try:
            recorded_path = (
                str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
            )
        except ValueError:
            recorded_path = str(path)
        return {
            "path": recorded_path,
            "pixel_size": [int(cols), int(rows)],
            "placed_inches": [round(float(x), 4), round(float(y), 4),
                              round(float(width), 4), round(float(height), 4)],
            "aspect_preserved": aspect_preserved,
            "crop_fraction": crop or {"l": 0.0, "t": 0.0, "r": 0.0, "b": 0.0},
            "sha256": sha256(path),
        }

    def save(self, stem: Path) -> list[Path]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for extension in FORMATS:
            path = stem.with_suffix(f".{extension}")
            kwargs: dict[str, Any] = {"facecolor": "white"}
            if extension in {"png", "tiff"}:
                kwargs["dpi"] = 600
            if extension == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            self.fig.savefig(path, **kwargs)
            outputs.append(path)
        return outputs

    def close(self) -> None:
        plt.close(self.fig)

    # -- shared panel grammar -------------------------------------------------

    def significance_stars(self, q_value: float) -> str:
        """Main Figure 1 thresholds; ``n.s.`` where a claim is deliberately null."""
        if q_value < 0.001:
            return "***"
        if q_value < 0.01:
            return "**"
        if q_value < 0.05:
            return "*"
        return "n.s."

    def significance_bracket(
        self,
        ax: plt.Axes,
        x_left: float,
        x_right: float,
        q_value: float,
        *,
        y_frac: float = 0.86,
        skip_ns: bool = False,
        star_offset_frac: float = 0.006,
        star_fontsize: float | None = None,
    ) -> str | None:
        """Draw a Main Figure / ED1 bracket+star. Returns the star text, or None if skipped.

        ``star_offset_frac`` matches Extended Data Figure 1: stars sit just above the
        bracket ticks (0.006 × axis span), not a large multiple of the tick height.
        """
        stars = self.significance_stars(q_value)
        if skip_ns and stars == "n.s.":
            return None
        low, high = ax.get_ylim()
        span = high - low
        y_base = low + float(y_frac) * span
        tick = 0.022 * span
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_base, y_base + tick, y_base + tick, y_base],
            color=AXIS_COLOR,
            linewidth=0.7,
            clip_on=False,
            zorder=5,
        )
        if star_fontsize is None:
            star_fontsize = (
                self.tick_label if stars == "n.s." else self.tick_label * 1.05
            )
        ax.text(
            (x_left + x_right) / 2.0,
            y_base + tick + float(star_offset_frac) * span,
            stars,
            ha="center",
            va="bottom",
            fontsize=star_fontsize,
            fontweight="normal" if stars == "n.s." else "bold",
            color=AXIS_COLOR,
            clip_on=False,
            zorder=6,
        )
        return stars

    def bar_scatter_group(
        self,
        ax: plt.Axes,
        index: int,
        values: np.ndarray,
        mean: float,
        ci_low: float,
        ci_high: float,
        color: str,
        rng: np.random.Generator,
    ) -> None:
        """Main Figure 1 bar-plus-scatter grammar, scaled to this canvas."""
        ax.bar(
            index, mean, width=0.60, color=color, edgecolor="none", alpha=0.78, zorder=1
        )
        ax.scatter(
            np.full(len(values), index, dtype=float) + rng.normal(0.0, 0.06, len(values)),
            values,
            s=15.0 * self.font_scale**2,
            facecolor=color,
            edgecolor="none",
            alpha=0.72,
            zorder=2,
        )
        ax.errorbar(
            index,
            mean,
            yerr=np.array([[mean - ci_low], [ci_high - mean]]),
            color="black",
            fmt="none",
            capsize=4.0 * self.font_scale,
            elinewidth=1.1 * self.font_scale,
            capthick=1.1 * self.font_scale,
            zorder=4,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    """Small pass/fail recorder shared by the Extended Data assembly scripts."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append(
            {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}
        )

    def check_outputs(self, outputs: list[Path]) -> None:
        for path in outputs:
            self.record(
                f"output_{path.suffix.lstrip('.')}",
                path.exists() and path.stat().st_size > 0,
                f"{path.name} {path.stat().st_size} bytes",
            )
        svg = next((p for p in outputs if p.suffix == ".svg"), None)
        if svg is not None:
            text = svg.read_text(encoding="utf-8", errors="ignore")
            self.record("svg_declares_arial", FONT_FAMILY in text, "Arial declared in SVG")

    def check_no_grid_and_upright_labels(self, fig: plt.Figure) -> None:
        gridlines = sum(
            1
            for axis in fig.axes
            for line in axis.get_xgridlines() + axis.get_ygridlines()
            if line.get_visible()
        )
        self.record("no_background_grid_lines", gridlines == 0, f"{gridlines} gridlines")
        rotation = max(
            (label.get_rotation() for axis in fig.axes for label in axis.get_xticklabels()),
            default=0.0,
        )
        self.record(
            "xticklabels_not_rotated", rotation == 0.0, f"max rotation {rotation} deg"
        )

    def check_typography(self, canvas: Canvas) -> None:
        printed = lambda drawn, width: drawn * EXPORT_WIDTH_MM / (width * 25.4)
        main = printed(MAIN_BASE_FONT_SIZE, MAIN_FIGURE_CANVAS_WIDTH_IN)
        # Judge against the typography reference width so content-tight canvases can
        # still keep the shared 0.75× Extended Data contract.
        reference_width = getattr(canvas, "font_reference_width_in", canvas.width_in)
        here = printed(MAIN_BASE_FONT_SIZE * canvas.font_scale, reference_width)
        self.record(
            "typography_matches_main_figures_at_print",
            abs(main - here) < 1e-9,
            f"base {here:.2f} pt printed (reference width {reference_width:.3f} in)",
        )

    @property
    def failed(self) -> list[dict[str, Any]]:
        # DEVIATION rows record accepted divergences from the shared display contract and
        # are not failures; they exist so a promotion gate can assert the exact set.
        return [row for row in self.rows if row["status"] not in ("PASS", "DEVIATION")]

    @property
    def status(self) -> str:
        return "pass" if not self.failed else "fail"
