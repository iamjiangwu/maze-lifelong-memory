"Draw Figure 2a from the included local-correlation image arrays."

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ca_analysis.figure_panels import (
    PANEL_LAYOUT,
    render_stacked_local_correlation_panel,
)

FONT_FAMILY = "Arial"

BASE_FONT_SIZE = 8.0

AXIS_LABEL_SIZE = 8.0

TICK_LABEL_SIZE = 7.0

TITLE_SIZE = 8.0


def _set_common_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
        }
    )


def _standardize_axes_text(fig, *, font_scale: float = 1.0) -> None:
    axis_label_size = AXIS_LABEL_SIZE * font_scale
    tick_label_size = TICK_LABEL_SIZE * font_scale
    title_size = TITLE_SIZE * font_scale
    for ax in fig.axes:
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontsize(axis_label_size)
        ax.yaxis.label.set_fontsize(axis_label_size)
        ax.tick_params(labelsize=tick_label_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(FONT_FAMILY)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(FONT_FAMILY)
                text.set_fontsize(tick_label_size)


def _render_panel_a(size: tuple[float, float]):
    _set_common_rc()
    layout = {
        **PANEL_LAYOUT,
        "panel_size": size,
        "grid_hspace": 0.05,
    }
    style = {
        "origin": "upper",
        "shared_clim": True,
        "shared_vmin": None,
        "shared_vmax": None,
        "show_axes": False,
        "show_colorbars": False,
        "show_panel_title": False,
        "show_subplot_titles": False,
        "side_label_transform": "axes",
        "side_label_rotation": 90,
        "side_label_ha": "center",
        "side_label_va": "center",
        "side_label_fontsize": 8.0,
        "side_label_fontweight": "normal",
        "top_side_label_xy": (-0.060, 0.50),
        "bottom_side_label_xy": (-0.060, 0.50),
        "show_scale_bar": True,
        "scale_bar_panel": "bottom",
        "scale_bar_xy": (295.0, 278.0),
        "scale_bar_length_px": 60.0,
        "scale_bar_linewidth": 2.4,
        "scale_bar_color": "white",
        "scale_bar_text_color": "white",
        "scale_bar_text_offset_px": -10.0,
        "scale_bar_fontsize": 6.4,
        "scale_bar_fontweight": "normal",
    }
    fig, _axes, _data, _paths = render_stacked_local_correlation_panel(
        top_fig_path=Path(__file__).resolve().parents[1]
        / "panel_a_sources/422_corr.npz",
        bottom_fig_path=Path(__file__).resolve().parents[1]
        / "panel_a_sources/422-3_corr.npz",
        panel_size=size,
        layout=layout,
        style=style,
        labels={
            "top_side_label": "Day 1",
            "bottom_side_label": "Day 91",
            "scale_bar_label": "100 \u00b5m",
        },
        save=False,
    )
    axes_flat = list(_axes.ravel()) if hasattr(_axes, "ravel") else list(_axes)
    if len(axes_flat) >= 2:
        axes_flat[0].set_position([0.055, 0.506, 0.940, 0.492])
        axes_flat[1].set_position([0.055, 0.006, 0.940, 0.492])
    _standardize_axes_text(fig)
    return fig
