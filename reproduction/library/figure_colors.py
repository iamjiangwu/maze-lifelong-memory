"""Shared semantic color conventions for main and Extended Data figures."""

from __future__ import annotations

from copy import deepcopy


REPRESENTATION_COLORS = {
    "place_map": "#B95A58",
    "pv": "#e29957",
    "ensemble_rate": "#4a5e65",
    "level_map": "#7976a2",
}

REPRESENTATION_COLOR_ALIASES = {
    "place": "place_map",
    "placemap": "place_map",
    "place_map_correlation": "place_map",
    "pv_correlation": "pv",
    "rate": "ensemble_rate",
    "firing_rate": "ensemble_rate",
    "ensemble_rate_correlation": "ensemble_rate",
    "level": "level_map",
    "level_map_correlation": "level_map",
}

CORRELATION_MEASURE_COLORS = {
    "place_map_correlation": REPRESENTATION_COLORS["place_map"],
    "pv_correlation": REPRESENTATION_COLORS["pv"],
    "ensemble_rate_correlation": REPRESENTATION_COLORS["ensemble_rate"],
    "level_map_correlation": REPRESENTATION_COLORS["level_map"],
}

CORRELATION_EXPERIENCE_COLORS = {
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


def get_representation_colors(
    overrides: dict | None = None, *, include_aliases: bool = True
) -> dict[str, str]:
    colors = deepcopy(REPRESENTATION_COLORS)
    if overrides:
        for key, value in overrides.items():
            canonical_key = REPRESENTATION_COLOR_ALIASES.get(key, key)
            colors[canonical_key] = value
    if include_aliases:
        for alias, canonical_key in REPRESENTATION_COLOR_ALIASES.items():
            if canonical_key in colors:
                colors[alias] = colors[canonical_key]
    return colors


def get_correlation_measure_colors(overrides: dict | None = None) -> dict[str, str]:
    colors = deepcopy(CORRELATION_MEASURE_COLORS)
    if overrides:
        representation_colors = get_representation_colors(
            overrides, include_aliases=False
        )
        metric_by_representation = {
            "place_map": "place_map_correlation",
            "pv": "pv_correlation",
            "ensemble_rate": "ensemble_rate_correlation",
            "level_map": "level_map_correlation",
        }
        for representation_key, metric_key in metric_by_representation.items():
            if representation_key in representation_colors:
                colors[metric_key] = representation_colors[representation_key]
        for key, value in overrides.items():
            if key not in REPRESENTATION_COLOR_ALIASES and key not in REPRESENTATION_COLORS:
                colors[key] = value
    return colors


def get_correlation_experience_colors(
    overrides: dict | None = None,
) -> dict[str, dict[str, str]]:
    colors = deepcopy(CORRELATION_EXPERIENCE_COLORS)
    if overrides:
        metric_by_representation = {
            "place_map": "place_map_correlation",
            "pv": "pv_correlation",
            "ensemble_rate": "ensemble_rate_correlation",
            "level_map": "level_map_correlation",
        }
        for metric, group_colors in overrides.items():
            canonical_metric = REPRESENTATION_COLOR_ALIASES.get(metric, metric)
            canonical_metric = metric_by_representation.get(
                canonical_metric, canonical_metric
            )
            if isinstance(group_colors, dict) and canonical_metric in colors:
                colors[canonical_metric].update(group_colors)
            else:
                colors[canonical_metric] = deepcopy(group_colors)
    return colors


__all__ = [
    "CORRELATION_EXPERIENCE_COLORS",
    "CORRELATION_MEASURE_COLORS",
    "REPRESENTATION_COLOR_ALIASES",
    "REPRESENTATION_COLORS",
    "get_correlation_experience_colors",
    "get_correlation_measure_colors",
    "get_representation_colors",
]
