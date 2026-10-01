"""Shared semantic colors for behavioral Main and Extended Data figures."""

from __future__ import annotations

from copy import deepcopy


# Strong colors are for markers and lines; fills are for low-alpha
# distributions and bands.  Aliases keep historical group names out of
# individual renderers without mixing these colors with neural code colors.
BEHAVIOR_GROUP_COLORS = {
    "naive": "#1F90D5",
    "return": "#D82929",
    "second_day": "#5F8C62",
    "latent_day2": "#B3A99B",
}

BEHAVIOR_GROUP_FILL_COLORS = {
    "naive": "#8BBDEB",
    "return": "#DA2323",
    "second_day": "#5F8C62",
    "latent_day2": "#B3A99B",
}

BEHAVIOR_GROUP_COLOR_ALIASES = {
    "LTM": "return",
    "ltm": "return",
    "experienced": "second_day",
    "experienced_day2": "second_day",
    "second": "second_day",
    "latent": "latent_day2",
}

# Established ED3 turning-policy identities.  These remain distinct from
# condition colors and from neural-representation colors.
TURNING_POLICY_COLORS = {
    "Bf": "#7F7F7F",  # P_SF
    "Ba": "#7E5FA8",  # P_SA
    "Lf": "#1F6FB2",  # P_BF
    "Lo": "#2CA25F",  # P_BS
}

TURNING_MODEL_COLORS = {
    "four_bias": "#69737F",
    "depth_stratified": "#D26A3A",
}

DIAGNOSTIC_COLORS = {
    "pass": "#287C70",
    "fail": "#A5ADB4",
}

FIGURE_NEUTRAL_COLORS = {
    "background": "#FFFFFF",
    "ink": "#27313A",
    "muted": "#68737D",
    "reference": "#6F6F6F",
    "light": "#D7DDE2",
    "pale": "#EEF1F3",
    "row_band": "#F6F6F4",
}


def get_behavior_group_colors(
    overrides: dict | None = None,
    *,
    include_aliases: bool = True,
    fills: bool = False,
) -> dict[str, str]:
    colors = deepcopy(
        BEHAVIOR_GROUP_FILL_COLORS if fills else BEHAVIOR_GROUP_COLORS
    )
    if overrides:
        for key, value in overrides.items():
            canonical_key = BEHAVIOR_GROUP_COLOR_ALIASES.get(key, key)
            colors[canonical_key] = value
    if include_aliases:
        for alias, canonical_key in BEHAVIOR_GROUP_COLOR_ALIASES.items():
            if canonical_key in colors:
                colors[alias] = colors[canonical_key]
    return colors


__all__ = [
    "BEHAVIOR_GROUP_COLOR_ALIASES",
    "BEHAVIOR_GROUP_COLORS",
    "BEHAVIOR_GROUP_FILL_COLORS",
    "DIAGNOSTIC_COLORS",
    "FIGURE_NEUTRAL_COLORS",
    "TURNING_MODEL_COLORS",
    "TURNING_POLICY_COLORS",
    "get_behavior_group_colors",
]
