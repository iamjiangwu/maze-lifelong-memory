"""Recompute the manuscript's three captioned maxT families with BH.

The raw exact tests, sidedness, estimands, and statistical units are unchanged.
ED5 follows the established within-metric local-family policy (three BH3
families); ED8 and ED10 retain their four- and three-test family boundaries.
maxT-adjusted values are retained only as the historical comparison.
"""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs/manuscript_figure_captions_2026-08-10"
OUT_CSV = OUT_DIR / "MAXT_TO_BH_RECOMPUTATION_AUDIT_2026-08-16.csv"
OUT_REPORT = OUT_DIR / "MAXT_TO_BH_RECOMPUTATION_REPORT_2026-08-16.md"


def read_rows(relative_path: str) -> list[dict[str, str]]:
    with (REPO / relative_path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def bh_adjust(raw_p: list[float]) -> list[float]:
    """Standard step-up BH adjusted p values in input order."""

    order = sorted(range(len(raw_p)), key=raw_p.__getitem__)
    adjusted = [0.0] * len(raw_p)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(raw_p) - reverse_rank + 1
        running = min(running, raw_p[index] * len(raw_p) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


audit_rows: list[dict[str, object]] = []


def add_family(
    *,
    family_id: str,
    figure_panel: str,
    source_file: str,
    rows: list[dict[str, str]],
    raw_column: str,
    old_column: str,
    test_id,
    alternative: str,
    statistical_unit: str,
) -> None:
    raw = [float(row[raw_column]) for row in rows]
    adjusted = bh_adjust(raw)
    for row, raw_p, bh_p in zip(rows, raw, adjusted):
        old_p = float(row[old_column])
        audit_rows.append(
            {
                "figure_panel": figure_panel,
                "family_id": family_id,
                "family_size": len(rows),
                "test_id": test_id(row),
                "alternative": alternative,
                "statistical_unit": statistical_unit,
                "raw_exact_p": f"{raw_p:.17g}",
                "historical_maxT_adjusted_p": f"{old_p:.17g}",
                "bh_adjusted_p": f"{bh_p:.17g}",
                "historical_star": stars(old_p),
                "bh_star": stars(bh_p),
                "star_changed": stars(old_p) != stars(bh_p),
                "source_file": source_file,
                "raw_column": raw_column,
            }
        )


ed5_source = (
    "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
    "ed06_day_combination_paired_stats_corrected36.csv"
)
ed5_rows = [
    row
    for row in read_rows(ed5_source)
    if row["family"] == "d1v1_pairwise" and row["test"] == "paired t-test"
]
ed5_panels = {
    "place_map_correlation": "Extended Data Figure 5b",
    "pv_correlation": "Extended Data Figure 5c",
    "ensemble_rate_correlation": "Extended Data Figure 5d",
}
for ed5_metric, ed5_panel in ed5_panels.items():
    add_family(
        family_id=f"ed5_{ed5_metric}_bh3",
        figure_panel=ed5_panel,
        source_file=ed5_source,
        rows=[row for row in ed5_rows if row["metric"] == ed5_metric],
        raw_column="p_exact_signflip",
        old_column="p_maxT9_exact_signflip",
        test_id=lambda row: f"{row['metric']}|{row['contrast']}",
        alternative="two-sided",
        statistical_unit="animal",
    )

ed8_source = (
    "docs/ed08_panel_b_all_eligible_2026-08-16/"
    "ED08_PANEL_B_ALL_ELIGIBLE_STATISTICS.csv"
)
ed8_rows = read_rows(ed8_source)
add_family(
    family_id="ed8_b_decoder_bh4",
    figure_panel="Extended Data Figure 8b",
    source_file=ed8_source,
    rows=ed8_rows,
    raw_column="p_exact_mouse_signflip",
    old_column="p_exact_maxT4",
    test_id=lambda row: f"{row['variant']}|{row['resample_method']}",
    alternative="one-sided",
    statistical_unit="animal",
)

ed10_source = (
    "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
    "extended_data_fig_11/fixed_neuron_correlation_synthesis_2026-08-09/"
    "REWARD_LEVEL_PROFILE_TESTS.csv"
)
ed10_rows = [
    row
    for row in read_rows(ed10_source)
    if row["specification"] == "pooled"
    and row["weighting"] == "pair_weighted"
    and row["role"] == "stacked_marginal_profile_test"
]
add_family(
    family_id="ed10_b_profile_bh3",
    figure_panel="Extended Data Figure 10b",
    source_file=ed10_source,
    rows=ed10_rows,
    raw_column="exact_p_two_sided",
    old_column="maxT_adjusted_p",
    test_id=lambda row: row["claim"],
    alternative="two-sided",
    statistical_unit="animal-cluster score contribution",
)

if len(audit_rows) != 16:
    raise AssertionError(f"Expected 16 captioned tests, found {len(audit_rows)}")

with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
    writer.writeheader()
    writer.writerows(audit_rows)

report = [
    "# maxT-to-Benjamini--Hochberg recomputation audit (2026-08-16)",
    "",
    "The three historical captioned maxT families were recomputed from their raw exact-test p values. ED5 uses the established within-metric local-family policy (three BH3 families); ED8 and ED10 retain their four- and three-test family boundaries. Sidedness, estimands, and biological statistical units were unchanged.",
    "",
    "| Panel | Test | Raw exact p | Historical maxT p | BH-adjusted p | Stars |",
    "|---|---|---:|---:|---:|---:|",
]
for row in audit_rows:
    report.append(
        f"| {row['figure_panel']} | `{row['test_id']}` | {row['raw_exact_p']} | "
        f"{row['historical_maxT_adjusted_p']} | {row['bh_adjusted_p']} | "
        f"{row['historical_star']} -> {row['bh_star']} |"
    )
report.extend(
    [
        "",
        "Star thresholds are `*** < 0.001`, `** < 0.01`, `* < 0.05`, otherwise `n.s.`.",
        "The complete machine-readable table is `MAXT_TO_BH_RECOMPUTATION_AUDIT_2026-08-16.csv`.",
        "",
    ]
)
OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print(f"Wrote {OUT_CSV.relative_to(REPO)} ({len(audit_rows)} tests)")
print(f"Wrote {OUT_REPORT.relative_to(REPO)}")
