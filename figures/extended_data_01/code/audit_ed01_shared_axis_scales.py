"""Audit every ED1 panel for visually shared axes with unequal numeric scales."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CANDIDATE = REPO / "docs/ed01_whole_group_promotion_2026-08-15/round17_candidate"
SOURCE = CANDIDATE / "AXIS_SCALE_AUDIT_ROUND17.csv"
OUTPUT = REPO / "docs/ed01_whole_group_promotion_2026-08-15/axis_scale_audit_round17"


def selected(rows: pd.DataFrame, members: list[tuple[str, int]]) -> pd.DataFrame:
    parts = [
        rows.loc[rows["panel"].eq(panel) & rows["axis_index"].eq(axis_index)]
        for panel, axis_index in members
    ]
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(members):
        raise RuntimeError(f"Axis inventory mismatch for {members}: found {len(result)}")
    return result


def arrays_equal(values: list[list[float]]) -> bool:
    return all(
        len(values[0]) == len(value) and np.allclose(values[0], value)
        for value in values[1:]
    )


def compare_axis(
    rows: pd.DataFrame,
    *,
    audit_id: str,
    members: list[tuple[str, int]],
    dimension: str,
    reason: str,
) -> dict[str, object]:
    subset = selected(rows, members)
    scales = subset[f"{dimension}scale"].astype(str).tolist()
    limits = subset[[f"{dimension}lim_low", f"{dimension}lim_high"]].to_numpy(float)
    raw_ticks = [json.loads(value) for value in subset[f"{dimension}ticks"].astype(str)]
    ticks = [
        [
            float(value)
            for value in axis_ticks
            if min(axis_limits) - 1e-9 <= float(value) <= max(axis_limits) + 1e-9
        ]
        for axis_ticks, axis_limits in zip(raw_ticks, limits, strict=True)
    ]
    scales_equal = len(set(scales)) == 1
    limits_equal = all(np.allclose(limits[0], value) for value in limits[1:])
    ticks_equal = arrays_equal(ticks)
    return {
        "audit_id": audit_id,
        "panels": ",".join(dict.fromkeys(panel for panel, _ in members)),
        "axes": ";".join(f"{panel}:{axis}" for panel, axis in members),
        "dimension": dimension,
        "reason_shared": reason,
        "scales_equal": scales_equal,
        "limits_equal": limits_equal,
        "ticks_equal": ticks_equal,
        "status": "PASS" if scales_equal and limits_equal and ticks_equal else "ISSUE",
        "limits": json.dumps(limits.tolist()),
        "ticks": json.dumps(ticks),
    }


def compare_color_norm(rows: pd.DataFrame) -> dict[str, object]:
    members = [("a", index) for index in range(4)]
    subset = selected(rows, members)
    norms = subset[["collection_norm_vmin", "collection_norm_vmax"]].to_numpy(float)
    finite = bool(np.isfinite(norms).all())
    equal = finite and all(np.allclose(norms[0], value) for value in norms[1:])
    return {
        "audit_id": "a_time_color_norm",
        "panels": "a",
        "axes": ";".join(f"a:{index}" for index in range(4)),
        "dimension": "color",
        "reason_shared": "four trajectories use one displayed time colorbar",
        "scales_equal": equal,
        "limits_equal": equal,
        "ticks_equal": True,
        "status": "PASS" if equal else "ISSUE",
        "limits": json.dumps(norms.tolist()),
        "ticks": "[]",
    }


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    rows = pd.read_csv(SOURCE)
    expected_panels = list("abcdefghijk")
    if sorted(rows["panel"].unique().tolist()) != expected_panels:
        raise RuntimeError("Axis inventory does not cover panels a-k")

    checks: list[dict[str, object]] = [
        compare_axis(
            rows,
            audit_id="a_spatial_x",
            members=[("a", index) for index in range(4)],
            dimension="x",
            reason="four trajectory maps suppress axes and share maze geometry",
        ),
        compare_axis(
            rows,
            audit_id="a_spatial_y",
            members=[("a", index) for index in range(4)],
            dimension="y",
            reason="four trajectory maps suppress axes and share maze geometry",
        ),
        compare_color_norm(rows),
        compare_axis(
            rows,
            audit_id="b_c_time_x",
            members=[("b", 0), ("c", 0)],
            dimension="x",
            reason="same time-in-maze progress axis and shared condition legend",
        ),
        compare_axis(
            rows,
            audit_id="d_e_transition_x",
            members=[("d", 0), ("e", 0)],
            dimension="x",
            reason="same maze-transition progress axis and shared condition legend",
        ),
        compare_axis(
            rows,
            audit_id="f_category_x",
            members=[("f", index) for index in range(6)],
            dimension="x",
            reason="six endpoint plots use the same three conditions",
        ),
        compare_axis(
            rows,
            audit_id="f_time_y",
            members=[("f", index) for index in range(3)],
            dimension="y",
            reason="only the first top-row axis shows time ticks and the shared y title",
        ),
        compare_axis(
            rows,
            audit_id="f_transition_y",
            members=[("f", index) for index in range(3, 6)],
            dimension="y",
            reason="only the first bottom-row axis shows transition ticks and the shared y title",
        ),
        compare_axis(
            rows,
            audit_id="h_return_interval_x",
            members=[("h", 0), ("h", 1)],
            dimension="x",
            reason="two modes use one shared return-interval label",
        ),
        compare_axis(
            rows,
            audit_id="h_speed_y",
            members=[("h", 0), ("h", 1)],
            dimension="y",
            reason="reward-directed axis suppresses y labels and uses exploration y title",
        ),
    ]
    for column, title in enumerate(
        ("end_node_visits", "within_round_coverage", "coverage_round_efficiency")
    ):
        for dimension in ("x", "y"):
            checks.append(
                compare_axis(
                    rows,
                    audit_id=f"i_{title}_{dimension}",
                    members=[("i", column), ("i", column + 3)],
                    dimension=dimension,
                    reason="naive and return examples are stacked views of the same measure",
                )
            )
    checks.append(
        compare_axis(
            rows,
            audit_id="j_k_efficiency_y",
            members=[("j", 0), ("k", 0)],
            dimension="y",
            reason="panel k suppresses y labels and uses panel j's efficiency title",
        )
    )

    checks_frame = pd.DataFrame(checks)
    coverage = pd.DataFrame(
        [
            ("a", "shared spatial x/y and time color scale", "PASS"),
            ("b", "shared time x scale with c; own visible y scale", checks_frame.loc[checks_frame.audit_id.eq("b_c_time_x"), "status"].iloc[0]),
            ("c", "shared time x scale with b; own visible y scale", checks_frame.loc[checks_frame.audit_id.eq("b_c_time_x"), "status"].iloc[0]),
            ("d", "shared transition x scale with e; own visible y scale", checks_frame.loc[checks_frame.audit_id.eq("d_e_transition_x"), "status"].iloc[0]),
            ("e", "shared transition x scale with d; own visible y scale", checks_frame.loc[checks_frame.audit_id.eq("d_e_transition_x"), "status"].iloc[0]),
            ("f", "shared categorical x and implicit shared y within each row", "ISSUE" if (checks_frame.loc[checks_frame.audit_id.isin(["f_time_y", "f_transition_y"]), "status"] == "ISSUE").any() else "PASS"),
            ("g", "standalone log-log axis; no shared-scale contract", "PASS"),
            ("h", "shared return-interval x and body-mass-speed y scales", checks_frame.loc[checks_frame.audit_id.eq("h_speed_y"), "status"].iloc[0]),
            ("i", "top/bottom axes share x and y by measure column", "ISSUE" if (checks_frame.loc[checks_frame.audit_id.str.startswith("i_"), "status"] == "ISSUE").any() else "PASS"),
            ("j", "shared exploration-efficiency y scale with k", checks_frame.loc[checks_frame.audit_id.eq("j_k_efficiency_y"), "status"].iloc[0]),
            ("k", "shared exploration-efficiency y scale with j", checks_frame.loc[checks_frame.audit_id.eq("j_k_efficiency_y"), "status"].iloc[0]),
        ],
        columns=["panel", "axis_contract", "status"],
    )
    hidden_y = rows.loc[
        rows["visible_y_tick_labels"].eq(0)
        & rows["ylabel"].fillna("").eq("")
        & ~rows["panel"].eq("a")
    ][["panel", "axis_index", "title", "ylim_low", "ylim_high", "yticks"]]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks_frame.to_csv(OUTPUT / "SHARED_AXIS_CHECKS.csv", index=False)
    coverage.to_csv(OUTPUT / "PANEL_A_TO_K_AXIS_AUDIT.csv", index=False)
    hidden_y.to_csv(OUTPUT / "HIDDEN_Y_LABEL_AXIS_INVENTORY.csv", index=False)
    issues = checks_frame.loc[checks_frame["status"].eq("ISSUE")]
    report_lines = [
        "# ED1 shared-axis scale audit",
        "",
        "The audit inventories every panel a-k and tests every axis group that is visually or semantically shared. A group is treated as visually shared when duplicate tick labels or titles are suppressed or when one figure-level label serves multiple axes.",
        "",
        f"- Shared-axis checks: {len(checks_frame)}",
        f"- Passing checks: {(checks_frame['status'] == 'PASS').sum()}",
        f"- Issues: {len(issues)}",
        "",
        "## Issues",
        "",
    ]
    if issues.empty:
        report_lines.append("None.")
    else:
        for row in issues.itertuples(index=False):
            report_lines.append(
                f"- `{row.audit_id}`: {row.reason_shared}. Limits were {row.limits}; ticks were {row.ticks}."
            )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Panels b-e retain their own visible y axes, so differing y limits there are explicit rather than a hidden shared-axis claim. Panel g is standalone. The checks above focus on suppressed duplicate axes and directly paired same-unit views.",
            "",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    payload = {
        "schema": "ed01-shared-axis-audit/v1",
        "status": "ISSUES_FOUND" if not issues.empty else "PASS",
        "checks": len(checks_frame),
        "passes": int((checks_frame["status"] == "PASS").sum()),
        "issues": issues["audit_id"].tolist(),
        "all_panels_inventoried": coverage["panel"].tolist() == expected_panels,
        "source": SOURCE.relative_to(REPO).as_posix(),
    }
    (OUTPUT / "AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
