"""Build the full-roster ED2 same-animal Naive reconstruction candidate."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "ed02_1616_restored_depth_common_i_candidate_2026-08-20"
PARENT = REPO / "docs" / "figure_version_candidates" / PARENT_NAME
ANALYSIS = REPO / "docs" / "ed02_matched_naive_reconstruction_1616_2026-08-20"
CANDIDATE = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed02_1616_restored_matched_naive_candidate_2026-08-20"
)
COMPOSITE_STEM = "ed02_1616_restored_matched_naive_candidate"
CHANGED_PANELS = {"d", "h"}
ANALYSIS_SUBDIR = "round27_matched_naive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def selected_columns_match(
    source_path: Path,
    rendered_path: Path,
    columns: list[str],
) -> bool:
    """Compare scientific columns while tolerating CSV float serialization."""
    source = pd.read_csv(source_path)[columns].reset_index(drop=True)
    rendered = pd.read_csv(rendered_path)[columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            source,
            rendered,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        return False
    return True


def update_manifest_hashes(path: Path, root: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    fields = list(rows[0])
    for row in rows:
        artifact = root / row["path"]
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if "sha256" in row:
            row["sha256"] = sha256(artifact)
        if "bytes" in row:
            row["bytes"] = str(artifact.stat().st_size)
        if "rows" in row and artifact.suffix.lower() == ".csv":
            row["rows"] = str(len(pd.read_csv(artifact)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_manifest(root: Path) -> None:
    rows = []
    for path in sorted((root / "analysis_inputs").rglob("*.csv")):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(pd.read_csv(path)),
            }
        )
    pd.DataFrame(rows).to_csv(root / "ANALYSIS_INPUT_MANIFEST.csv", index=False)


def replace_analysis_inputs() -> None:
    replacements = {
        "primary_30min_strict_joint_policy_contribution_summary.csv": (
            ANALYSIS / "plot_ready" / "panel_d_policy_contributions.csv"
        ),
        "primary_30min_strict_joint_gap_recapitulation_summary.csv": (
            ANALYSIS / "plot_ready" / "panel_h_four_vs_depth_gap.csv"
        ),
    }
    strict = CANDIDATE / "analysis_inputs" / "strict"
    for filename, source in replacements.items():
        shutil.copy2(source, strict / filename)


def refresh_cohort_audit() -> None:
    """Replace the pooled-donor panel-h audit with matched-pair membership."""
    audit = CANDIDATE / "cohort_audit"
    source = ANALYSIS / "source_tables"
    for stale in (
        "depth_common_cluster_bootstrap_draws.csv",
        "depth_common_counterfactual_session_rows.csv",
        "four_policy_cluster_bootstrap_draws.csv",
        "four_policy_counterfactual_session_rows.csv",
        "four_policy_point_summary.csv",
    ):
        (audit / stale).unlink(missing_ok=True)
    for path in sorted(source.glob("*.csv")):
        shutil.copy2(path, audit / path.name)

    pairing = pd.read_csv(source / "return_to_naive_pairing_and_support.csv")
    common = pairing.loc[pairing["pair_common_primary_eligible"].astype(bool)].copy()
    return_sessions = set(common["nickname"].astype(str))
    naive_sessions = set(common["matched_naive_session"].astype(str))
    if len(return_sessions) != 54 or len(naive_sessions) != 17:
        raise RuntimeError(
            f"Unexpected matched panel-h membership: {len(return_sessions)} Return, "
            f"{len(naive_sessions)} Naive"
        )

    membership_path = audit / "session_membership.csv"
    membership = pd.read_csv(membership_path)
    missing_naive = naive_sessions - set(
        membership.loc[membership["group"].eq("naive"), "nickname"].astype(str)
    )
    if missing_naive:
        naive_inputs = pd.read_csv(source / "matched_naive_policy_rows.csv")
        naive_inputs = naive_inputs.loc[
            naive_inputs["nickname"].astype(str).isin(missing_naive)
        ].copy()
        if set(naive_inputs["nickname"].astype(str)) != missing_naive:
            raise RuntimeError("Missing matched Naive policy rows for panel-h audit")
        appended = pd.DataFrame(
            {
                "window_minutes": naive_inputs["window_minutes"],
                "nickname": naive_inputs["nickname"],
                "group": "naive",
                "animal_id": naive_inputs["animal_id"],
                "n_Bf": naive_inputs["n_Bf"],
                "n_Ba": naive_inputs["n_Ba"],
                "n_Lf": naive_inputs["n_Lf"],
                "n_Lo": naive_inputs["n_Lo"],
                "n_Lo_shallow": naive_inputs["n_Lo_shallow"],
                "n_Lo_mid": naive_inputs["n_Lo_mid"],
                "n_Lo_deep": naive_inputs["n_Lo_deep"],
                "primary_policy_support": naive_inputs["primary_policy_support"],
                "primary_depth_support": naive_inputs["primary_depth_support"],
                "included_panel_b": False,
                "included_panel_c": False,
                "included_panel_d": False,
                "included_panel_e": False,
                "included_panel_g": False,
                "included_panel_h": True,
                "identity_excluded": False,
                "cohort_note": (
                    "same-animal Naive donor added for Round-27 matched reconstruction"
                ),
            }
        )
        membership = pd.concat([membership, appended], ignore_index=True)
    membership["included_panel_h"] = False
    membership.loc[
        membership["group"].eq("LTM")
        & membership["nickname"].astype(str).isin(return_sessions),
        "included_panel_h",
    ] = True
    membership.loc[
        membership["group"].eq("naive")
        & membership["nickname"].astype(str).isin(naive_sessions),
        "included_panel_h",
    ] = True
    drawn_return = set(
        membership.loc[
            membership["group"].eq("LTM") & membership["included_panel_h"],
            "nickname",
        ].astype(str)
    )
    drawn_naive = set(
        membership.loc[
            membership["group"].eq("naive") & membership["included_panel_h"],
            "nickname",
        ].astype(str)
    )
    if drawn_return != return_sessions or drawn_naive != naive_sessions:
        raise RuntimeError("Matched panel-h sessions are not closed in session_membership.csv")
    membership.to_csv(membership_path, index=False)

    panel_counts_path = audit / "panel_cohort_counts.csv"
    panel_counts = pd.read_csv(panel_counts_path)
    panel_counts.loc[
        panel_counts["panel"].eq("h") & panel_counts["group"].eq("naive"),
        ["support_rule", "n_sessions", "n_mice", "contains_640day1"],
    ] = ["matched four-policy and all-depth support", 17, 17, False]
    panel_counts.loc[
        panel_counts["panel"].eq("h") & panel_counts["group"].eq("LTM"),
        ["support_rule", "n_sessions", "n_mice", "contains_640day1"],
    ] = ["matched four-policy and all-depth support", 54, 17, False]
    panel_counts.to_csv(panel_counts_path, index=False)

    drawn_counts_path = audit / "drawn_panel_cohort_counts.csv"
    drawn_counts = pd.read_csv(drawn_counts_path)
    drawn_counts.loc[
        drawn_counts["panel"].eq("h") & drawn_counts["group"].eq("naive"),
        ["n_sessions", "n_mice", "contains_640day1"],
    ] = [17, 17, False]
    drawn_counts.loc[
        drawn_counts["panel"].eq("h") & drawn_counts["group"].eq("LTM"),
        ["n_sessions", "n_mice", "contains_640day1"],
    ] = [54, 17, False]
    drawn_counts.to_csv(drawn_counts_path, index=False)


def rewrite_caption_notes() -> None:
    path = CANDIDATE / "CAPTION_NOTES.md"
    text = path.read_text(encoding="utf-8")
    panel_d = pd.read_csv(ANALYSIS / "plot_ready" / "panel_d_policy_contributions.csv")
    panel_h = pd.read_csv(ANALYSIS / "plot_ready" / "panel_h_four_vs_depth_gap.csv")
    policy_tex = {
        "Bf": "P_SF",
        "Ba": "P_SA",
        "Lf": "P_BF",
        "Lo": "P_BS",
    }
    d_text = ", ".join(
        f"{policy_tex[row.parameter]} {100 * row.delta_efficiency_mean:+.1f} "
        f"percentage points [{100 * row.delta_efficiency_ci_low:+.1f}, "
        f"{100 * row.delta_efficiency_ci_high:+.1f}]"
        for row in panel_d.itertuples(index=False)
    )
    h_lookup = panel_h.set_index("model")
    four = h_lookup.loc["four_bias"]
    depth = h_lookup.loc["depth_stratified"]
    text = text.replace(
        "# Extended Data Figure 2 Round 26 caption notes",
        "# Extended Data Figure 2 Round 27 caption notes",
        1,
    )
    text = re.sub(
        r"\*\*d,\*\*.*?(?=\n\n\*\*e,\*\*)",
        "**d,** Change in simulated Return-day exploration efficiency when one policy "
        "at a time was replaced by the value measured in the same animal's Naive "
        "session. Estimates and animal-cluster bootstrap 95% confidence intervals "
        f"were {d_text}. Positive values "
        "indicate greater efficiency with the observed Return-day policy. Because "
        "policies were substituted one at a time, these changes are nonlinear and "
        "should not be added.",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\*\*h,\*\*.*?(?=\n\n\*\*i,\*\*)",
        "**h,** Reconstruction of the paired Return-minus-Naive half-coverage "
        "efficiency difference. Each Return session was simulated using either its "
        "own policies or the policies measured in the same animal's Naive session. "
        "The common four-policy and depth-support cohort contained 54 Return sessions "
        "and 17 matched Naive baselines from 17 animals. The four-policy walk "
        f"reconstructed {100 * four.fraction_recapitulated:.1f}% "
        f"[{100 * four.fraction_ci_low:.1f}, {100 * four.fraction_ci_high:.1f}] of "
        "the observed paired difference; the depth-varying model reconstructed "
        f"{100 * depth.fraction_recapitulated:.1f}% "
        f"[{100 * depth.fraction_ci_low:.1f}, {100 * depth.fraction_ci_high:.1f}]. "
        "Intervals are 95% "
        "animal-cluster bootstrap confidence intervals. Reconstruction is "
        "contemporaneous and is neither prospective prediction nor variance explained.",
        text,
        flags=re.S,
    )
    text += (
        "\nRound-27 scientific correction: panels d and h now use each animal's own "
        "Naive session rather than a pooled Naive-group policy mean. The accepted "
        "58-session Return roster is fully paired for the four-policy substitutions "
        "in d. Panel h contains 54 Return sessions from 17 animals because mouse 640's "
        "Naive baseline has only two shallow-depth P_BS events and therefore fails the "
        "common depth-support rule. Panels a-c, e-g and i are unchanged. Monte Carlo "
        "walk noise and within-session policy-estimation noise are not propagated.\n"
    )
    path.write_text(text, encoding="utf-8")


def build() -> dict[str, Any]:
    if CANDIDATE.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {CANDIDATE}")
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    if not (ANALYSIS / "RUN_MANIFEST.json").is_file():
        raise FileNotFoundError(ANALYSIS / "RUN_MANIFEST.json")

    shutil.copytree(PARENT, CANDIDATE)
    for stale in (
        "RECORD_HASHES.csv",
        "RECORD_VALIDATION.json",
        "PROMOTION_COMPLETE.txt",
    ):
        (CANDIDATE / stale).unlink(missing_ok=True)

    provenance = CANDIDATE / "analysis_provenance" / ANALYSIS_SUBDIR
    shutil.copytree(ANALYSIS, provenance)
    shutil.copy2(
        REPO / "scripts" / "analyze_ed02_matched_naive_reconstruction.py",
        CANDIDATE / "source_code" / "analyze_ed02_matched_naive_reconstruction.py",
    )
    shutil.copy2(Path(__file__), CANDIDATE / "source_code" / Path(__file__).name)
    replace_analysis_inputs()
    refresh_cohort_audit()

    stage = CANDIDATE / ".round27_render_stage"
    renderer = load_module(
        REPO / "scripts" / "render_ed02_slide3_aligned_round11.py",
        "ed02_round27_round11_renderer",
    )
    round11_build_panel_h = renderer.build_panel_h

    def build_panel_h_without_left_clip(
        renderer_state: Any,
        frame: pd.DataFrame,
    ) -> tuple[Any, dict[str, pd.DataFrame]]:
        fig, tables = round11_build_panel_h(renderer_state, frame)
        # The matched-donor interval begins farther left than the superseded
        # pooled-donor interval, leaving insufficient horizontal room for the
        # original one-line label. Wrapping only that label preserves the axes,
        # data positions, typography, and composite layout without clipping.
        for label in fig.axes[0].texts:
            if label.get_text() == "Four-policy":
                label.set_text("Four-\npolicy")
                label.set_linespacing(0.9)
        return fig, tables

    renderer.build_panel_h = build_panel_h_without_left_clip
    try:
        renderer.render(CANDIDATE, stage, 600)
        for panel in CHANGED_PANELS:
            for source in (stage / "panels").glob(f"ed02{panel}*"):
                shutil.copy2(source, CANDIDATE / "panels" / source.name)
            for source in (stage / "plot_ready").glob(f"panel_{panel}_*"):
                shutil.copy2(source, CANDIDATE / "plot_ready" / source.name)
            for source in (stage / "ppt_vector").glob(f"ed02{panel}*"):
                shutil.copy2(source, CANDIDATE / "ppt_vector" / source.name)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    rewrite_caption_notes()
    for manifest_name in (
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
    ):
        update_manifest_hashes(CANDIDATE / manifest_name, CANDIDATE)
    write_analysis_manifest(CANDIDATE)

    source_manifest_path = CANDIDATE / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "status": "unpromoted_scientific_review_candidate",
            "source_candidate": f"docs/figure_version_candidates/{PARENT_NAME}",
            "review_round": 29,
            "round27_matched_naive_reconstruction": {
                "changed_panels": sorted(CHANGED_PANELS),
                "donor": "same animal's original Naive Day-1 policy vector",
                "panel_d_support": "58 Return sessions from 18 animals",
                "panel_h_support": "54 Return sessions and 17 Naive baselines from 17 animals",
                "panel_h_exclusion": (
                    "four mouse-640 Return sessions because the matched 640day1 "
                    "baseline has 2 shallow-depth P_BS events"
                ),
                "analysis_manifest": (
                    f"analysis_provenance/{ANALYSIS_SUBDIR}/RUN_MANIFEST.json"
                ),
            },
        }
    )
    write_json(source_manifest_path, source_manifest)

    exporter = load_module(
        REPO / "scripts" / "export_ed02_round11_vector_figure.py",
        "ed02_round27_exporter",
    )
    composite = exporter.export(
        CANDIDATE,
        CANDIDATE / "composite",
        composite_stem=COMPOSITE_STEM,
        review_round=29,
        manifest_name="COMPOSITE_MANIFEST.json",
    )

    panel_d = pd.read_csv(CANDIDATE / "plot_ready" / "panel_d_policy_contributions.csv")
    panel_h = pd.read_csv(CANDIDATE / "plot_ready" / "panel_h_four_vs_depth_gap.csv")
    cohort_counts = pd.read_csv(
        CANDIDATE / "cohort_audit" / "drawn_panel_cohort_counts.csv"
    )
    cohort_lookup = {
        (row.panel, row.group): (int(row.n_sessions), int(row.n_mice))
        for row in cohort_counts.itertuples(index=False)
    }
    membership = pd.read_csv(CANDIDATE / "cohort_audit" / "session_membership.csv")
    d_lookup = panel_d.set_index("parameter")["delta_efficiency_mean"].to_dict()
    h_lookup = panel_h.set_index("model")["fraction_recapitulated"].to_dict()

    unchanged_panels = all(
        sha256(path) == sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob("*")
        if path.is_file() and path.name[4] not in CHANGED_PANELS
    )
    changed_panels = all(
        sha256(path) != sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob("*")
        if path.is_file() and path.name[4] in CHANGED_PANELS
    )
    unchanged_tables = all(
        sha256(path) == sha256(PARENT / "plot_ready" / path.name)
        for path in (CANDIDATE / "plot_ready").glob("panel_*.csv")
        if path.name[6] not in CHANGED_PANELS
    )
    checks = {
        "panel_d_values_finite": bool(
            panel_d[
                [
                    "delta_efficiency_mean",
                    "delta_efficiency_ci_low",
                    "delta_efficiency_ci_high",
                ]
            ].notna().all().all()
        ),
        "panel_d_support_58_return_18_mice": (
            set(panel_d["n_return_rows"].astype(int)) == {58}
            and set(panel_d["n_return_mice"].astype(int)) == {18}
        ),
        "panel_h_values_finite": bool(
            panel_h[
                ["fraction_recapitulated", "fraction_ci_low", "fraction_ci_high"]
            ].notna().all().all()
        ),
        "panel_h_support_54_pairs_17_mice": (
            set(panel_h["n_return_rows"].astype(int)) == {54}
            and set(panel_h["n_return_mice"].astype(int)) == {17}
            and set(panel_h["n_naive_rows"].astype(int)) == {17}
        ),
        "panel_h_cohort_audit_matches_54_return_17_naive": (
            cohort_lookup.get(("h", "LTM")) == (54, 17)
            and cohort_lookup.get(("h", "naive")) == (17, 17)
            and len(
                membership.loc[
                    membership["group"].eq("LTM")
                    & membership["included_panel_h"].astype(bool)
                ]
            )
            == 54
            and len(
                membership.loc[
                    membership["group"].eq("naive")
                    & membership["included_panel_h"].astype(bool)
                ]
            )
            == 17
        ),
        "strict_sources_match_plot_ready": (
            selected_columns_match(
                CANDIDATE
                / "analysis_inputs"
                / "strict"
                / "primary_30min_strict_joint_policy_contribution_summary.csv",
                CANDIDATE / "plot_ready" / "panel_d_policy_contributions.csv",
                [
                    "window_minutes",
                    "parameter",
                    "n_return_rows",
                    "n_return_mice",
                    "delta_efficiency_mean",
                    "delta_efficiency_ci_low",
                    "delta_efficiency_ci_high",
                    "contrast",
                    "point_weighting",
                    "uncertainty",
                ],
            )
            and selected_columns_match(
                CANDIDATE
                / "analysis_inputs"
                / "strict"
                / "primary_30min_strict_joint_gap_recapitulation_summary.csv",
                CANDIDATE / "plot_ready" / "panel_h_four_vs_depth_gap.csv",
                [
                    "window_minutes",
                    "model",
                    "n_naive_rows",
                    "n_return_rows",
                    "n_naive_mice",
                    "n_return_mice",
                    "delta_real",
                    "delta_simulated",
                    "fraction_recapitulated",
                    "fraction_ci_low",
                    "fraction_ci_high",
                    "point_weighting",
                    "uncertainty",
                    "model_norm",
                    "denominator_definition",
                    "numerator_definition",
                ],
            )
        ),
        "only_d_h_panels_changed": unchanged_panels and changed_panels,
        "only_d_h_plot_ready_tables_changed": unchanged_tables,
        "nine_panels_four_formats": all(
            len(list((CANDIDATE / "panels").glob(f"*.{suffix}"))) == 9
            for suffix in ("svg", "pdf", "png", "tiff")
        ),
        "composite_vector_only": composite["vector_audit"]["embedded_raster_images"]
        == 0,
        "composite_600dpi_square": composite["png_audit"]["size"] == [4500, 4500]
        and composite["tiff_audit"]["size"] == [4500, 4500],
        "caption_has_matched_not_pooled_donor": (
            "same animal's Naive session"
            in (CANDIDATE / "CAPTION_NOTES.md").read_text(encoding="utf-8")
            and "Round-27 scientific correction" in (
                CANDIDATE / "CAPTION_NOTES.md"
            ).read_text(encoding="utf-8")
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    qa = {"status": status, "checks": checks, "composite": composite}
    write_json(CANDIDATE / "QA.json", qa)
    (CANDIDATE / "QA.md").write_text(
        "# ED2 Round 27 same-animal Naive candidate QA\n\n"
        f"- Overall: `{status.upper()}`\n"
        + "\n".join(
            f"- {'PASS' if passed else 'FAIL'}: {name}"
            for name, passed in checks.items()
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "status": status,
        "parent": PARENT_NAME,
        "changed_panels": sorted(CHANGED_PANELS),
        "four_policy_fraction": float(h_lookup["four_bias"]),
        "depth_varying_fraction": float(h_lookup["depth_stratified"]),
        "panel_h_return_sessions": int(panel_h.iloc[0]["n_return_rows"]),
        "panel_h_mice": int(panel_h.iloc[0]["n_return_mice"]),
    }
    write_json(CANDIDATE / "DATA_BUILD_RESULT.json", result)
    (CANDIDATE / "REPORT.md").write_text(
        "# ED2 Round 27 same-animal Naive reconstruction candidate\n\n"
        f"Status: **{status.upper()}**.\n\n"
        "Panels d and h replace the pooled Naive-group donor with each mouse's own "
        "Naive-session policy values. Panel d retains 58 Return sessions from 18 "
        "animals. Panel h uses 54 Return sessions from 17 animals because the matched "
        "640day1 baseline fails the shallow-depth support rule. The four-policy and "
        f"depth-varying reconstructions are {100 * h_lookup['four_bias']:.1f}% and "
        f"{100 * h_lookup['depth_stratified']:.1f}%, respectively. All other "
        "panels and plot-ready tables are byte-identical to Round 26.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError(f"Candidate QA failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
