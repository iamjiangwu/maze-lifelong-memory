"""Build the final full-roster ED2 candidate with clear panel-h labels."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "ed02_1616_restored_matched_naive_candidate_2026-08-20"
PARENT = REPO / "docs" / "figure_version_candidates" / PARENT_NAME
CANDIDATE = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed02_round29_1616_restored_candidate_2026-08-20"
)
COMPOSITE_STEM = "ed02_round29_1616_restored_candidate"
PANEL_H_STEM = "ed02h_four_vs_depth_reconstruction_round11"
LABEL_CLEARANCE = 0.018


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


def update_manifest_hashes(path: Path, root: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
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


def write_source_code_manifest(root: Path) -> None:
    source_code = root / "source_code"
    with (root / "SOURCE_CODE_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(source_code.glob("*"), key=lambda item: item.name.lower()):
            if path.is_file():
                writer.writerow(
                    (path.relative_to(root).as_posix(), sha256(path), path.stat().st_size)
                )


def build() -> dict[str, Any]:
    if CANDIDATE.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {CANDIDATE}")
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    shutil.copytree(PARENT, CANDIDATE)
    for stale in (
        "RECORD_HASHES.csv",
        "RECORD_VALIDATION.json",
        "PROMOTION_COMPLETE.txt",
    ):
        (CANDIDATE / stale).unlink(missing_ok=True)

    shutil.copy2(Path(__file__), CANDIDATE / "source_code" / Path(__file__).name)
    label_audit: dict[str, dict[str, float | str]] = {}
    renderer = load_module(
        REPO / "scripts" / "render_ed02_slide3_aligned_round11.py",
        "ed02_round28_round11_renderer",
    )
    round11_build_panel_h = renderer.build_panel_h

    def build_panel_h_with_clear_upright_labels(
        renderer_state: Any,
        frame: pd.DataFrame,
    ) -> tuple[Any, dict[str, pd.DataFrame]]:
        fig, tables = round11_build_panel_h(renderer_state, frame)
        rows = tables["four_vs_depth_gap"].set_index("model_norm")
        specifications = {
            "Four-policy": ("four_bias", "Four-\npolicy"),
            r"Depth $P_{BS}$": (
                "depth_stratified",
                r"Depth $\mathrm{P}_{\mathrm{BS}}$",
            ),
        }
        found: set[str] = set()
        for label in fig.axes[0].texts:
            original = label.get_text()
            if original not in specifications:
                continue
            model, replacement = specifications[original]
            interval_left = float(rows.loc[model, "delta_simulated_ci_low"])
            label_x = interval_left - LABEL_CLEARANCE
            label.set_text(replacement)
            label.set_x(label_x)
            label.set_ha("right")
            if model == "four_bias":
                label.set_linespacing(0.9)
            label_audit[model] = {
                "text": replacement,
                "label_right_anchor": label_x,
                "interval_left": interval_left,
                "clearance_data_units": interval_left - label_x,
            }
            found.add(model)
        if found != {"four_bias", "depth_stratified"}:
            raise RuntimeError(f"Could not locate both panel-h model labels: {found}")
        return fig, tables

    renderer.build_panel_h = build_panel_h_with_clear_upright_labels
    stage = CANDIDATE / ".round28_render_stage"
    try:
        renderer.render(CANDIDATE, stage, 600)
        for source in (stage / "panels").glob(f"{PANEL_H_STEM}.*"):
            shutil.copy2(source, CANDIDATE / "panels" / source.name)
        for source in (stage / "ppt_vector").glob(f"{PANEL_H_STEM}.*"):
            shutil.copy2(source, CANDIDATE / "ppt_vector" / source.name)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    caption_path = CANDIDATE / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.replace(
        "# Extended Data Figure 2 Round 27 caption notes",
        "# Extended Data Figure 2 Round 29 caption notes",
        1,
    )
    caption += (
        "\nRound-29 display preservation: panel h retains the exact full-roster values, "
        "intervals, cohort, axes and frame. The two model labels are placed with "
        "positive clearance from the confidence intervals, and P_BS is upright "
        "roman. All other panels and every plot-ready table are unchanged.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    for manifest_name in ("PANEL_MANIFEST.csv", "PPT_VECTOR_MANIFEST.csv"):
        update_manifest_hashes(CANDIDATE / manifest_name, CANDIDATE)
    source_manifest_path = CANDIDATE / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "status": "unpromoted_display_review_candidate",
            "source_candidate": f"docs/figure_version_candidates/{PARENT_NAME}",
            "review_round": 29,
            "round28_panel_h_label_correction": {
                "scientific_values_changed": False,
                "changed_panel": "h",
                "label_clearance_data_units": LABEL_CLEARANCE,
                "policy_typography": "upright roman P_BS",
            },
        }
    )
    write_json(source_manifest_path, source_manifest)
    write_source_code_manifest(CANDIDATE)

    exporter = load_module(
        REPO / "scripts" / "export_ed02_round11_vector_figure.py",
        "ed02_round28_exporter",
    )
    composite = exporter.export(
        CANDIDATE,
        CANDIDATE / "composite",
        composite_stem=COMPOSITE_STEM,
        review_round=29,
        manifest_name="COMPOSITE_MANIFEST.json",
    )

    unchanged_non_h = all(
        sha256(path) == sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob("*")
        if path.is_file() and path.name[4] != "h"
    )
    changed_h = all(
        sha256(path) != sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob(f"{PANEL_H_STEM}.*")
    )
    unchanged_tables = all(
        sha256(path) == sha256(PARENT / "plot_ready" / path.name)
        for path in (CANDIDATE / "plot_ready").glob("panel_*.csv")
    )
    svg = (CANDIDATE / "panels" / f"{PANEL_H_STEM}.svg").read_text(
        encoding="utf-8"
    )
    depth_start = svg.index("<!-- Depth")
    depth_block = svg[depth_start : depth_start + 1000]
    checks = {
        "only_panel_h_artwork_changed": unchanged_non_h and changed_h,
        "all_plot_ready_tables_byte_identical": unchanged_tables,
        "both_label_clearances_positive": (
            set(label_audit) == {"four_bias", "depth_stratified"}
            and all(
                abs(float(row["clearance_data_units"]) - LABEL_CLEARANCE) < 1e-12
                for row in label_audit.values()
            )
        ),
        "depth_pbs_is_upright": (
            "mathrm" in label_audit["depth_stratified"]["text"]
            and "font: italic" not in depth_block
        ),
        "composite_vector_only": composite["vector_audit"]["embedded_raster_images"]
        == 0,
        "composite_600dpi_square": composite["png_audit"]["size"] == [4500, 4500]
        and composite["tiff_audit"]["size"] == [4500, 4500],
    }
    status = "pass" if all(checks.values()) else "fail"
    qa = {
        "status": status,
        "checks": checks,
        "label_audit": label_audit,
        "composite": composite,
    }
    write_json(CANDIDATE / "QA.json", qa)
    (CANDIDATE / "QA.md").write_text(
        "# ED2 Round 29 full-roster panel-h label QA\n\n"
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
        "changed_panel": "h",
        "scientific_values_changed": False,
        "label_audit": label_audit,
    }
    write_json(CANDIDATE / "DATA_BUILD_RESULT.json", result)
    (CANDIDATE / "REPORT.md").write_text(
        "# ED2 Round 29 full-roster panel-h label candidate\n\n"
        f"Status: **{status.upper()}**. Panel h uses a dedicated label column with "
        f"{LABEL_CLEARANCE:.3f} efficiency units of clearance before each confidence "
        "interval. P_BS is upright roman. Scientific values, axes, cohort and all "
        "other panels are unchanged from the preceding full-roster candidate.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError(f"Candidate QA failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
