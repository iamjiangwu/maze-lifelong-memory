"""Close the ED5 round-20 version record after a validated source build.

This script writes record metadata and the complete hash ledger. It validates
the already-exported Desktop artwork but does not mutate the Desktop, current
pointers, the archive, or manuscript files.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


RECORD = Path(__file__).resolve().parents[1]
VOR_ROOT = RECORD.parent
REPO = VOR_ROOT.parents[1]
PARENT_NAME = "extended_data_fig05_2026-09-07_round19_current_gallery"
TARGET_NAME = "extended_data_fig05_2026-09-08_round20_session_pair_means"
DESKTOP_CURRENT_RELATIVE = Path(
    "version_record_panels_2026-08-11_round2/extended_data_figure_05"
)
DESKTOP_ARCHIVE_RELATIVE = Path(
    "version_record_archive_2026-09-08/extended_data_figure_05_round19"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*")):
            if path.is_file() and path != ledger:
                writer.writerow(
                    (
                        path.relative_to(record).as_posix(),
                        sha256(path),
                        path.stat().st_size,
                    )
                )


def desktop_validation(desktop_root: Path, parent: Path) -> dict[str, object]:
    current = desktop_root / DESKTOP_CURRENT_RELATIVE
    archive = desktop_root / DESKTOP_ARCHIVE_RELATIVE
    public_files = [
        "ed05_version_record_300dpi.png",
        "ed05_version_record_600dpi.png",
        "ed05_version_record_600dpi.tiff",
        "ed05_version_record.pdf",
        "ed05_version_record.svg",
        "individual_panels/ed05_panel_a_place_map_gallery.png",
        "individual_panels/ed05_panel_a_place_map_gallery.tiff",
        "individual_panels/ed05_panel_a_place_map_gallery.pdf",
        "individual_panels/ed05_panel_a_place_map_gallery.svg",
    ]
    current_matches = {
        name: (current / name).is_file()
        and sha256(current / name) == sha256(RECORD / name)
        for name in public_files
    }
    archived_files = sorted(path for path in archive.rglob("*") if path.is_file())
    archive_matches = {
        path.relative_to(archive).as_posix(): (
            parent / path.relative_to(archive)
        ).is_file()
        and sha256(path) == sha256(parent / path.relative_to(archive))
        for path in archived_files
    }
    result = {
        "status": "PASS",
        "desktop_root": str(desktop_root),
        "current_folder": str(current),
        "archived_round19_folder": str(archive),
        "current_public_files_match_record_by_sha256": all(current_matches.values()),
        "top_level_png_matches_record_by_sha256": sha256(
            desktop_root / "ed05_version_record_600dpi.png"
        )
        == sha256(RECORD / "ed05_version_record_600dpi.png"),
        "archived_round19_file_count": len(archived_files),
        "archived_round19_files_match_parent_by_sha256": bool(archived_files)
        and all(archive_matches.values()),
        "current_files": current_matches,
        "archive_files": archive_matches,
    }
    if not all(
        result[key]
        for key in (
            "current_public_files_match_record_by_sha256",
            "top_level_png_matches_record_by_sha256",
            "archived_round19_files_match_parent_by_sha256",
        )
    ):
        result["status"] = "FAIL"
        raise AssertionError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--desktop-root", required=True, type=Path)
    args = parser.parse_args()
    if RECORD.name != TARGET_NAME:
        raise AssertionError(RECORD)
    parent = VOR_ROOT / PARENT_NAME
    if not parent.is_dir():
        parent = VOR_ROOT.parent / "figure_version_records_archive_2026-08-09" / PARENT_NAME
    if not parent.is_dir():
        raise FileNotFoundError(PARENT_NAME)
    build_root = args.build_dir.resolve()
    build = build_root / "figure_rebuild/assembly"
    source_result = json.loads(
        (build_root / "SOURCE_REBUILD_RESULT.json").read_text(encoding="utf-8")
    )
    if source_result["status"] != "PASS":
        raise AssertionError(source_result)
    checks = pd.read_csv(build / "ED05_VALIDATION_ROUND20.csv")
    if not checks["status"].eq("PASS").all():
        raise AssertionError(checks.loc[~checks["status"].eq("PASS")].to_dict("records"))

    old = np.asarray(Image.open(parent / "ed05_version_record_600dpi.png").convert("RGB"))
    new = np.asarray(Image.open(RECORD / "ed05_version_record_600dpi.png").convert("RGB"))
    if old.shape != new.shape:
        raise AssertionError((old.shape, new.shape))
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    if not len(ys) or changed[:3400].any() or changed[3400:, 2350:].any():
        raise AssertionError("Round-20 raster changes were not confined to panels b-d")

    descriptive = pd.read_csv(
        RECORD / "plot_ready/ed05_day_combination_session_pair_descriptive_cr1.csv"
    )
    contrasts = pd.read_csv(
        RECORD / "plot_ready/ed05_day_combination_session_pair_contrasts_cr1_bh3.csv"
    )
    metric_order = (
        "place_map_correlation",
        "pv_correlation",
        "ensemble_rate_correlation",
    )
    variant_order = ("d1v1", "d1v2", "d2v1", "d2v2")
    exact_means = {
        metric: [
            float(
                descriptive.loc[
                    descriptive["metric"].eq(metric)
                    & descriptive["variant"].eq(variant),
                    "mean",
                ].iloc[0]
            )
            for variant in variant_order
        ]
        for metric in metric_order
    }
    adjusted_p = {
        metric: contrasts.loc[
            contrasts["metric"].eq(metric),
            "p_bh3_within_metric_exact_cluster_signflip",
        ].astype(float).tolist()
        for metric in metric_order
    }

    desktop = desktop_validation(args.desktop_root.resolve(), parent)
    write_json(
        RECORD / "source_validation/DESKTOP_EXPORT_VALIDATION.json", desktop
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    validation = {
        "status": "PASS",
        "record": TARGET_NAME,
        "parent_record": PARENT_NAME,
        "created_utc": timestamp,
        "change_scope": "ED5 panels b-d aggregation and statistics only",
        "session_pairs_per_metric_variant": 36,
        "animals": 14,
        "representation_space": {
            "place_map": "15x15 grid-position rate maps",
            "population_vector": "matched-neuron population vectors on common spatial support",
            "ensemble_rate": "matched-neuron session rates on ED5 common-neuron support",
        },
        "reward_context": "long-gap rewarded maze-session day combinations",
        "point_estimate": "equal-weighted mean across session-pair correlations",
        "uncertainty": "intercept-only CR1 clustered by animal; Student t with 13 df",
        "primary_test": "two-sided exact mouse-cluster sign flip of the equal-session-pair mean difference",
        "multiplicity": "BH3 separately within each neural metric across the three d1v1 contrasts",
        "exact_means": exact_means,
        "caption_means_two_decimals": {
            metric: [f"{value:.2f}" for value in values]
            for metric, values in exact_means.items()
        },
        "bh3_exact_cluster_signflip_p": adjusted_p,
        "maps_or_cimaps_recomputed": False,
        "pair_correlations_recomputed": False,
        "saved_pair_correlations_reaggregated": True,
        "derived_tables_recomputed_from_record_pair_sources": True,
        "source_only_figure_rebuild": source_result,
        "accepted_artwork_used_as_scientific_input": False,
        "raster_changed_bbox_exclusive": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ],
        "panel_a_pixels_unchanged": True,
        "panel_e_pixels_unchanged": True,
        "statistical_significance_pattern_unchanged": {
            "place_map_significant_brackets": 0,
            "population_vector_significant_brackets": 3,
            "ensemble_rate_significant_brackets": 3,
        },
        "local_caption_files_updated": [
            "docs/manuscript_figure_captions_2026-08-10/figure_captions_main1-5_extended1-10.tex",
            "docs/manuscript_figure_captions_2026-08-10/figure_pages_with_captions_stats_audited_2026-08-16.tex",
        ],
        "overleaf_modified": False,
        "desktop_export": desktop,
        "separate_estimator_note": "ED5 ensemble-rate correlations retain the existing common-neuron-support gate; this aggregation-only update does not replace them with the ungated Figure 2 estimator.",
    }
    write_json(RECORD / "RECORD_VALIDATION.json", validation)
    write_json(
        RECORD / "SESSION_PAIR_AGGREGATION_UPDATE_2026_09_08.json", validation
    )

    with (RECORD / "SOURCE_DATA_AUDIT.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("role", "path", "sha256", "bytes"))
        for path in sorted((RECORD / "plot_ready").iterdir()):
            if path.is_file():
                writer.writerow(
                    (
                        path.stem,
                        path.relative_to(RECORD).as_posix(),
                        sha256(path),
                        path.stat().st_size,
                    )
                )

    manifest = json.loads(
        (build / "MANIFEST_ROUND20.json").read_text(encoding="utf-8")
    )
    prefix = RECORD.relative_to(REPO).as_posix()
    manifest.update(
        {
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "outputs": [
                prefix + "/ed05_version_record" + suffix
                for suffix in (".svg", ".pdf", "_600dpi.png", "_600dpi.tiff")
            ],
            "source_audit": prefix + "/SOURCE_DATA_AUDIT.csv",
            "validation": prefix + "/ED05_VALIDATION.csv",
            "caption_notes": prefix + "/CAPTION_NOTES.md",
            "source_update": validation,
            "viewable_without_powerpoint": {
                "png": prefix + "/extended_data_figure5_source_rerender.png",
                "tiff": prefix + "/extended_data_figure5_source_rerender.tiff",
            },
            "gallery_outputs": {
                suffix: prefix
                + "/individual_panels/ed05_panel_a_place_map_gallery."
                + suffix
                for suffix in ("svg", "pdf", "png", "tiff")
            },
        }
    )
    manifest["gallery_full_map_contract"]["correlation_label_audit"] = (
        prefix + "/plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv"
    )
    write_json(RECORD / "ED05_ASSEMBLY_MANIFEST.json", manifest)

    caption_source = build_root / "figure_rebuild/CAPTION_NOTES_ROUND20.md"
    (RECORD / "CAPTION_NOTES.md").write_text(
        caption_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (RECORD / "MANIFEST.md").write_text(
        f"# {TARGET_NAME}\n\n"
        f"Supersedes `{PARENT_NAME}`. Panels b-d now show the 36 saved session-pair "
        "correlations directly and weight those pairs equally. Means and 95% confidence "
        "intervals use CR1 standard errors clustered by the 14 animals; planned d1-d1 "
        "contrasts use two-sided exact mouse-cluster sign-flip tests with BH3 correction "
        "within each metric. Panel a, panel e, cohort, pair-level correlations, map "
        "estimators, and layout are unchanged. The ED5 ensemble-rate source retains its "
        "existing common-neuron-support gate.\n",
        encoding="utf-8",
    )
    (RECORD / "REBUILD_README.md").write_text(
        f"# Rebuild {TARGET_NAME}\n\n"
        "From the repository root, run "
        "`python -B docs/figure_version_records/extended_data_fig05_2026-09-08_round20_session_pair_means/source_code/build_ed05_session_pair_round20.py --output-dir <fresh-directory>`. "
        "The record-local builder regenerates the b-d session-pair tables from the three "
        "packaged pair-level source tables, verifies exact parity with the frozen derived "
        "tables, stages the frozen rendering code, and writes fresh SVG/PDF/PNG/TIFF "
        "outputs. Accepted figure artwork is not read as a scientific input. The public "
        "current-record adapter provides the equivalent check-and-build entry point after "
        "pointer promotion.\n",
        encoding="utf-8",
    )
    (RECORD / "PROMOTION_COMPLETE.txt").write_text(
        f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n",
        encoding="utf-8",
    )
    write_hashes(RECORD)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
