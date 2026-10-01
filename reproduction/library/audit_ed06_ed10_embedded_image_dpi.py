"""Audit effective embedded-image DPI for current ED6--ED10 figure PDFs.

The audit distinguishes native discrete matrices and legend strips from dense
continuous scientific clouds.  It verifies the current ED9 successor that
source-natively raises the two formerly avoidable continuous clouds to 600 dpi.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


REPO = Path(__file__).resolve().parents[1]
RECORDS = {
    "ED6": REPO
    / "docs/figure_version_records/extended_data_fig06_2026-08-11_round4_arrow_population_vector",
    "ED7": REPO
    / "docs/figure_version_records/extended_data_fig07_2026-08-11_round13_compact_legends",
    "ED8": REPO
    / "docs/figure_version_records/extended_data_fig08_2026-08-11_round13_population_vector_label",
    "ED9": REPO
    / "docs/figure_version_records/extended_data_fig09_2026-08-12_round26_margin_star_density_resolution",
    "ED10": REPO
    / "docs/figure_version_records/extended_data_fig10_2026-08-11_round10_upright_policy_labels",
}
CANDIDATE = (
    REPO
    / "docs/figure_refinement_2026-08-12/"
    "ed09_panel_b_margin_panel_f_star_density_600dpi"
)
OUTPUT = CANDIDATE / "ED6_ED10_EMBEDDED_IMAGE_DPI_AUDIT.json"
PROMOTION_MAPPING = CANDIDATE / "PROMOTION_MAPPING.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def pdf_images(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            for image in page.get_images(full=True):
                xref = int(image[0])
                width_px = int(image[2])
                height_px = int(image[3])
                for rect in page.get_image_rects(xref):
                    dpi_x = width_px / (float(rect.width) / 72.0)
                    dpi_y = height_px / (float(rect.height) / 72.0)
                    rows.append(
                        {
                            "page": page_index,
                            "xref": xref,
                            "pixels": [width_px, height_px],
                            "bbox_pt": [
                                float(rect.x0),
                                float(rect.y0),
                                float(rect.x1),
                                float(rect.y1),
                            ],
                            "effective_dpi": [float(dpi_x), float(dpi_y)],
                            "minimum_effective_dpi": float(min(dpi_x, dpi_y)),
                        }
                    )
    return rows


def current_pdfs(record_dir: Path) -> list[Path]:
    roots = sorted(record_dir.glob("*.pdf"))
    individuals = sorted((record_dir / "individual_panels").glob("*.pdf"))
    return roots + individuals


def classify(figure: str, pdf: Path, row: dict[str, Any]) -> dict[str, Any]:
    width, height = row["pixels"]
    x0, y0, x1, y1 = row["bbox_pt"]
    result = dict(row)
    result["classification"] = "unclassified"
    result["source_native_policy"] = "review_required"

    if figure == "ED8":
        if width == height == 15:
            result.update(
                {
                    "classification": "native_discrete_data_grid",
                    "source_native_policy": "acceptable_native_resolution",
                    "scientific_representation": "15x15 equal-mouse decoding-error matrix",
                    "native_data_shape": [15, 15],
                }
            )
        elif min(width, height) <= 8 and max(width, height) >= 128:
            result.update(
                {
                    "classification": "colorbar_legend_strip",
                    "source_native_policy": "non_scientific_legend_key",
                    "scientific_representation": None,
                }
            )
    elif figure == "ED9":
        if "E1_joint_place_level" in pdf.name:
            result.update(
                {
                    "classification": "dense_continuous_scientific_cloud",
                    "source_native_policy": "requires_at_least_600_effective_dpi",
                    "scientific_representation": "16901-cell place-map by level-map correlation cloud",
                    "display_panel": "g",
                }
            )
        elif "G1_consecutive_gap" in pdf.name:
            result.update(
                {
                    "classification": "dense_continuous_scientific_cloud",
                    "source_native_policy": "requires_at_least_600_effective_dpi",
                    "scientific_representation": "9661-point consecutive-gap level-map correlation cloud",
                    "display_panel": "j",
                }
            )
        elif "C1_transition_topology" in pdf.name or (y0 < 30 and x0 < 150):
            if min(width, height) <= 5:
                result.update(
                    {
                        "classification": "colorbar_legend_strip",
                        "source_native_policy": "non_scientific_legend_key",
                        "scientific_representation": None,
                    }
                )
            else:
                result.update(
                    {
                        "classification": "native_discrete_data_grid",
                        "source_native_policy": "acceptable_native_resolution",
                        "scientific_representation": "7x7 level-transition probability-difference matrix",
                        "native_data_shape": [7, 7],
                        "display_interpolation": "nearest",
                    }
                )
        elif 260 <= y0 <= 280:
            result.update(
                {
                    "classification": "dense_continuous_scientific_cloud",
                    "source_native_policy": "requires_at_least_600_effective_dpi",
                    "scientific_representation": "16901-cell place-map by level-map correlation cloud",
                    "display_panel": "g",
                }
            )
        elif 370 <= y0 <= 385:
            result.update(
                {
                    "classification": "dense_continuous_scientific_cloud",
                    "source_native_policy": "requires_at_least_600_effective_dpi",
                    "scientific_representation": "9661-point consecutive-gap level-map correlation cloud",
                    "display_panel": "j",
                }
            )
    result["policy_pass"] = (
        result["source_native_policy"]
        in {"acceptable_native_resolution", "non_scientific_legend_key"}
        or (
            result["source_native_policy"]
            == "requires_at_least_600_effective_dpi"
            and result["minimum_effective_dpi"] >= 599.0
        )
    )
    return result


def scan_figure(figure: str, paths: list[Path]) -> dict[str, Any]:
    files = []
    for path in paths:
        images = [classify(figure, path, row) for row in pdf_images(path)]
        files.append({"pdf": record(path), "embedded_images": images})
    all_images = [image for item in files for image in item["embedded_images"]]
    return {
        "files": files,
        "embedded_image_count": len(all_images),
        "classification_counts": {
            name: sum(image["classification"] == name for image in all_images)
            for name in sorted({image["classification"] for image in all_images})
        },
        "all_images_classified": all(
            image["classification"] != "unclassified" for image in all_images
        ),
        "policy_pass": all(image["policy_pass"] for image in all_images),
    }


def main() -> int:
    current = {
        figure: scan_figure(figure, current_pdfs(record_dir))
        for figure, record_dir in RECORDS.items()
    }
    candidate_paths = [
        CANDIDATE
        / "extended_data_figure9_panel_b_margin_panel_f_star_density_600dpi_candidate.pdf",
        *sorted((CANDIDATE / "individual_panels").glob("*.pdf")),
    ]
    successor = scan_figure("ED9", candidate_paths)
    current_ed9_continuous = [
        image
        for item in current["ED9"]["files"]
        for image in item["embedded_images"]
        if image["classification"] == "dense_continuous_scientific_cloud"
    ]
    candidate_ed9_continuous = [
        image
        for item in successor["files"]
        for image in item["embedded_images"]
        if image["classification"] == "dense_continuous_scientific_cloud"
    ]
    checks = {
        "ed6_master_and_individuals_have_no_embedded_images": current["ED6"][
            "embedded_image_count"
        ]
        == 0,
        "ed7_master_and_individuals_have_no_embedded_images": current["ED7"][
            "embedded_image_count"
        ]
        == 0,
        "ed8_images_are_native_grid_or_legend_only": current["ED8"][
            "all_images_classified"
        ]
        and current["ED8"]["policy_pass"],
        "ed9_current_continuous_clouds_are_600dpi_in_master_and_individuals": len(
            current_ed9_continuous
        )
        == 4
        and all(row["minimum_effective_dpi"] >= 599.0 for row in current_ed9_continuous),
        "ed9_candidate_continuous_clouds_are_600dpi_in_master_and_individuals": len(
            candidate_ed9_continuous
        )
        == 4
        and all(row["minimum_effective_dpi"] >= 599.0 for row in candidate_ed9_continuous),
        "ed9_candidate_all_images_classified_and_policy_pass": successor[
            "all_images_classified"
        ]
        and successor["policy_pass"],
        "ed10_master_and_individuals_have_no_embedded_images": current["ED10"][
            "embedded_image_count"
        ]
        == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema": "ed06-ed10-embedded-image-effective-dpi-audit/v1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Pointer-selected current ED6-ED10 master and individual-panel PDFs plus the source-native ED9 candidate retained as build evidence.",
        "classification_policy": {
            "native_discrete_data_grid": "Accept at native matrix resolution when source cells are discrete and nearest-neighbor displayed.",
            "colorbar_legend_strip": "Record but do not treat as scientific raster content.",
            "dense_continuous_scientific_cloud": "Require >=599 measured effective dpi in master and individual PDF.",
        },
        "current_records": {
            figure: record(record_dir / "RECORD_HASHES.csv")
            for figure, record_dir in RECORDS.items()
        },
        "current": current,
        "ed9_candidate": successor,
        "before_after": {
            "current_continuous_minimum_dpi": [
                row["minimum_effective_dpi"] for row in current_ed9_continuous
            ],
            "candidate_continuous_minimum_dpi": [
                row["minimum_effective_dpi"] for row in candidate_ed9_continuous
            ],
        },
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    candidate_manifest_path = CANDIDATE / "CANDIDATE_MANIFEST.json"
    focused_validation_path = CANDIDATE / "FOCUSED_VALIDATION.json"
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    mapping = {
        "schema": "ed09-density600-display-repair-promotion-mapping/v1",
        "status": status,
        "promoted": False,
        "parent_record": RECORDS["ED9"].relative_to(REPO).as_posix(),
        "candidate_root": CANDIDATE.relative_to(REPO).as_posix(),
        "evidence": {
            "candidate_manifest": record(candidate_manifest_path),
            "focused_validation": record(focused_validation_path),
            "ed6_ed10_effective_dpi_audit": record(OUTPUT),
            "renderer": record(
                REPO / "scripts/render_ed09_panel_b_margin_panel_f_star_candidate.py"
            ),
            "validator": record(
                REPO / "scripts/validate_ed09_panel_b_margin_panel_f_star_candidate.py"
            ),
            "dpi_auditor": record(Path(__file__)),
        },
        "root_file_mapping": {
            candidate_manifest["outputs"]["svg"]["path"]: "ed09_version_record.svg",
            candidate_manifest["outputs"]["pdf"]["path"]: "ed09_version_record.pdf",
            candidate_manifest["outputs"]["png"]["path"]: "ed09_version_record_600dpi.png",
            candidate_manifest["outputs"]["tiff"]["path"]: "ed09_version_record_600dpi.tiff",
            candidate_manifest["outputs"]["preview"]["path"]: "ed09_version_record_300dpi.png",
        },
        "individual_panel_mapping": {
            row["path"]: f"individual_panels/{Path(row['path']).name}"
            for role in ("C2", "B2", "E1", "G1")
            for row in candidate_manifest["changed_panel_outputs"][role].values()
        },
        "unchanged_individual_panels": (
            "Copy byte-exact from parent; candidate manifest other_panel_parity must remain PASS."
        ),
        "required_promotion_gates": [
            "CANDIDATE_MANIFEST.json status=PASS",
            "FOCUSED_VALIDATION.json status=PASS",
            "ED6_ED10_EMBEDDED_IMAGE_DPI_AUDIT.json status=PASS",
            "new immutable successor record only; then pointer/archive/hash closure",
        ],
    }
    PROMOTION_MAPPING.write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
