"""Build a native Main Figure 5 candidate with the accepted historical row order.

The immutable 2026-08-08 full-audit record is the display authority.  Its panel b
used average-linkage clustering of the combined map-shape, centre-of-mass and
cross-session-similarity distance, followed by the established stable argmax
refinement.  This script computes that order once from the hash-locked 36-pair
plot-ready table, caches it, and reuses the cache for native matplotlib renders.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


PARENT_RECORD = (
    REPO_ROOT
    / "docs"
    / "figure_version_records_archive_2026-08-09"
    / "main_figure5_2026-08-08_full_audit"
)
PLOT_READY = PARENT_RECORD / "source_data" / "level_vectors_tree_depth7_corrected36_d1v1.csv"
PLOT_READY_SHA256 = "7dde368f1d87d4355c74565948eda5b19f93610c639d750400abf5014ca55dff"
SUPERSEDED_PREFERRED_ORDER = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure5_2026-08-11_round5_rebuild_provenance_closure"
    / "source_validation"
    / "refinement_round2"
    / "figure5b_preferred_level_similarity_refined_order.npy"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "docs"
    / "figure5b_historical_order_correction_2026-08-12"
)

PAIR_GROUPS = {
    "once": [
        ("418RRday1", "418-3day1"),
        ("422RRday1", "422-3day1"),
        ("434RRday1", "434-3day1"),
        ("451Rday1", "451-3day1"),
        ("468Lday1", "468-3day1"),
        ("488Rday1", "488-3day1"),
        ("489Lday1", "489-3day1"),
        ("579RLday1", "579-3day1"),
        ("837day1", "837-3day1"),
        ("838day1", "838-3day1"),
        ("839day1", "839-3day1"),
        ("842day1", "842-3day1"),
        ("862day1", "862-3day1"),
    ],
    "twice": [
        ("1616bday1", "1616b-3day1"),
        ("418-3day1", "418-6day1"),
        ("422-3day1", "422-6day1"),
        ("451-3day1", "451-6day1"),
        ("468-3day1", "468-6day1"),
        ("488-3day1", "488-6day1"),
        ("489-3day1", "489-6day1"),
        ("837-3day1", "837-6day1"),
        ("838-3day1", "838-6day1"),
        ("839-3day1", "839-6day1"),
        ("842-3day1", "842-6day1"),
        ("862-3day1", "862-6day1"),
    ],
    "three times": [
        ("418-6day1", "418-9day1"),
        ("451-6day1", "451-9day1"),
        ("468-6day1", "468-9day1"),
        ("837-6day1", "837-9day1"),
        ("838-6day1", "838-9day1"),
        ("839-6day1", "839-9day1"),
        ("842-6day1", "842-9day1"),
    ],
    "four times": [
        ("468-9day1", "468-12day1"),
        ("837-9day1", "837-12day1"),
        ("838-9day1", "838-12day1"),
        ("839-9day1", "839-12day1"),
    ],
}
PAIR_ORDER = [pair for pairs in PAIR_GROUPS.values() for pair in pairs]


def _load_maintained_order_functions():
    """Load only the two maintained functions, avoiding jw_utils import side effects."""

    source_path = REPO_ROOT / "src" / "ca_analysis" / "jw_utils.py"
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"order_hier_shape_index", "refine_order_by_argmax"}
    ]
    if {node.name for node in selected} != {
        "order_hier_shape_index",
        "refine_order_by_argmax",
    }:
        raise RuntimeError("Maintained row-order functions were not found.")
    namespace = {
        "np": np,
        "pdist": pdist,
        "linkage": linkage,
        "leaves_list": leaves_list,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["order_hier_shape_index"], namespace["refine_order_by_argmax"]


order_hier_shape_index, refine_order_by_argmax = _load_maintained_order_functions()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _small_n_condensed_equivalence() -> list[dict[str, object]]:
    """Prove the optimized condensed path matches the legacy full matrix."""

    checks = []
    for n_rows in (8, 32, 64):
        rng = np.random.default_rng(20260812 + n_rows)
        train = rng.random((7, n_rows))
        similarity = rng.normal(size=n_rows)
        legacy = order_hier_shape_index(
            train,
            similarity=similarity,
            lam_index=1.0,
            lam_sim=1.0,
            _condensed_threshold=n_rows + 1,
        )
        condensed = order_hier_shape_index(
            train,
            similarity=similarity,
            lam_index=1.0,
            lam_sim=1.0,
            _condensed_threshold=0,
        )
        checks.append(
            {
                "n_rows": n_rows,
                "exact_order_match": bool(np.array_equal(legacy, condensed)),
            }
        )
    return checks


def _ordered_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    digest = _sha256(PLOT_READY)
    if digest != PLOT_READY_SHA256:
        raise RuntimeError(f"Plot-ready hash drift: {digest} != {PLOT_READY_SHA256}")
    rows = pd.read_csv(PLOT_READY)
    ordered = []
    for source, target in PAIR_ORDER:
        pair = rows.loc[
            rows["src_session"].astype(str).eq(source)
            & rows["dst_session"].astype(str).eq(target)
        ]
        if pair.empty:
            raise RuntimeError(f"Missing accepted pair {source}->{target}")
        ordered.append(pair)
    ordered_rows = pd.concat(ordered, ignore_index=True)
    train = ordered_rows[[f"train_level_{level}" for level in range(7)]].to_numpy(float).T
    similarity = pd.to_numeric(ordered_rows["level_corr"], errors="coerce").to_numpy(float)
    if train.shape != (7, 16901) or similarity.shape != (16901,):
        raise RuntimeError(f"Unexpected accepted input shape: train={train.shape}, similarity={similarity.shape}")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(similarity)):
        raise RuntimeError("Accepted hierarchical inputs must be finite.")
    return train, similarity, ordered_rows


def compute_order(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order_path = out_dir / "figure5b_historical_hierarchical_refined_order.npy"
    metadata_path = order_path.with_suffix(".json")
    if order_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _sha256(order_path) != metadata["refined_order_sha256"]:
            raise RuntimeError("Existing historical-order cache hash mismatch.")
        return metadata

    train, similarity, ordered_rows = _ordered_inputs()
    started = time.perf_counter()
    hierarchical = order_hier_shape_index(
        train,
        similarity=similarity,
        lam_index=1.0,
        lam_sim=1.0,
        _condensed_threshold=0,
    )
    norms = np.linalg.norm(train, axis=0)
    normalized = train / np.where(norms > 0, norms, 1.0)
    refined = refine_order_by_argmax(normalized, base_order=hierarchical, eps=1e-6)
    elapsed = time.perf_counter() - started
    expected = np.arange(train.shape[1], dtype=int)
    if not np.array_equal(np.sort(refined), expected):
        raise RuntimeError("Historical refined order is not an exact permutation.")
    np.save(order_path, np.asarray(refined, dtype=np.int64))
    metadata = {
        "schema": "main-figure5b-historical-hierarchical-order/v1",
        "ordering_method": "historical_average_linkage_then_stable_argmax",
        "distance": "1-cosine(map shape) + 1*dCOM^2 + 1*dSimilarity^2",
        "linkage": "average",
        "lam_index": 1.0,
        "lam_similarity": 1.0,
        "refine": "stable grouping by preferred Day-i level",
        "source_record": str(PARENT_RECORD.relative_to(REPO_ROOT)),
        "plot_ready_csv": str(PLOT_READY.relative_to(REPO_ROOT)),
        "plot_ready_sha256": PLOT_READY_SHA256,
        "pair_count": len(PAIR_ORDER),
        "row_count": int(train.shape[1]),
        "pair_key_sequence_sha256": hashlib.sha256(
            "\n".join(ordered_rows["pair_key"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "refined_order_file": str(order_path.relative_to(REPO_ROOT)),
        "refined_order_sha256": _sha256(order_path),
        "is_exact_permutation": True,
        "compute_seconds": elapsed,
    }
    _write_json(metadata_path, metadata)
    return metadata


def _find_direct_child(root: ET.Element, element_id: str) -> ET.Element:
    for child in list(root):
        if child.attrib.get("id") == element_id:
            return child
    raise RuntimeError(f"Missing direct child {element_id!r}")


def _export_bundle(svg_path: Path, *, dpi: int) -> dict[str, str]:
    import render_main_figs_v3_publication as publication

    png_path = svg_path.with_name(f"{svg_path.stem}_{dpi}dpi.png")
    pdf_path = svg_path.with_suffix(".pdf")
    tiff_path = svg_path.with_name(f"{svg_path.stem}_{dpi}dpi.tiff")
    html_path = publication._export_svg_with_chrome(
        svg_path,
        png_path,
        pdf_path,
        width_mm=180.0,
        height_mm=124.23121387283236,
        dpi=dpi,
    )
    with Image.open(png_path) as image:
        image.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    return {
        key: str(path.relative_to(REPO_ROOT))
        for key, path in {
            "svg": svg_path,
            "pdf": pdf_path,
            "png": png_path,
            "tiff": tiff_path,
            "html": html_path,
        }.items()
    }


def _embedded_images(svg_path: Path) -> list[Image.Image]:
    root = ET.parse(svg_path).getroot()
    output = []
    for element in root.iter():
        if not element.tag.endswith("image"):
            continue
        href = next((value for key, value in element.attrib.items() if key.endswith("href")), None)
        if href and href.startswith("data:"):
            output.append(Image.open(io.BytesIO(base64.b64decode(href.split(",", 1)[1]))).convert("RGBA"))
    return output


def build_candidate(out_dir: Path, *, dpi: int) -> dict[str, object]:
    import render_main_figs_v3_publication as publication

    metadata = compute_order(out_dir)
    order_path = REPO_ROOT / metadata["refined_order_file"]
    helper = publication._load_script_module(
        "preview_figure3_assembly_historical_order_candidate",
        publication.FIGURE5_HELPER,
    )
    panel_spec = helper.PANEL_LAYOUT_SPEC["e"]
    figure, data = helper._render_panel_e(
        (float(panel_spec["render_w"]), float(panel_spec["render_h"])),
        refined_order_path=order_path,
        refined_order_sha256=metadata["refined_order_sha256"],
        return_data=True,
    )
    payload = publication._figure_to_svg_payload("figure5_e", figure, dpi=dpi, fixed_bbox=True)
    panel_dir = out_dir / "figure5_individual_panels"
    panel_exports = publication._write_svg_payload_outputs(
        payload,
        panel_dir,
        "figure5_panel_b_level_code_heatmap_historical_order",
        dpi=dpi,
    )

    root = ET.parse(PARENT_RECORD / "figure5_version_record.svg").getroot()
    group = _find_direct_child(root, "panel_e")
    for child in list(group):
        group.remove(child)
    for child in payload["children"]:
        group.append(copy.deepcopy(child))
    svg_path = out_dir / "figure5_historical_order_normalized_rate_candidate.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    outputs = _export_bundle(svg_path, dpi=dpi)

    cached = np.load(order_path, allow_pickle=False)
    parity_figure, parity_data = helper._render_panel_e(
        (float(panel_spec["render_w"]), float(panel_spec["render_h"])),
        refined_order_path=order_path,
        refined_order_sha256=metadata["refined_order_sha256"],
        return_data=True,
    )
    parity_payload = publication._figure_to_svg_payload(
        "figure5_e_parity", parity_figure, dpi=100, fixed_bbox=True
    )
    parity_svg = out_dir / "figure5_panel_b_100dpi_parity.svg"
    parity_root = ET.Element(publication._svg_tag("svg"))
    for child in parity_payload["children"]:
        parity_root.append(copy.deepcopy(child))
    ET.ElementTree(parity_root).write(parity_svg, encoding="utf-8", xml_declaration=True)

    reference_images = _embedded_images(PARENT_RECORD / "figure5_version_record.svg")[:2]
    candidate_images = _embedded_images(parity_svg)[:2]
    comparisons = []
    for reference, candidate in zip(reference_images, candidate_images, strict=True):
        candidate_original_size = candidate.size
        same_size = candidate.size == reference.size
        if not same_size:
            candidate = candidate.resize(reference.size, Image.Resampling.LANCZOS)
        left = np.asarray(reference, dtype=np.int16)
        right = np.asarray(candidate, dtype=np.int16)
        comparisons.append(
            {
                "reference_size": list(reference.size),
                "candidate_original_size": list(candidate_original_size),
                "pixel_exact": bool(np.array_equal(left, right)),
                "mean_absolute_channel_error": float(np.mean(np.abs(left - right))),
                "rgb_correlation": float(np.corrcoef(left[..., :3].ravel(), right[..., :3].ravel())[0, 1]),
            }
        )
    small_n_equivalence = _small_n_condensed_equivalence()
    superseded_preferred = np.load(SUPERSEDED_PREFERRED_ORDER)
    changed_positions = int(np.count_nonzero(cached != superseded_preferred))
    validation = {
        "schema": "main-figure5b-historical-order-candidate-validation/v1",
        "status": "PASS" if (
            np.array_equal(cached, np.asarray(data["refined_order"], dtype=np.int64))
            and np.array_equal(cached, np.asarray(parity_data["refined_order"], dtype=np.int64))
            and all(item["rgb_correlation"] > 0.995 for item in comparisons)
            and all(item["exact_order_match"] for item in small_n_equivalence)
        ) else "FAIL",
        "cache_applied_exactly": bool(np.array_equal(cached, np.asarray(data["refined_order"], dtype=np.int64))),
        "historical_reference": str((PARENT_RECORD / "figure5_version_record.svg").relative_to(REPO_ROOT)),
        "heatmap_resolution_normalized_comparisons": comparisons,
        "small_n_condensed_vs_legacy_equivalence": small_n_equivalence,
        "superseded_preferred_level_similarity_comparison": {
            "cache": str(SUPERSEDED_PREFERRED_ORDER.relative_to(REPO_ROOT)),
            "same_positions": int(cached.size - changed_positions),
            "changed_positions": changed_positions,
            "row_count": int(cached.size),
        },
        "normalized_rate_present": "Normalized rate" in svg_path.read_text(encoding="utf-8"),
        "source_row_count": int(data["n_neurons"]),
        "source_pair_count": len(PAIR_ORDER),
        "plot_ready_sha256": PLOT_READY_SHA256,
    }
    _write_json(out_dir / "VALIDATION.json", validation)
    manifest = {
        "schema": "main-figure5b-historical-order-candidate/v1",
        "status": validation["status"],
        "parent_record": str(PARENT_RECORD.relative_to(REPO_ROOT)),
        "current_record_unchanged": "docs/figure_version_records/main_figure5_2026-08-11_round5_rebuild_provenance_closure",
        "change": "restore accepted historical hierarchical-plus-argmax row order; retain Normalized rate label",
        "order_cache": metadata,
        "outputs": outputs,
        "individual_panel_assets": {
            key: str(value.relative_to(REPO_ROOT)) if isinstance(value, Path) else value
            for key, value in panel_exports.items()
        },
        "validation": str((out_dir / "VALIDATION.json").relative_to(REPO_ROOT)),
        "changed_renderer_files": [
            str(Path(__file__).relative_to(REPO_ROOT)),
            "src/ca_analysis/jw_utils.py",
        ],
        "maintained_order_function": "src/ca_analysis/jw_utils.py::order_hier_shape_index",
    }
    _write_json(out_dir / "CANDIDATE_MANIFEST.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--compute-order-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if args.compute_order_only:
        result = compute_order(out_dir)
    else:
        result = build_candidate(out_dir, dpi=int(args.dpi))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
