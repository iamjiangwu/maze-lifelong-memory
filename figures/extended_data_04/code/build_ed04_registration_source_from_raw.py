"""Build the Extended Data Figure 4 registration-quality source tables from raw CellReg.

Extended Data Figure 4 (slide 5 under the merged numbering; sources under the old
``extended_data_fig_05``) is the neuron-matching quality-control figure. Everything it
shows comes from the CellReg outputs themselves:

* ``modeled_data_struct.mat``   -> candidate features (centroid distance, footprint
                                   correlation) for every directed query/candidate pair
                                   inside the run-specific search radius;
* ``cellRegistered*.mat``       -> the accepted registration and its per-cell scores;
* ``cell_to_index_map_*.mat``   -> the exported cell-to-index map.

The maintained builder ``build_figure2_registration_quality_panels.py`` additionally
loads each pair's saved population-vector similarity pickle. That is a **Main Figure 2
scope cross-check** -- it re-derives the activity gate and asserts the 21,412 -> 16,901
denominator -- and is unrelated to matching quality. It currently fails for animal 842
0->3 because the corrected zero-reinserted CIMap on the footprint drive no longer agrees
with the June-11 saved similarity artefacts, and repointing that row instead trips the
denominator audit (``saved_final=16897`` versus ``corrected=16901``, the four manual
drops).

This script therefore drives only the raw registration path and skips the similarity
cross-check, so the Extended Data Figure 4 source data can be regenerated from raw
without depending on Figure 2 bookkeeping. Panels are generated separately from the
tables written here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOOTPRINT_ROOT = Path(r"F:\footprint-new")
DEFAULT_SCOPE_CSV = (
    REPO_ROOT
    / "docs"
    / "main_figs_v3_source_render_2026-06-24"
    / "figure_scope_exclusions_2026-06-29"
    / "figure2_distribution_shuffle_36_no_1616_short"
    / "current_panel_base_pairs_36_no_1616_short.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed04_registration_quality"
    / "source_from_raw"
)


def load_builder() -> Any:
    """Import the maintained builder for its raw CellReg readers."""
    path = REPO_ROOT / "scripts" / "build_figure2_registration_quality_panels.py"
    spec = importlib.util.spec_from_file_location("registration_quality_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(scope_csv: Path, footprint_root: Path, output_dir: Path) -> dict[str, Any]:
    builder = load_builder()
    scope = pd.read_csv(scope_csv, dtype={"animal": str})
    if scope.shape[0] != 36 or scope["animal"].nunique() != 14:
        raise ValueError(
            f"Expected the 36-pair / 14-animal scope; got {scope.shape[0]} pairs "
            f"and {scope['animal'].nunique()} animals."
        )
    if ("1616", "1616day1", "1616bday1") in list(
        scope[["animal", "src_session", "dst_session"]].itertuples(index=False, name=None)
    ):
        raise ValueError("The short 1616 pair must not appear in the scope.")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_frames: list[pd.DataFrame] = []
    registered_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []

    for index, row in scope.reset_index(drop=True).iterrows():
        animal = str(row["animal"])
        src_session = str(row["src_session"])
        dst_session = str(row["dst_session"])
        pair_label = f"{src_session}->{dst_session}"
        pair_code = builder._pair_code(row)
        animal_dir = footprint_root / animal
        cellreg_dir = animal_dir / f"cellreg_{pair_code}"
        modeled_path = cellreg_dir / "modeled_data_struct.mat"
        cellregistered_path, n_cellregistered = builder._latest_cellregistered(cellreg_dir)
        exported_cimap_path = animal_dir / f"cell_to_index_map_{pair_code}.mat"

        candidate_df, candidate_lookup, candidate_audit = builder._extract_candidate_features(
            modeled_path,
            animal=animal,
            pair_label=pair_label,
            pair_code=pair_code,
            src_session=src_session,
            dst_session=dst_session,
        )
        with h5py.File(cellregistered_path, "r") as mat_file:
            cleaned_cimap = np.asarray(
                mat_file["cell_registered_struct"]["cell_to_index_map"], dtype=float
            ).T.astype(int)
        exported_cimap = builder._load_exported_cimap(exported_cimap_path)
        (
            corrected_cimap,
            cimap_match_status,
            removed_counts,
        ) = builder._validate_cellreg_to_exported_cimap(
            cleaned_cimap,
            exported_cimap=exported_cimap,
            animal_dir=animal_dir,
            pair_code=pair_code,
        )
        registered_df, symmetry_deltas = builder._extract_registered_features(
            cellregistered_path,
            corrected_cimap=corrected_cimap,
            source_exported_cimap=exported_cimap,
            candidate_lookup=candidate_lookup,
            microns_per_pixel=float(candidate_audit["microns_per_pixel"]),
            animal=animal,
            pair_label=pair_label,
            pair_code=pair_code,
            src_session=src_session,
            dst_session=dst_session,
        )
        candidate_frames.append(candidate_df)
        registered_frames.append(registered_df)
        audits.append(
            {
                "pair_index": int(index),
                "animal": animal,
                "pair_label": pair_label,
                "pair_code": pair_code,
                "cellregistered_path": str(cellregistered_path),
                "exported_cimap_path": str(exported_cimap_path),
                "cimap_match_status": cimap_match_status,
                "n_cellregistered_candidates": int(n_cellregistered),
                "n_directed_candidates": int(candidate_df.shape[0]),
                "n_registered_pairs": int(registered_df.shape[0]),
                "microns_per_pixel": float(candidate_audit["microns_per_pixel"]),
                "search_radius_um": float(
                    candidate_audit.get("search_radius_um", np.nan)
                ),
            }
        )
        print(
            f"[{index + 1}/36] {pair_label}: registered={registered_df.shape[0]}, "
            f"candidates={candidate_df.shape[0]}, cimap={cimap_match_status}",
            flush=True,
        )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    registered = pd.concat(registered_frames, ignore_index=True)
    audit = pd.DataFrame(audits)

    candidates.to_csv(output_dir / "ed04_candidate_features_directed_long.csv", index=False)
    registered.to_csv(output_dir / "ed04_registered_pair_features_long.csv", index=False)
    audit.to_csv(output_dir / "ed04_pair_source_audit.csv", index=False)

    summary = {
        "figure": "Extended Data Figure 4 (merged numbering; slide 5)",
        "sources_resolve_under": "extended_data_fig_05 (old numbering)",
        "generated_from": "raw CellReg outputs under the footprint root",
        "footprint_root": str(footprint_root),
        "scope_csv": str(scope_csv.relative_to(REPO_ROOT)),
        "scope": "final sorted 36-pair scaffold, short 1616 pair excluded",
        "n_session_pairs": int(audit.shape[0]),
        "n_animals": int(scope["animal"].nunique()),
        "n_directed_candidate_pairs": int(candidates.shape[0]),
        "n_registered_pairs_before_activity_gate": int(registered.shape[0]),
        "cimap_match_status_counts": audit["cimap_match_status"].value_counts().to_dict(),
        "registered_pairs_per_session_pair": {
            "min": int(audit["n_registered_pairs"].min()),
            "median": int(audit["n_registered_pairs"].median()),
            "max": int(audit["n_registered_pairs"].max()),
        },
        "directed_candidates_per_session_pair": {
            "min": int(audit["n_directed_candidates"].min()),
            "median": int(audit["n_directed_candidates"].median()),
            "max": int(audit["n_directed_candidates"].max()),
        },
        "excluded_by_design": (
            "the Main Figure 2 similarity cross-check and activity gate are not applied "
            "here; this figure is about neuron matching, and the gate belongs to the "
            "Figure 2 denominator, which the Methods describe"
        ),
        "pooling_note": (
            "distributions pool directed technical observations; each registered cell "
            "contributes about 5.6 candidate pairings inside the search radius, so "
            "session pairs with more cells contribute more counts"
        ),
    }
    (output_dir / "ED04_SOURCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--footprint-root", type=Path, default=DEFAULT_FOOTPRINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build(args.scope_csv, args.footprint_root, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
