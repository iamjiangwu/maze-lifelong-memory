"""Freeze honest per-side provenance for the accepted ED5 example gallery.

The 32 examples remain unchanged.  This audit distinguishes the ordinary
unsorted, sorted, and preserved presort ID spaces and records the two cases
that cannot be represented as exact current-sorted IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SELECTED = (
    REPO
    / "docs/extended_data_figure_panel_refresh_2026-07-12/source_rendered_new/"
    "extended_data_fig_06/example_place_map_shortlist_32_clear_fields_2026-08-02/selected_32.csv"
)
BRIDGE = REPO / "docs/ed05_example_id_bridge_2026-08-07/ed05_selected32_unsorted_to_sorted_ids.csv"
POOL = REPO / "docs/pv_variant_place_map_mechanism_expanded_mazeaware_plus_839_842_2026-04-03/pair_manifest.csv"
OUT = REPO / "docs/ed05_example_id_bridge_2026-08-07"
EXTERNAL_CONTEXT = Path(
    "C:/Users/jiangwu/Documents/Codex/2026-07-31/fo/outputs/"
    "ed06_place_map_candidates_100_v2/intermediate_context_maps_v3/"
    "v3__839-9day1_presort__p3e7217ec9b35e80a__ncf0319dc5a4f98cd.manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    selected = pd.read_csv(SELECTED).sort_values("visual_rank", kind="mergesort")
    bridge = pd.read_csv(BRIDGE).sort_values("visual_rank", kind="mergesort")
    merged = selected.merge(
        bridge,
        on=["visual_rank", "animal_id", "pair_key", "src_session", "dst_session"],
        suffixes=("_selected", "_bridge"),
        validate="one_to_one",
    )
    if len(merged) != 32:
        raise RuntimeError(f"Expected 32 selected examples, found {len(merged)}")

    pool = pd.read_csv(POOL)
    train_mask_exposed = pool["exclude_train_ids"].fillna("[]").astype(str).ne("[]")
    test_mask_exposed = pool["exclude_test_ids"].fillna("[]").astype(str).ne("[]")
    pool_mask_exposed = train_mask_exposed | test_mask_exposed
    if (
        len(pool) != 136
        or int(train_mask_exposed.sum()) != 26
        or int(test_mask_exposed.sum()) != 22
        or int(pool_mask_exposed.sum()) != 36
    ):
        raise RuntimeError("The frozen gallery candidate-pool mask exposure changed")

    context = json.loads(EXTERNAL_CONTEXT.read_text(encoding="utf-8"))
    if context.get("session_id") != "839-9day1" or context.get("neural_variant") != "presort":
        raise RuntimeError("The rank-11 external context no longer identifies the presort root")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        rank = int(row.visual_rank)
        for side, session_column, id_column, sorted_column in (
            ("train", "src_session", "train_id_unsorted", "train_id_sorted"),
            ("test", "dst_session", "test_id_unsorted", "test_id_sorted"),
        ):
            session = str(getattr(row, session_column))
            displayed_id = int(getattr(row, id_column))
            mapped = getattr(row, sorted_column)
            variant = "ordinary"
            id_space = "ordinary_unsorted_session_local"
            mapping_status = "exact_footprint_bridge_to_current_sorted"
            if rank == 14:
                variant = "sorted"
                id_space = "current_sorted_session_local"
                mapping_status = "already_sorted_no_bridge_needed"
                mapped = displayed_id
            elif rank == 11 and side == "test":
                variant = "presort"
                id_space = "preserved_presort_third_session_local"
                mapping_status = "unbridgeable_to_current_sorted_from_frozen_repository_evidence"
                mapped = None
            elif rank == 31 and side == "train":
                mapping_status = "ordinary_component_dropped_by_sorting"
                mapped = None
            rows.append(
                {
                    "visual_rank": rank,
                    "candidate_id": str(row.candidate_id),
                    "pair_key": str(row.pair_key),
                    "side": side,
                    "session": session,
                    "displayed_session_local_id": displayed_id,
                    "actual_neural_variant": variant,
                    "source_id_space": id_space,
                    "current_sorted_session_local_id": (
                        "" if pd.isna(mapped) else int(float(mapped))
                    ),
                    "mapping_status": mapping_status,
                    "source_component": str(row.source_component),
                    "source_file": str(row.source_file_selected),
                    "external_context_manifest": (
                        str(EXTERNAL_CONTEXT) if rank == 11 and side == "test" else ""
                    ),
                    "external_context_manifest_sha256": (
                        sha256(EXTERNAL_CONTEXT) if rank == 11 and side == "test" else ""
                    ),
                    "candidate_pool_rows": len(pool),
                    "candidate_pool_rows_with_train_manual_mask_exposure": int(
                        train_mask_exposed.sum()
                    ),
                    "candidate_pool_rows_with_test_manual_mask_exposure": int(
                        test_mask_exposed.sum()
                    ),
                    "candidate_pool_rows_with_any_manual_mask_exposure": int(
                        pool_mask_exposed.sum()
                    ),
                    "candidate_pool_exclusion_profile": (
                        "sparsity_reviewed_plus_1616_1224_1668_bidirectional"
                    ),
                    "displayed_cell_is_known_masked_id": False,
                    "panel_status": "illustrative_mixed_provenance_qualified",
                }
            )

    output = pd.DataFrame(rows)
    if len(output) != 64:
        raise RuntimeError("Per-side provenance must contain 64 rows")
    counts = output["actual_neural_variant"].value_counts().to_dict()
    if counts != {"ordinary": 61, "sorted": 2, "presort": 1}:
        raise RuntimeError(f"Unexpected gallery provenance counts: {counts}")
    if int(output["current_sorted_session_local_id"].astype(str).eq("").sum()) != 2:
        raise RuntimeError("Expected exactly two unrepresented current-sorted sides")

    OUT.mkdir(parents=True, exist_ok=True)
    provenance_path = OUT / "ED5_PANEL_A_PER_SIDE_PROVENANCE.csv"
    output.to_csv(provenance_path, index=False)
    summary = {
        "status": "qualified",
        "panel": "ED5a",
        "displayed_examples": 32,
        "per_side_rows": 64,
        "actual_neural_variant_counts": counts,
        "candidate_pool_rows": len(pool),
        "candidate_pool_rows_with_train_manual_mask_exposure": int(train_mask_exposed.sum()),
        "candidate_pool_rows_with_test_manual_mask_exposure": int(test_mask_exposed.sum()),
        "candidate_pool_rows_with_any_manual_mask_exposure": int(pool_mask_exposed.sum()),
        "displayed_known_masked_ids": 0,
        "exact_current_sorted_sides": 62,
        "unrepresented_current_sorted_sides": [
            "rank 11 test: preserved presort third ID space",
            "rank 31 train: ordinary component dropped by sorting",
        ],
        "display_decision": (
            "Examples retained as instructed; displayed maps and r values are accepted, "
            "but the panel is not described as sorted-current or selection-mask-free."
        ),
        "provenance_csv": str(provenance_path.relative_to(REPO)),
        "provenance_sha256": sha256(provenance_path),
    }
    (OUT / "ED5_PANEL_A_PROVENANCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
