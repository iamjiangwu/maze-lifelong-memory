"""Prepare and render ED5's approved current-neuron gallery from cached sources.

The successor uses a normalized-map numeric pack, not accepted figure artwork.
Rebuilds use the version-record adapter and frozen rendering code. This script
does not change current pointers, archive records, or export to the Desktop.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
PARENT_NAME = "extended_data_fig05_2026-09-05_round18_15x15_labels"
TARGET_NAME = "extended_data_fig05_2026-09-07_round19_current_gallery"


def load_adapter():
    path = RECORDS / "rebuild_current_extended_data_figures.py"
    spec = importlib.util.spec_from_file_location("ed5_round19_source_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(candidate: Path) -> Path:
    parent, target = RECORDS / PARENT_NAME, RECORDS / TARGET_NAME
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    validation = json.loads((candidate / "CANDIDATE_VALIDATION.json").read_text())
    for key in ("all_S_activity_gates_pass", "all_32_pairs_present_in_current_CIMaps",
                "unchanged_31_pair_display_pixels_exact"):
        if validation[key] is not True:
            raise ValueError(f"Candidate failed {key}")
    selected = pd.read_csv(candidate / "selected_32.csv", float_precision="round_trip")
    with np.load(candidate / "selected_maps_fields_45_shortlist32.npz", allow_pickle=False) as pack:
        assert str(pack["schema_version"]) == "current_sorted_normalized_gallery_v1"
        assert np.array_equal(pack["analysis_r15"], selected["place_r_15_analysis"])
        assert np.array_equal(pack["source_neuron_zero_based"], selected["train_id"])
        assert np.array_equal(pack["target_neuron_zero_based"], selected["test_id"])
    target.mkdir()
    for name in ("source_code", "individual_panels"):
        shutil.copytree(parent / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plot = target / "plot_ready"
    plot.mkdir()
    replaced = {"selected_32.csv", "selected_display_fields.csv", "selected_maps_fields_45_shortlist32.npz",
                "ED5_PANEL_A_PER_SIDE_PROVENANCE.csv", "ED5_PANEL_A_PROVENANCE_SUMMARY.json",
                "ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv", "FIGURE2_PANEL_B_LABEL_PARITY.csv"}
    for path in (parent / "plot_ready").iterdir():
        if path.is_file() and path.name not in replaced:
            shutil.copy2(path, plot / path.name)
    for name in ("selected_32.csv", "selected_maps_fields_45_shortlist32.npz",
                 "gallery_15x15_conversion_audit.csv"):
        shutil.copy2(candidate / name, plot / name)
    provenance = []
    for row in selected.itertuples(index=False):
        for side, session, neuron in (("source", row.src_session, row.train_id),
                                       ("target", row.dst_session, row.test_id)):
            rate = float(getattr(row, side + "_S_activity_per_minute"))
            canonical = float(getattr(row, side + "_canonical_S_activity_per_minute"))
            provenance.append({
                "visual_rank": int(row.visual_rank), "candidate_id": row.candidate_id,
                "pair_key": row.pair_key, "side": side, "session_id": session,
                "neuron_index_zero_based": int(neuron), "index_basis": row.source_id_space,
                "actual_neural_variant": "sorted", "current_cimap_match_verified": True,
                "S_activity_per_minute": rate, "canonical_S_activity_per_minute": canonical,
                "activity_gate_pass": rate >= 0.1 and canonical >= 0.1,
                "display_source": row.display_source,
            })
    pd.DataFrame(provenance).to_csv(plot / "ED5_PANEL_A_PER_SIDE_PROVENANCE.csv", index=False)
    (plot / "ED5_PANEL_A_PROVENANCE_SUMMARY.json").write_text(json.dumps({
        "status": "qualified", "map_pack_schema": validation["map_pack_schema"],
        "pair_count": len(selected), "side_count": len(provenance),
        "index_basis": "current_sorted_zero_based", "all_current_cimap_matches_verified": True,
        "historical_selection_provenance": PARENT_NAME,
    }, indent=2) + "\n")
    shutil.copy2(parent / "CONTEXT_SAMPLING_CORRECTION_2026_08_20.json",
                 target / "CONTEXT_SAMPLING_CORRECTION_2026_08_20.json")
    for name in ("render_ed05_example_place_maps_assembly.py", Path(__file__).name):
        shutil.copy2(REPO / "scripts" / name, target / "source_code" / name)
    return target


def render(record: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    adapter = load_adapter()
    with adapter.frozen_stage(record) as stage:
        result = adapter.render_ed5(record, output, stage)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    record = prepare(args.candidate_dir.resolve()) if args.candidate_dir else RECORDS / TARGET_NAME
    print(json.dumps(render(record, args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
