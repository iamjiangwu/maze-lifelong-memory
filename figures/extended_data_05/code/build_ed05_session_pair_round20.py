"""Rebuild ED5 round 20 from this version record's packaged source data.

The build regenerates the ED5 b-d session-pair summaries into a fresh audit
directory, verifies exact parity with the frozen plot-ready tables, and then
uses the repository's isolated version-record adapter to render SVG, PDF, PNG,
and TIFF outputs. Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


RECORD = Path(__file__).resolve().parents[1]
VOR_ROOT = RECORD.parent
ADAPTER = VOR_ROOT / "rebuild_current_extended_data_figures.py"
REAGGREGATION = Path(__file__).with_name(
    "recompute_ed05_pair_weighted_day_combinations.py"
)

TABLES = {
    "pair_values": "ed05_day_combination_session_pair_values.csv",
    "descriptive": "ed05_day_combination_session_pair_descriptive_cr1.csv",
    "contrasts": "ed05_day_combination_session_pair_contrasts_cr1_bh3.csv",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rebuild_tables(output: Path) -> dict[str, str]:
    analysis = load_module(REAGGREGATION, "ed05_round20_reaggregation")
    pair_values, source_checks = analysis.load_pair_values(RECORD / "plot_ready")
    descriptive = analysis.summarize(pair_values)
    contrasts = analysis.contrasts(pair_values)
    regenerated = {
        "pair_values": pair_values,
        "descriptive": descriptive,
        "contrasts": contrasts,
        "aggregation_comparison": analysis.aggregation_comparison(
            pair_values, descriptive
        ),
        "source_validation": source_checks,
    }
    output.mkdir(parents=True)
    paths: dict[str, str] = {}
    for role, frame in regenerated.items():
        filename = TABLES.get(role, f"ed05_day_combination_{role}.csv")
        path = output / filename
        frame.to_csv(path, index=False)
        paths[role] = path.name
        if role in TABLES:
            expected = pd.read_csv(RECORD / "plot_ready" / TABLES[role])
            observed = pd.read_csv(path)
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=False,
                rtol=1e-12,
                atol=1e-12,
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    table_paths = rebuild_tables(output / "analysis_rebuild")
    adapter = load_module(ADAPTER, "ed05_round20_source_adapter")
    with adapter.frozen_stage(RECORD) as stage:
        render_result = adapter.render_ed5(RECORD, output / "figure_rebuild", stage)
    payload = {
        "status": "PASS",
        "record": RECORD.name,
        "accepted_artwork_used_as_scientific_input": False,
        "recomputed_tables": table_paths,
        "render": render_result,
    }
    (output / "SOURCE_REBUILD_RESULT.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
