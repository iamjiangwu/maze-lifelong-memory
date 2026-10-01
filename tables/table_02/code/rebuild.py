#!/usr/bin/env python3
"""Rebuild Supplementary Table 2 from the record-local source snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_LEDGER_SHA256 = "b252a26476802f2e5522bec360d45f8fdb278dd425d3f2dd48ee083e51dc6ca4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--renderer-python", default=sys.executable, help="Python with matplotlib and PyMuPDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = record / "source_code"
    ledger = output / "comprehensive_interval_ledger.csv"
    ledger_manifest = output / "comprehensive_interval_ledger_manifest.json"

    run(
        [
            sys.executable,
            str(code / "build_source.py"),
            "--behavior-points",
            "source_data/behavior_return_points.csv",
            "--candidate-registry",
            "source_data/same_stage_candidate_universe_attrition.csv",
            "--experiment-metafile",
            "source_data/experiment_metafile.csv",
            "--calcium-source",
            "source_data/calcium_long_gap_pairs_corrected36.csv",
            "--output-csv",
            str(ledger),
            "--manifest",
            str(ledger_manifest),
        ],
        cwd=record,
    )
    observed_ledger_hash = sha256(ledger)
    if observed_ledger_hash != EXPECTED_LEDGER_SHA256:
        raise RuntimeError(
            f"Ledger hash mismatch: expected {EXPECTED_LEDGER_SHA256}, observed {observed_ledger_hash}"
        )

    run(
        [args.node, str(code / "build_workbook.mjs"), str(output), str(output)],
        cwd=record,
    )
    workbook = output / "Supplementary_Table_2_Long_Gap_Intervals.xlsx"
    run([args.node, str(code / "validate_workbook.mjs"), str(workbook)], cwd=record)
    run(
        [
            args.renderer_python,
            str(code / "render_publication_table.py"),
            "--source-csv",
            str(ledger),
            "--output-dir",
            str(output),
            "--pdf-output",
            str(output / "Supplementary_Table_2_Long_Gap_Intervals.pdf"),
        ],
        cwd=record,
    )

    required = [
        ledger,
        ledger_manifest,
        workbook,
        output / "Supplementary_Table_2_Long_Gap_Intervals.csv",
        output / "Supplementary_Table_2_Long_Gap_Intervals.pdf",
        output / "Supplementary_Table_2_Long_Gap_Intervals_publication.svg",
        output / "Supplementary_Table_2_Long_Gap_Intervals_publication.png",
        output / "Supplementary_Table_2_Long_Gap_Intervals_publication.tiff",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rebuilt outputs: {missing}")
    validation = {
        "status": "pass",
        "source_boundary": "maintained_derived",
        "ledger_rows": 58,
        "ledger_sha256": observed_ledger_hash,
        "outputs": [
            {
                "path": path.relative_to(record).as_posix()
                if path.is_relative_to(record)
                else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
        ],
    }
    (output / "SOURCE_REBUILD_VALIDATION.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

