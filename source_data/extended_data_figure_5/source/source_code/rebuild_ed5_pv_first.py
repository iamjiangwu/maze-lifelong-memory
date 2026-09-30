"""Rebuild ED5 with population vector correlation in b and place-map correlation in c.

Place this entrypoint under a record's source_code directory. It uses only that
record's frozen numerical inputs and renderers plus the shared VOR runtime.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RECORD = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    repo = next(root for root in RECORD.parents if (root / "docs/figure_version_records/rebuild_current_extended_data_figures.py").is_file())
    driver_path = repo / "docs/figure_version_records/rebuild_current_extended_data_figures.py"
    spec = importlib.util.spec_from_file_location("ed05_round22_source_adapter", driver_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    reads: set[str] = set()

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        try:
            path = Path(args[0]).resolve()
        except (OSError, ValueError, TypeError):
            return
        if path.is_relative_to(RECORD) and not path.is_relative_to(output) and path.is_file():
            reads.add(path.relative_to(RECORD).as_posix())

    sys.addaudithook(audit)
    output.mkdir(parents=True)
    with adapter.frozen_stage(RECORD) as stage:
        result = adapter.render_ed5(RECORD, output / "figure_rebuild", stage)
    artwork_reads = sorted(path for path in reads if Path(path).suffix.lower() in {".pdf", ".svg", ".png", ".tif", ".tiff"})
    if artwork_reads:
        raise RuntimeError(f"Accepted artwork unexpectedly read: {artwork_reads}")
    payload = {"status": "PASS", "record": RECORD.name, "render": result,
               "accepted_artwork_used_as_scientific_input": False,
               "scientific_data_reestimated": False,
               "record_files_read": sorted(reads), "record_artwork_reads": artwork_reads,
               "display_change": "Swap only the visual slots for b and c: population vectors first, place maps second. Preserve metric drawing order, random jitter, all numerical inputs, and all other panels."}
    (output / "SOURCE_REBUILD_RESULT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "record_files_read"}, indent=2))


if __name__ == "__main__":
    main()


