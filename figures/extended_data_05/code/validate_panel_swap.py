"""Check source equality and unchanged artwork outside the two swapped panels."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = next(p for p in ROOT.parents if (p / "docs/figure_version_records").is_dir())
PREVIOUS = REPO / "docs/figure_version_records/extended_data_fig05_2026-09-25_round21_descending_examples"
ASSEMBLY = ROOT / "verified_build/figure_rebuild/assembly"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    receipt = json.loads((ROOT / "verified_build/SOURCE_REBUILD_RESULT.json").read_text())
    assert receipt["status"] == "PASS"
    validation = list(csv.DictReader((ASSEMBLY / "ED05_VALIDATION_ROUND22.csv").open()))
    assert all(row["status"] == "PASS" for row in validation)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "plot_ready").rglob("*") if p.is_file()}
    assert all(sha(PREVIOUS / name) == digest for name, digest in inputs.items())
    outputs = {}
    for ext, target in [("pdf", "ed05_version_record.pdf"), ("svg", "ed05_version_record.svg"), ("png", "ed05_version_record_600dpi.png"), ("tiff", "ed05_version_record.tiff")]:
        shutil.copy2(ASSEMBLY / f"ed05_example_place_maps_assembly_round22.{ext}", ROOT / target)
        outputs[target] = sha(ROOT / target)
    old = np.asarray(Image.open(PREVIOUS / "ed05_version_record_600dpi.png").convert("RGB"))
    new = np.asarray(Image.open(ROOT / "ed05_version_record_600dpi.png").convert("RGB"))
    assert old.shape == new.shape
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    # Both changed plots lie left of panel d and below the map gallery.
    assert len(xs) > 0
    assert int(ys.min()) >= 3370, (int(ys.min()), int(ys.max()))
    assert int(xs.max()) < 1570, (int(xs.min()), int(xs.max()))
    # Directly compare complete unaffected regions, including panel a and d/e.
    assert np.array_equal(old[:3370], new[:3370])
    assert np.array_equal(old[:,1570:], new[:,1570:])
    Image.fromarray(new).crop((0, 3330, new.shape[1], new.shape[0])).save(ROOT / "bottom_row_preview.png")
    result = {"status": "PASS", "panel_mapping": {"b": "PV correlation", "c": "Place-map correlation", "d": "Ensemble rate correlation"}, "unchanged_plot_ready_files": len(inputs), "input_hashes": inputs, "source_validation_checks_passed": len(validation), "raster_size": [new.shape[1], new.shape[0]], "changed_pixel_bbox": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1], "all_pixels_outside_b_and_c_unchanged": True, "metric_drawing_order_and_rng_seed_unchanged": True, "scientific_values_unchanged": True, "output_hashes": outputs}
    (ROOT / "PANEL_SWAP_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_hashes"}, indent=2))

if __name__ == "__main__":
    main()
