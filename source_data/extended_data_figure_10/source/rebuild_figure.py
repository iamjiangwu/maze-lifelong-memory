"""Source-only ED10 absolute-correlation display; replay with --bundle and --output."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/figures/ed10_absolute_heatmap_2026-09-27"
RECORD_NAME = "extended_data_fig10_2026-09-25_round18_smaller_ns"
SOURCE_FILES = [
    "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv",
    "plot_ready/REWARD_LEVEL_COMMONALITY.csv",
    "plot_ready/REWARD_LEVEL_PROFILE_TESTS.csv",
    "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv",
    "plot_ready/level_policy_bridge_join_corrected36.csv",
    "plot_ready/level_policy_bridge_stats_corrected36.csv",
    "source_authority/fixed_neuron_synthesis/REPORT.md",
    "source_authority/fixed_neuron_synthesis/ED10_CAPTION_AND_SOURCE_GUIDE.md",
    "source_authority/fixed_neuron_synthesis/NEURAL_ELIGIBILITY_CONTRACT.json",
    "source_authority/EXCLUDED_SESSION_CONTRACT.json",
    "main_figure_authority/figure5_source_manifest.json",
    "source_code/preview_figure3_assembly.py",
    "source_code/_ed10_neural_eligibility.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(bundle):
    record = ROOT / "docs/figure_version_records" / RECORD_NAME
    frozen = bundle / "frozen_source/extended_data_fig10_round18_inputs"
    if frozen.exists():
        raise FileExistsError(frozen)
    rows = []
    for relative in SOURCE_FILES:
        source, target = record / relative, frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        rows.append({"path": target.relative_to(bundle).as_posix(), "sha256": sha(target),
                     "authority": source.relative_to(ROOT).as_posix()})
    renderer = ROOT / "scripts/render_ed10_slide12_powerpoint_matched.py"
    target = frozen / "source_code" / renderer.name
    shutil.copy2(renderer, target)
    rows.append({"path": target.relative_to(bundle).as_posix(), "sha256": sha(target),
                 "authority": renderer.relative_to(ROOT).as_posix()})
    adapter = bundle / "source_code"
    adapter.mkdir(exist_ok=True)
    for name in ("rebuild_current_extended_data_figures.py", "SOURCE_REBUILD_ED_RUNTIME.zip",
                 "SOURCE_REBUILD_ED_CONTRACT.json"):
        source = ROOT / "docs/figure_version_records" / name
        target = adapter / name
        shutil.copy2(source, target)
        rows.append({"path": target.relative_to(bundle).as_posix(), "sha256": sha(target),
                     "authority": source.relative_to(ROOT).as_posix()})
    shutil.copy2(Path(__file__), bundle / "rebuild_figure.py")
    write_json(bundle / "SOURCE_INPUTS.json", {"status": "PASS", "files": rows})
    (bundle / "REBUILD_README.md").write_text(
        "# ED10 absolute-correlation review\n\n"
        "Run `python -B rebuild_figure.py --bundle . --output <new-empty-directory>`.\n\n"
        "All required tables, renderer, adapter and Python support modules are frozen here. "
        "No accepted figure image is a rendering input. Panel a displays abs(Pearson r) "
        "in both colors and labels. Signed coefficients and original CR1/BH3 tests are retained; "
        "panels b/c and figure geometry are unchanged.\n\n"
        "The coefficient is pooled across 30 session pairs from 14 mice; stars use two-sided "
        "pair-row OLS with mouse-clustered CR1 and row-wise BH3. Panel c retains its separate "
        "36-pair cohort. This display change does not promote an accepted record or edit manuscript text.\n",
        encoding="utf-8",
    )


def build(bundle, output):
    inputs = json.loads((bundle / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))
    for row in inputs["files"]:
        assert sha(bundle / row["path"]) == row["sha256"], row["path"]
    adapter = bundle / "source_code/rebuild_current_extended_data_figures.py"
    spec = importlib.util.spec_from_file_location("ed10_absolute_adapter", adapter)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    driver.REPO_ROOT = bundle
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runtime_authorities").exists():
        raise FileExistsError("Use a fresh output directory")
    frozen = bundle / "frozen_source/extended_data_fig10_round18_inputs"
    with driver.frozen_stage(frozen) as stage:
        result = driver.render_ed10(frozen, output, stage)
    assert result["status"] == "PASS"
    stem = output / "assembly/round18/extended_data_figure10_source_rerender_round18"
    for suffix in ("svg", "pdf"):
        shutil.copy2(stem.with_suffix("." + suffix), output / ("ed10_absolute_heatmap." + suffix))
    for suffix in ("png", "tiff"):
        shutil.copy2(stem.with_suffix("." + suffix), output / ("ed10_absolute_heatmap_600dpi." + suffix))
    with fitz.open(stem.with_suffix(".pdf")) as document:
        document[0].get_pixmap(dpi=300, alpha=False).save(output / "ed10_absolute_heatmap_300dpi.png")
    display = pd.read_csv(output / "source_data/ED10_PANEL_A_ABSOLUTE_DISPLAY.csv", keep_default_na=False)
    assert len(display) == 18
    assert (display.absolute_pearson_r == display.pooled_pearson_r.abs()).all()
    assert display.stars.ne("").sum() == 12
    assert display.absolute_pearson_r.between(0, .8).all()
    write_json(output / "DISPLAY_VALIDATION.json", {
        "status": "PASS", "cell_count": 18, "all_values_exact_absolute_correlations": True,
        "star_count": 12, "scale": [0, .8], "label": "Absolute Pearson correlation",
        "renderer_audit_status": result["status"], "all_frozen_input_hashes_verified": True,
    })
    print(json.dumps({"status": "PASS", "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--bundle", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.prepare:
        bundle.mkdir(parents=True, exist_ok=True)
        prepare(bundle)
    build(bundle, args.output.resolve() if args.output else bundle)


if __name__ == "__main__":
    main()
