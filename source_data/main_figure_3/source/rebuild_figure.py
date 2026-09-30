"""Change only Figure 3d's label to Cycle, preserving the latest PV wording."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "output/figures/figure3_panel_d_cycle_2026-09-27"
PREVIOUS = ROOT / "output/figures/pv_terminology_2026-09-27"
RECORD = ROOT / "docs/figure_version_records/main_figure3_2026-09-25_round13_population_vector_first"
OLD_LINE = 'ax.text(0.5, 0.10, "No. of cycles", ha="center", va="top", fontsize=label_size, color=muted)\n    return fig'
NEW_LINE = OLD_LINE.replace('"No. of cycles"', '"Cycle"')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(bundle):
    if (bundle / "SOURCE_INPUTS.json").exists():
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    driver_path = ROOT / "docs/figure_version_records/rebuild_current_main_figures.py"
    driver = load(driver_path, "figure3_cycle_prepare")
    frozen = bundle / "frozen_source"
    input_rows = []
    for source in driver._contract_paths(RECORD, driver.CONTRACTS["main_figure_3"]):
        target = frozen / source.relative_to(RECORD)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        input_rows.append({"path":target.relative_to(bundle).as_posix(), "sha256":sha(target)})
    source = PREVIOUS / "source_adapters/main_figure_3/render_main_figs_v3_publication_8054aa9ff2ec.py"
    text = source.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "8054aa9ff2ec98551ecd047c7063cc11f8569323ef45cabee93ba5136f06447e"
    assert text.count(OLD_LINE) == 1
    renderer = frozen / "source_code/render_main_figs_v3_publication.py"
    renderer.write_text(text.replace(OLD_LINE, NEW_LINE), encoding="utf-8")
    for row in input_rows:
        row["sha256"] = sha(bundle / row["path"])
    adapter = bundle / "source_code"
    adapter.mkdir()
    for source in (driver_path, driver_path.with_name("SOURCE_REBUILD_MAIN_RUNTIME.zip")):
        target = adapter / source.name
        shutil.copy2(source, target)
        input_rows.append({"path":target.relative_to(bundle).as_posix(), "sha256":sha(target)})
    shutil.copy2(Path(__file__), bundle / "rebuild_figure.py")
    (bundle / "SOURCE_INPUTS.json").write_text(json.dumps({"files":input_rows}, indent=2)+"\n",encoding="utf-8")
    (bundle / "REBUILD_README.md").write_text(
        "# Figure 3d Cycle label\n\n"
        "Rebuild with `python -B rebuild_figure.py --bundle . --output <fresh-directory>`. "
        "Frozen sources preserve the current PV terminology; only panel d's No. of cycles label becomes Cycle. "
        "No artwork is used as a rendering input. Accepted records and scientific data are unchanged.\n",
        encoding="utf-8")


def build(bundle, output):
    for row in json.loads((bundle / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))["files"]:
        assert sha(bundle / row["path"]) == row["sha256"], row["path"]
    driver = load(bundle / "source_code/rebuild_current_main_figures.py", "figure3_cycle_driver")
    driver.REPO_ROOT = bundle
    driver._render_main3(bundle / "frozen_source", output, {})
    import fitz
    for suffix in (".svg", ".pdf"):
        shutil.copy2(output / ("main_figure_3_source_rebuild"+suffix), output / ("figure3_cycle_label"+suffix))
    for suffix in (".png", ".tiff"):
        shutil.copy2(output / ("main_figure_3_source_rebuild_600dpi"+suffix), output / ("figure3_cycle_label_600dpi"+suffix))
    with fitz.open(output / "figure3_cycle_label.pdf") as document:
        document[0].get_pixmap(dpi=300, alpha=False).save(output / "figure3_cycle_label_300dpi.png")


def canonical_svg(path, change_label=False):
    root = ET.parse(path).getroot()
    if change_label:
        panel = next(el for el in root.iter() if el.get("id") == "panel_figure3_d")
        labels = [el for el in panel.iter() if el.text == "No. of cycles"]
        assert len(labels) == 1
        labels[0].text = "Cycle"
    for parent in root.iter():
        for child in list(parent):
            if child.tag.endswith("}metadata"):
                parent.remove(child)
    text = ET.tostring(root, encoding="unicode")
    ids = list(dict.fromkeys(re.findall(r'\bid="([^"]+)"', text)))
    mapping = {value:f"ID_{i:06d}" for i,value in enumerate(ids)}
    pattern = re.compile("|".join(re.escape(value) for value in sorted(ids,key=len,reverse=True)))
    return re.sub(r">\s+<", "><", pattern.sub(lambda match:mapping[match.group()], text))


def validate(output):
    previous = PREVIOUS / "artifacts/main_figure_3"
    old = previous / "main_figure_3_source_rebuild.svg"
    new = output / "figure3_cycle_label.svg"
    assert canonical_svg(old, True) == canonical_svg(new), "Unexpected SVG artwork change"
    panels = []
    for source in sorted((previous / "figure3_individual_panels").glob("*.png")):
        if "_panel_d_" in source.name:
            continue
        assert sha(source) == sha(output / "figure3_individual_panels" / source.name), source.name
        panels.append(source.name)
    assert len(panels) == 12
    receipt = {"status":"PASS", "entire_svg_unchanged_except_panel_d_label":True,
               "other_12_panel_pngs_byte_identical":panels, "latest_PV_terminology_preserved":True,
               "source_inputs_verified":True, "previous_desktop_sha256":sha(previous / "main_figure_3_source_rebuild_600dpi.png")}
    (output / "LABEL_ONLY_VALIDATION.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(output),"label":"Cycle"}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--bundle",type=Path,default=DEFAULT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    output=args.output.resolve() if args.output else bundle
    if args.prepare:
        prepare(bundle)
    build(bundle,output)
    if args.prepare:
        validate(output)
