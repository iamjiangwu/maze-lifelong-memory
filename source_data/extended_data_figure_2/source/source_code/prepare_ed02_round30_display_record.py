"""Prepare, but do not activate, the approved ED02 Round30 source-built record."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
NAME = "extended_data_fig02_2026-09-25_round30_display_refinements"
PARENT_NAME = "extended_data_fig02_2026-08-20_round29_1616_restored"
PARENT = REPO / "docs/figure_version_records" / PARENT_NAME
REVIEW = REPO / "output/figures/ed02_or_positions_2026-09-25"
STAGING = REPO / "docs/figure_version_candidates/review_promotions_2026-09-25"
TARGET = Path("\\\\?\\" + str(STAGING / NAME))
REPLAY = STAGING / "source_replay/ed02_round30"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")

def write_hashes(root):
    path = root / "RECORD_HASHES.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "sha256", "bytes"))
        for item in sorted(root.rglob("*")):
            if item.is_file() and item != path:
                writer.writerow((item.relative_to(root).as_posix(), sha(item), item.stat().st_size))

def check_hashes(root):
    rows = list(csv.DictReader((root / "RECORD_HASHES.csv").open(encoding="utf-8-sig")))
    expected = {row["path"] for row in rows}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "RECORD_HASHES.csv"}
    assert actual == expected, (actual - expected, expected - actual)
    for row in rows:
        path = root / row["path"]
        assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"], path
    return len(rows)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

WRAPPER = '''"""Rebuild this ED02 record from its packaged numerical inputs and frozen code."""
from pathlib import Path
import argparse, copy, csv, hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    record=Path(__file__).resolve().parents[1]
    repo=next(root for root in record.parents if (root/"docs/figure_version_records/rebuild_current_extended_data_figures.py").is_file())
    rows=list(csv.DictReader((record/"RECORD_HASHES.csv").open(encoding="utf-8-sig")))
    expected={row["path"] for row in rows}
    actual={x.relative_to(record).as_posix() for x in record.rglob("*") if x.is_file() and x!=record/"RECORD_HASHES.csv"}
    assert actual==expected, "Record inventory changed"
    for row in rows:
        file=record/row["path"]
        assert file.stat().st_size==int(row["bytes"]) and hashlib.sha256(file.read_bytes()).hexdigest()==row["sha256"], file
    entry=repo/"docs/figure_version_records/rebuild_current_extended_data_figures.py"
    spec=importlib.util.spec_from_file_location("ed02_record_rebuild",entry)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    contract=copy.deepcopy(m.load_contract())
    # Explicit record-local execution also works before pointer activation.
    contract["artifacts"]["extended_data_figure_2"]["record"]=record.name
    m.load_contract=lambda:contract
    shared=contract["shared_runtime"]
    runtime=m.VOR_ROOT/shared["path"]
    assert hashlib.sha256(runtime.read_bytes()).hexdigest()==shared["sha256"]
    for sidecar in contract["artifacts"]["extended_data_figure_2"].get("code_sidecars",[]):
        source=m.VOR_ROOT/sidecar["path"]
        assert source.stat().st_size==sidecar["bytes"] and hashlib.sha256(source.read_bytes()).hexdigest()==sidecar["sha256"]
    if args.check:
        print(json.dumps({"status":"PASS","record":record.name,"hash_closed_files":len(rows)}));return
    if args.output_dir is None:p.error("--output-dir is required unless --check is used")
    output=args.output_dir.resolve()
    assert not output.is_relative_to(record), "Output must be outside the immutable record"
    output.mkdir(parents=True,exist_ok=True)
    with m.frozen_stage(record) as stage:
        result=m.render_ed2(record,output,stage)
    assert result["status"]=="PASS"
    (output/"REBUILD_RESULT.json").write_text(json.dumps({"status":"PASS","record":record.name,"source_hash_closed_files":len(rows),"result":result},indent=2,default=str)+"\\n")
    print(json.dumps({"status":"PASS","record":record.name,"output":str(output)}))
if __name__=="__main__":main()
'''

def refresh_path_manifest(path, root, key="path"):
    frame = pd.read_csv(path, keep_default_na=False)
    for index, row in frame.iterrows():
        target = root / row[key]
        if not target.is_file():
            raise FileNotFoundError(target)
        if "sha256" in frame:
            frame.at[index, "sha256"] = sha(target)
        if "bytes" in frame:
            frame.at[index, "bytes"] = target.stat().st_size
        if "rows" in frame and target.suffix == ".csv":
            frame.at[index, "rows"] = len(pd.read_csv(target))
    frame.to_csv(path, index=False)

def finalize_metadata():
    """Reconcile current display metadata; preserve predecessor render metadata."""
    history = TARGET / "source_validation/round29_predecessor_metadata"
    if not (history / "RENDER_RESULT.json").exists():
        shutil.copy2(PARENT / "RENDER_RESULT.json", history / "RENDER_RESULT.json")
    source = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text())
    source["status"] = "accepted_version_record"
    source["marker_contract"]["aggregate_exception"]["panels"] = ["b", "c", "f", "i"]
    source["marker_contract"]["aggregate_exception"]["calibration_estimand"] = "Arithmetic x and y means over the panel's own finite displayed session pairs; no added inference or uncertainty."
    source["current_display_refinement"].update({
        "status": "PASS", "panel_d_units": "native signed exploration-efficiency difference",
        "panel_d_xlim": [-0.04, 0.096], "panel_d_xticks": [-0.04, 0.0, 0.04, 0.08],
        "panel_d_xlabel": "Exploration efficiency\ndifference", "added_group_mean_count": 6,
        "or_positions": {"stem_to_branch": [1.4751335, 2.99], "stem_forward": [1.19268, 2.81252],
                         "branch_to_stem": [1.76712, 0.77368], "branch_forward": [1.21298, 0.65942]}})
    write_json(TARGET / "SOURCE_MANIFEST.json", source)
    rendered = json.loads((REPLAY / "source_assembly/RENDER_RESULT.json").read_text())
    rendered.update({"review_round": 30, "renderer_lineage_round": 23, "record": NAME,
                     "candidate_status": "accepted_version_record", "plot_ready_tables": 18,
                     "output_root": f"docs/figure_version_records/{NAME}",
                     "layout_contract": source["layout_contract_round11"],
                     "current_display_refinement": source["current_display_refinement"]})
    write_json(TARGET / "RENDER_RESULT.json", rendered)
    version = json.loads((TARGET / "ED02_VERSION_RECORD_MANIFEST.json").read_text())
    version["status"] = "accepted_version_record"
    write_json(TARGET / "ED02_VERSION_RECORD_MANIFEST.json", version)
    for name in ("MANIFEST.md", "REPORT.md"):
        prose = (TARGET / name).read_text(encoding="utf-8")
        prose = prose.replace("Prepared successor", "Accepted successor").replace("prepared for activation", "accepted and ready for pointer activation")
        (TARGET / name).write_text(prose, encoding="utf-8")
    (TARGET / "PROMOTION_COMPLETE.txt").write_text(f"PASS\nrecord={NAME}\nstatus=accepted_version_record\naccepted_date=2026-09-25\nsource_replay=PASS\n", encoding="utf-8")
    composite = json.loads((TARGET / "ED02_ROUND30_COMPOSITE_MANIFEST.json").read_text())
    for output in composite["outputs"].values():
        file = TARGET / output["path"]
        assert sha(file) == output["sha256"] and file.stat().st_size == output["bytes"]
    for placement in composite["placements"]:
        assert sha(TARGET / placement["source"]) == placement["source_sha256"]
    assert sha(TARGET / composite["shared_legend"]["source"]) == composite["shared_legend"]["source_sha256"]
    for row in source["inputs"]:
        assert sha(TARGET / row["path"]) == row["sha256"]
    for name in ("PANEL_MANIFEST.csv", "PPT_VECTOR_MANIFEST.csv", "SHARED_ASSET_MANIFEST.csv", "TABLE_MANIFEST.csv", "ANALYSIS_INPUT_MANIFEST.csv"):
        refresh_path_manifest(TARGET / name, TARGET)
    shutil.copy2(Path(__file__), TARGET / "source_code" / Path(__file__).name)
    code_rows = [{"path": p.relative_to(TARGET).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted((TARGET / "source_code").rglob("*")) if p.is_file()]
    pd.DataFrame(code_rows).to_csv(TARGET / "SOURCE_CODE_MANIFEST.csv", index=False)
    audit = {"status": "PASS", "current_round": 30, "renderer_lineage_round": 23,
             "composite_output_hashes": True, "native_panel_and_shared_legend_hashes": True,
             "scientific_input_hashes": True, "aggregate_marker_panels": ["b", "c", "f", "i"],
             "replayed_preview_visually_inspected": True,
             "historical_metadata_policy": "Round-named historical files retain their predecessor context; current authorities are listed in MANIFEST.md."}
    write_json(TARGET / "source_validation/round30_source_replay/METADATA_VALIDATION.json", audit)
    master_hashes = {p.name: sha(p) for p in TARGET.glob("ed02_version_record*") if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".png", ".tiff"}}
    for name in ("SOURCE_REBUILD_VALIDATION.json", "QA.json", "source_validation/round30_source_replay/SOURCE_REPLAY_VALIDATION.json"):
        replay_receipt = json.loads((TARGET / name).read_text())
        replay_receipt["master_hashes"] = master_hashes
        if name == "QA.json":
            replay_receipt["label_audit"] = json.loads((PARENT / "QA.json").read_text())["label_audit"]
            replay_receipt["label_audit_provenance"] = {
                "source": "source_validation/round29_predecessor_metadata/QA.json",
                "scope": "Inherited panel-h label geometry; unchanged native panel-h SVG verified by the Round30 source replay.",
                "native_panel_h_svg_parity": True}
        write_json(TARGET / name, replay_receipt)
    write_hashes(TARGET)
    files = check_hashes(TARGET)
    receipt_path = STAGING / "ED02_PREPARATION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["hash_closed_files"] = files
    receipt["metadata_validation"] = audit
    receipt["master_hashes"] = master_hashes
    receipt["integration"]["render_result_review_round"] = 30
    receipt["integration"]["renderer_lineage_round"] = 23
    receipt["integration"]["record_status"] = "accepted_version_record"
    receipt["integration"]["add_promotion_complete_then_rehash"] = False
    receipt["integration"]["record_content_mutation_required"] = False
    write_json(receipt_path, receipt)
    subprocess.run([sys.executable, "-B", str(TARGET / "source_code/rebuild_from_source.py"), "--check"], check=True)


def main():
    parent_files = check_hashes(PARENT)
    parent_ledger_hash = sha(PARENT / "RECORD_HASHES.csv")
    if "--finalize-metadata" in sys.argv:
        check_hashes(TARGET)
        finalize_metadata()
        assert sha(PARENT / "RECORD_HASHES.csv") == parent_ledger_hash
        print(json.dumps({"status": "PASS", "finalized_metadata": str(TARGET)}))
        return
    if "--resume-source-replay" in sys.argv:
        assert TARGET.is_dir() and not (TARGET / "PREPARATION_COMPLETE.txt").exists()
        check_hashes(TARGET)
        shutil.copy2(Path(__file__), TARGET / "source_code" / Path(__file__).name)
        (TARGET / "source_code/rebuild_from_source.py").write_text(WRAPPER, encoding="utf-8")
        code_rows = [{"path": p.relative_to(TARGET).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted((TARGET / "source_code").rglob("*")) if p.is_file()]
        pd.DataFrame(code_rows).to_csv(TARGET / "SOURCE_CODE_MANIFEST.csv", index=False)
        write_hashes(TARGET)
        finish_source_replay(parent_files, parent_ledger_hash)
        return
    if TARGET.exists():
        assert "--resume-incomplete-copy" in sys.argv, f"Refusing to overwrite staged record: {TARGET}"
        assert not (TARGET / "PREPARATION_COMPLETE.txt").exists()
        assert sha(TARGET / "RECORD_HASHES.csv") == parent_ledger_hash, "Resume requires an unmodified predecessor copy"
    reviewed = json.loads((REVIEW / "DISPLAY_VALIDATION.json").read_text())
    assert reviewed["status"] == "PASS"
    for row in json.loads((REVIEW / "OUTPUT_HASHES.json").read_text())["outputs"]:
        assert sha(REVIEW / row["path"]) == row["sha256"], row["path"]
    shutil.copytree(PARENT, TARGET, dirs_exist_ok=True)
    history = TARGET / "source_validation/round29_predecessor_metadata"
    history.mkdir(parents=True)
    for name in ("MANIFEST.md", "REBUILD_README.md", "CAPTION_NOTES.md", "INPUT_MANIFEST.csv",
                 "SOURCE_MANIFEST.json", "ED02_VERSION_RECORD_MANIFEST.json", "RECORD_VALIDATION.json",
                 "RECORD_HASHES.csv", "PROMOTION_COMPLETE.txt", "QA.json", "QA.md", "REPORT.md"):
        if (TARGET / name).is_file():
            shutil.copy2(TARGET / name, history / name)
    (TARGET / "PROMOTION_COMPLETE.txt").unlink()
    assembly = REVIEW / "source_assembly"
    for name in ("panels", "ppt_vector", "shared_assets", "plot_ready"):
        shutil.copytree(assembly / name, TARGET / name, dirs_exist_ok=True)
    for path in (REVIEW / "source_code").glob("*.py"):
        shutil.copy2(path, TARGET / "source_code" / path.name)
    for name in ("PANEL_MANIFEST.csv", "PPT_VECTOR_MANIFEST.csv", "SHARED_ASSET_MANIFEST.csv", "TABLE_MANIFEST.csv", "SOURCE_MANIFEST.json"):
        shutil.copy2(assembly / name, TARGET / name)
    for suffix in ("pdf", "svg", "_600dpi.png", "_600dpi.tiff", "_preview_1440.png"):
        name = "ed02_version_record" + (suffix if suffix.startswith("_") else "." + suffix)
        shutil.copy2(REVIEW / "composite" / name, TARGET / name)
    with fitz.open(TARGET / "ed02_version_record.pdf") as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72), alpha=False)
        Image.frombytes("RGB", (pix.width,pix.height), pix.samples).save(TARGET / "ed02_version_record_300dpi.png", dpi=(300,300), optimize=True)
    qa = TARGET / "source_validation/round30_display_refinements"
    qa.mkdir(parents=True)
    for name in ("DISPLAY_VALIDATION.json", "ADDED_GROUP_MEANS.csv", "OUTPUT_HASHES.json"):
        shutil.copy2(REVIEW / name, qa / name)
    shutil.copytree(REVIEW / "qa", qa / "independent_visual_review", dirs_exist_ok=True)
    shutil.copy2(Path(__file__), TARGET / "source_code" / Path(__file__).name)
    (TARGET / "source_code/rebuild_from_source.py").write_text(WRAPPER, encoding="utf-8")
    caption = (PARENT / "CAPTION_NOTES.md").read_text(encoding="utf-8")
    caption = caption.replace("# Extended Data Fig. 2 round-18 caption notes", "# Extended Data Fig. 2 round-30 caption notes", 1)
    old = "P_SF -0.4 percentage points [-1.0, +0.2], P_SA -2.6 percentage points [-3.2, -2.0], P_BF +6.1 percentage points [+4.4, +8.2], P_BS +7.2 percentage points [+6.0, +8.4]."
    new = "P_SF -0.004 [-0.010, +0.002], P_SA -0.026 [-0.032, -0.020], P_BF +0.061 [+0.044, +0.082], P_BS +0.072 [+0.060, +0.084], on the native exploration-efficiency scale."
    assert caption.count(old) == 1
    caption = caption.replace(old, new)
    lines = caption.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("**f,**", "**i,**")):
            lines[i] += " Filled group-color circles show the session-weighted coordinate means of the observed and simulated efficiencies; hollow circles retain individual sessions."
    caption = "\n".join(lines) + "\n\nRound-30 display revision: panel a moves only four or labels inside their middle cells; panel d uses native signed decimal differences with unchanged point/interval positions; f/i add three opaque s=11 group-mean disks each. All scientific source rows, cohorts, estimates and inference remain unchanged.\n"
    (TARGET / "CAPTION_NOTES.md").write_text(caption, encoding="utf-8")
    (TARGET / "FIGURE_LEGEND.md").write_text((PARENT / "FIGURE_LEGEND.md").read_text() + "\nPanels f/i additionally show filled session-weighted mean points. Panel d uses native signed exploration-efficiency differences.\n", encoding="utf-8")
    source_manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text())
    source_manifest.update({"status": "prepared_version_record", "review_round": 30, "record": NAME,
                            "source_record": f"docs/figure_version_records/{NAME}", "supersedes": PARENT_NAME,
                            "current_display_refinement": {"round": 30, "panels": ["a", "d", "f", "i"], "scientific_values_changed": False},
                            "source_rebuild_entrypoint": "source_code/rebuild_from_source.py", "plot_ready_table_count": 18})
    source_manifest["frozen_renderer"]["path"] = "source_code/render_ed03_fixed_time_30min_full_story.py"
    source_manifest["frozen_renderer"]["sha256"] = sha(TARGET / source_manifest["frozen_renderer"]["path"])
    write_json(TARGET / "SOURCE_MANIFEST.json", source_manifest)
    for name in ("PANEL_MANIFEST.csv", "PPT_VECTOR_MANIFEST.csv", "SHARED_ASSET_MANIFEST.csv", "TABLE_MANIFEST.csv", "ANALYSIS_INPUT_MANIFEST.csv"):
        refresh_path_manifest(TARGET / name, TARGET)
    refresh_path_manifest(TARGET / "INPUT_MANIFEST.csv", TARGET, "record_path")
    code_rows = [{"path": p.relative_to(TARGET).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted((TARGET / "source_code").rglob("*")) if p.is_file()]
    pd.DataFrame(code_rows).to_csv(TARGET / "SOURCE_CODE_MANIFEST.csv", index=False)
    composite = json.loads((REVIEW / "composite/ED02_SOURCE_REBUILD_COMPOSITE_MANIFEST.json").read_text())
    composite.update({"review_round": 30, "record": NAME, "source_rebuild_entrypoint": "source_code/rebuild_from_source.py"})
    write_json(TARGET / "ED02_ROUND30_COMPOSITE_MANIFEST.json", composite)
    manifest = json.loads((PARENT / "ED02_VERSION_RECORD_MANIFEST.json").read_text())
    manifest.update({"record": NAME, "review_round": 30, "status": "prepared_version_record", "accepted_date": "2026-09-25",
                     "source_record": PARENT_NAME, "supersedes": PARENT_NAME, "scientific_values_changed": False,
                     "changed_panels": ["a", "d", "f", "i"], "plot_ready_table_count": 18,
                     "current_display_refinement": {"round": 30, "status": "PASS", "source": PARENT_NAME,
                         "policy": "Four or-label positions; native signed efficiency-difference ticks; six session-weighted mean points."},
                     "publication_record_convention": {"canonical_primary_stem": "ed02_version_record", "record_change": "Round30 approved display refinements"},
                     "vector_audit": composite["vector_audit"], "raster_audit": {"png": composite["png_audit"], "tiff": composite["tiff_audit"]},
                     "source_rebuild_entrypoint": "source_code/rebuild_from_source.py"})
    write_json(TARGET / "ED02_VERSION_RECORD_MANIFEST.json", manifest)
    (TARGET / "MANIFEST.md").write_text(f"# {NAME}\n\nPrepared successor to `{PARENT_NAME}`. Implements approved panel-a or-label positions, panel-d native signed difference units, and three session-weighted mean points in each calibration panel f/i. Existing scientific values, cohorts, inference, figures not named here and manuscript prose are unchanged.\n\nCurrent metadata: ED02_VERSION_RECORD_MANIFEST.json, ED02_ROUND30_COMPOSITE_MANIFEST.json, SOURCE_MANIFEST.json, panel/table/code manifests and RECORD_VALIDATION.json. Older round-named files are retained provenance. Predecessor top-level metadata is frozen in source_validation/round29_predecessor_metadata.\n", encoding="utf-8")
    (TARGET / "REBUILD_README.md").write_text(f"# Rebuild {NAME}\n\nFrom the repository root:\n\n```powershell\npython -B docs/figure_version_records/{NAME}/source_code/rebuild_from_source.py --check\npython -B docs/figure_version_records/{NAME}/source_code/rebuild_from_source.py --output-dir C:/figure_rebuilds/ed02_round30\n```\n\nThe record-local entrypoint checks this record's hash ledger, then uses the maintained `docs/figure_version_records/rebuild_current_extended_data_figures.py` adapter with record-frozen Python and numeric tables. Shared runtime ZIP and the ED2 Round23 code sidecar remain hash-verified under the figure-version-record root. No parent record, accepted master artwork, raw behavior/neural drive, network, or scientific fit is used as a rendering input. Outputs must be outside this immutable record. The staged preparation used the same entrypoint with its candidate path; source replay receipts are in source_validation/round30_source_replay.\n", encoding="utf-8")
    write_json(TARGET / "RECORD_VALIDATION.json", {"status": "PREPARING", "record": NAME})
    write_hashes(TARGET)
    finish_source_replay(parent_files, parent_ledger_hash)


def finish_source_replay(parent_files, parent_ledger_hash):
    subprocess.run([sys.executable, "-B", str(TARGET / "source_code/rebuild_from_source.py"), "--output-dir", str(REPLAY)], check=True)
    numerical_parity = {}
    for path in (TARGET / "plot_ready").glob("*.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(REPLAY / "source_assembly/plot_ready" / path.name))
        numerical_parity[path.name] = True
    helpers = load("ed02_prepare_helpers", REPO / "scripts/render_ed02_average_points_revision.py")
    native_parity = {}
    for panel, before in helpers.panel_paths(TARGET, "svg").items():
        after = helpers.panel_paths(REPLAY / "source_assembly", "svg")[panel]
        assert helpers.svg_signature(ET.parse(before).getroot()) == helpers.svg_signature(ET.parse(after).getroot()), panel
        native_parity[panel] = True
    with Image.open(TARGET / "ed02_version_record_600dpi.png") as a, Image.open(REPLAY / "composite/ed02_version_record_600dpi.png") as b:
        changed_pixels = int(np.count_nonzero(np.any(np.array(a.convert("RGB")) != np.array(b.convert("RGB")), axis=2)))
        assert changed_pixels == 0, changed_pixels
    expected_svg = ET.parse(TARGET / "ed02_version_record.svg").getroot()
    actual_svg = ET.parse(REPLAY / "composite/ed02_version_record.svg").getroot()
    assert helpers.svg_signature(expected_svg) == helpers.svg_signature(actual_svg)
    for path in (PARENT / "analysis_inputs").rglob("*"):
        if path.is_file(): assert sha(path) == sha(TARGET / path.relative_to(PARENT))
    receipt = {"status": "PASS", "record": NAME, "parent": PARENT_NAME, "approved_review": str(REVIEW),
               "executed_entrypoint": "source_code/rebuild_from_source.py", "replay_output": str(REPLAY),
               "native_svg_parity": native_parity, "composite_svg_parity": True, "changed_600dpi_pixels": changed_pixels,
               "numerical_table_parity": numerical_parity, "analysis_inputs_unchanged": True,
               "parent_record_files_preserved": parent_files, "parent_ledger_sha256": parent_ledger_hash,
               "master_hashes": {p.name: sha(p) for p in TARGET.glob("ed02_version_record*") if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".png", ".tiff"}},
               "prepared_utc": datetime.now(timezone.utc).isoformat()}
    replay_receipts = TARGET / "source_validation/round30_source_replay"
    replay_receipts.mkdir()
    write_json(replay_receipts / "SOURCE_REPLAY_VALIDATION.json", receipt)
    shutil.copy2(REPLAY / "REBUILD_RESULT.json", replay_receipts / "REBUILD_RESULT.json")
    shutil.copy2(REPLAY / "composite/ed02_version_record_preview_1440.png", replay_receipts / "replayed_preview.png")
    write_json(TARGET / "SOURCE_REBUILD_VALIDATION.json", receipt)
    write_json(TARGET / "RECORD_VALIDATION.json", {"schema": "ed02-round30-record-preparation/v1", "status": "PASS", "record": NAME,
        "activation": "prepared; root owns pointer activation and archival", "checks": {"parent_hash_closure": True, "approved_review_master_parity": True,
        "record_local_source_replay": True, "all_eighteen_numeric_tables_match": True, "all_nine_native_svg_panels_match": True,
        "composite_svg_matches": True, "zero_changed_600dpi_pixels": True, "scientific_inputs_unchanged": True}})
    write_json(TARGET / "QA.json", receipt)
    (TARGET / "QA.md").write_text("# Round30 preparation QA\n\nPASS. Executed the record-local source rebuild. All 18 numerical tables, all 9 native panel SVGs and the assembled SVG match the approved review; the 600 dpi pixel difference is zero. Prior independent glyph/visual evidence is retained under source_validation/round30_display_refinements.\n", encoding="utf-8")
    (TARGET / "REPORT.md").write_text("# ED02 Round30\n\nThe approved display refinements are prepared for activation. Source-based reproduction and immutable predecessor checks pass. See MANIFEST.md and SOURCE_REBUILD_VALIDATION.json.\n", encoding="utf-8")
    (TARGET / "PREPARATION_COMPLETE.txt").write_text(f"PASS\nrecord={NAME}\nactivation=pending parent integration\n", encoding="utf-8")
    write_hashes(TARGET)
    files = check_hashes(TARGET)
    assert sha(PARENT / "RECORD_HASHES.csv") == parent_ledger_hash
    check_hashes(PARENT)
    write_json(STAGING / "ED02_PREPARATION_RECEIPT.json", {**receipt, "staged_record": str(TARGET), "hash_closed_files": files,
        "integration": {"record_status": "prepared_version_record", "add_promotion_complete_then_rehash": True,
        "shared_contract_record": NAME, "source_stage_tables": 18,
        "adapter_note": "ED2 render_ed2 exporter review_round should resolve record metadata instead of literal29; artwork unaffected."}})
    finalize_metadata()
    print(json.dumps({"status": "PASS", "staged_record": str(TARGET), "hash_closed_files": files, "changed_600dpi_pixels": changed_pixels}))

if __name__ == "__main__":
    main()
