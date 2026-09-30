"""Rebuild this ED02 record from its packaged numerical inputs and frozen code."""
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
    (output/"REBUILD_RESULT.json").write_text(json.dumps({"status":"PASS","record":record.name,"source_hash_closed_files":len(rows),"result":result},indent=2,default=str)+"\n")
    print(json.dumps({"status":"PASS","record":record.name,"output":str(output)}))
if __name__=="__main__":main()
