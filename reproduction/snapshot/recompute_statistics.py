"""Recalculate publication statistics from release-local derived observations."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    root,out=args.source_root.resolve(),args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name in ['main','ed','root']:
        path=root/'shared'/f'source_release_{name}_statistics.py'
        spec=importlib.util.spec_from_file_location('release_statistics_'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        reports.append(module.run(root,out/name))
    status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL'
    report=dict(status=status,source_boundary='derived neuron/pair/mouse observations and frozen shuffle/simulation draws; no acquisition files',
                accepted_artwork_staged=False,analyses=sum(len(r['analyses']) for r in reports),plugins=reports)
    (out/'STATISTICS_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=status,analyses=report['analyses'],output=str(out))))
    raise SystemExit(0 if status=='PASS' else 1)


if __name__=='__main__':main()
