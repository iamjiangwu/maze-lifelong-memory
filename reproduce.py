"""Rebuild the paper's figures, tables and statistics from derived Source Data."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

REPOSITORY = Path(__file__).resolve().parent


def inside(root: Path, relative: str) -> Path:
    """Resolve a catalog path without allowing it to escape its root."""
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Invalid catalog path: {relative}')
    return path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_layout(repository: Path = REPOSITORY) -> dict:
    """Read the artifact list and the input/code inventory."""
    layout = json.loads((repository/'reproduction/layout.json').read_text(encoding='utf-8'))
    if layout['schema'] != 'source-data/v2':
        raise ValueError('Unsupported Source Data catalog version')
    aliases = [item['alias'] for item in layout['artifacts']]
    required = {str(i) for i in range(1,6)} | {f'ed{i}' for i in range(1,11)} | {f'table{i}' for i in range(1,4)}
    if len(aliases) != 18 or set(aliases) != required:
        raise ValueError('The catalog must contain all 18 publication artifacts')
    return layout


def verify_sources(layout: dict, repository: Path = REPOSITORY) -> dict:
    """Check distributed inputs and code against their recorded hashes."""
    seen = set()
    for row in layout['files']:
        path = inside(repository,row['repository_path'])
        if path in seen:
            raise ValueError('Duplicate inventory path: '+row['repository_path'])
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row['bytes'] or checksum(path) != row['sha256']:
            raise ValueError('Source checksum changed: '+row['repository_path'])
    return {'status':'PASS','artifacts':18,'source_files':len(seen)}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_inputs(layout: dict, artifact: dict, stage: Path) -> None:
    """Supply the relative paths required by behavioral/table renderers."""
    prefix = artifact['artifact_id']+'/'
    for row in layout['files']:
        for relative in row.get('loader_paths',[]):
            if relative.startswith(prefix) or relative.startswith('shared/'):
                destination = inside(stage,relative)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(inside(REPOSITORY,row['repository_path']),destination)


def build_staged(artifact: dict, stage: Path, output: Path, statistics_only: bool):
    """Run a behavioral/table renderer with local numerical inputs."""
    source, alias = stage/artifact['artifact_id']/'source', artifact['alias']
    if statistics_only:
        if alias == '1':
            module = load_module(REPOSITORY/'reproduction/behavior_statistics.py','behavior_statistics')
            records = module.StatisticsOutput(output)
            module.main1(stage,records)
            return records.analyses
        if alias in ('ed2','ed3'):
            module = load_module(REPOSITORY/'reproduction/policy_statistics.py','policy_statistics')
            analyses = []
            getattr(module,alias)(source,output,analyses)
            return analyses
        return []
    if alias == '1':
        module = load_module(source/'source_code/build_figure1.py','behavior_page')
        module.BASE = source
        module.build(output)
    elif alias in ('ed2','ed3','ed4'):
        module = load_module(REPOSITORY/'reproduction/extended_builds.py','extended_builds')
        module.REPO_ROOT, module.VOR_ROOT = stage, stage/'shared'
        with module.frozen_stage(source) as imports:
            result = getattr(module,'render_'+alias)(source,output,imports)
        if str(result['status']).upper() != 'PASS':
            raise RuntimeError(result)
        if alias == 'ed4':
            import fitz
            with fitz.open(output/'ed04_registration_quality_assembly.pdf') as pdf:
                pdf[0].get_pixmap(dpi=600,alpha=False).save(output/'ed04_registration_quality_assembly.png')
    elif alias == 'table1':
        subprocess.run([sys.executable,'-B',str(source/'source_code/build_workbook.py'),str(output)],check=True)
        subprocess.run([sys.executable,'-B',str(source/'source_code/build_pdf.py'),'--output-dir',str(output)],check=True)
    elif alias == 'table2':
        subprocess.run([sys.executable,'-B',str(source/'source_code/rebuild.py'),'--output-dir',str(output),'--renderer-python',sys.executable],cwd=source,check=True)


def build_one(layout: dict, artifact: dict, output: Path, statistics_only: bool):
    """Dispatch one artifact without importing raw-data analyses."""
    output.mkdir(parents=True,exist_ok=True)
    directory = inside(REPOSITORY,artifact['directory'])
    if artifact['builder'] == 'derived':
        module = load_module(directory/'code/derived_panels.py','source_'+artifact['alias'])
        return module.statistics(output) if statistics_only else module.render(output)
    if artifact['builder'] == 'glossary':
        if not statistics_only:
            subprocess.run([sys.executable,'-B',str(directory/'build.py'),'--output-dir',str(output)],check=True)
        return None
    with tempfile.TemporaryDirectory(prefix='maze_inputs_') as temporary:
        stage = Path(temporary).resolve()
        stage_inputs(layout,artifact,stage)
        return build_staged(artifact,stage,output,statistics_only)


def validate_output_folder(output: Path, parser: argparse.ArgumentParser) -> None:
    """Keep both the public launcher and its workers from overwriting sources."""
    protected = [REPOSITORY/folder for folder in ('figures','tables','reproduction','docs','tests','.git','.github')]
    if output==REPOSITORY or any(output==p or output.is_relative_to(p) for p in protected):
        parser.error('Choose an output folder outside the source and documentation folders')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',default='all',help='1–5, ed1–ed10, table1–table3, or all')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--statistics-only',action='store_true',help='Recalculate tests from included observations')
    modes.add_argument('--figures-only',action='store_true',help='Draw figures and build tables')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    layout = read_layout()
    if args.list:
        for item in layout['artifacts']:
            print(f'{item["alias"]:7} {item["label"]}: {item["title"]}')
        return 0
    selected = [item for item in layout['artifacts'] if args.figure.lower() in ('all',item['alias'],item['artifact_id'])]
    if not selected:
        parser.error('Unknown figure/table; run with --list for valid names')
    if args.worker:
        if len(selected)!=1 or args.output_dir is None:
            parser.error('A worker requires one artifact and an output directory')
        validate_output_folder(args.output_dir.resolve(), parser)
        build_one(layout,selected[0],args.output_dir.resolve(),args.statistics_only)
        return 0
    verification = verify_sources(layout)
    if args.check_only:
        print(json.dumps(verification))
        return 0
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output = (args.output_dir or REPOSITORY/'outputs'/suffix).resolve()
    validate_output_folder(output, parser)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    environment = dict(os.environ,PYTHONPATH=str(REPOSITORY/'reproduction/runtime'),PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTHONUTF8='1')
    reports = []
    for artifact in selected:
        destination = output/artifact['artifact_id']
        command = [sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--figure',artifact['alias'],'--output-dir',str(destination)]
        if args.statistics_only:
            command.append('--statistics-only')
        print('BUILD '+artifact['label'],flush=True)
        result = subprocess.run(command,env=environment,cwd=REPOSITORY)
        reports.append({'artifact':artifact['artifact_id'],'exit_code':result.returncode})
        if result.returncode:
            break
        if not args.statistics_only and not args.figures_only and artifact['builder']=='staged':
            result = subprocess.run(command[:-1]+[str(destination/'statistics'),'--statistics-only'],env=environment,cwd=REPOSITORY)
            if result.returncode:
                reports[-1]['exit_code'] = result.returncode
                break
    report = {'status':'PASS' if all(r['exit_code']==0 for r in reports) and len(reports)==len(selected) else 'FAIL',
              'scope':'build completion; numerical and artwork validation are separate',
              'catalog_sha256':checksum(REPOSITORY/'reproduction/layout.json'),'input_verification':verification,'artifacts':reports}
    (output/'BUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if report['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
