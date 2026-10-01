"""Build manuscript figures and tables from their included derived Source Data.

The public folder layout is optimized for browsing. This launcher restores the
byte-identical original input layout in a temporary directory, then calls the
unchanged, verified rendering/statistics engine. No raw-data processing runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Invalid catalog path: {relative}')
    return path


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_layout(repository: Path = REPOSITORY) -> dict:
    layout = json.loads((repository / 'reproduction/layout.json').read_text(encoding='utf-8'))
    if layout['schema'] != 'reader-layout/v1':
        raise ValueError('Unsupported reproduction layout version')
    return layout


def verify_sources(layout: dict, repository: Path = REPOSITORY) -> dict:
    """Check complete original-manifest coverage and every retained file hash."""
    manifest_path = inside(repository, layout['snapshot_manifest'])
    if checksum(manifest_path) != layout['snapshot_manifest_sha256']:
        raise ValueError('Original snapshot manifest checksum changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {row['path']: row['sha256'] for row in manifest['files']
                if row['role'] not in ('comparison_only', 'verification_only')}
    expected['RELEASE_MANIFEST.json'] = layout['snapshot_manifest_sha256']
    mapped = {row['snapshot_path']: row['sha256'] for row in layout['files']}
    if len(mapped) != len(layout['files']) or mapped != expected:
        raise ValueError('Relocation map does not cover the complete original source snapshot')
    checked = {}
    for row in layout['files']:
        path = inside(repository, row['repository_path'])
        if path not in checked:
            if not path.is_file():
                raise FileNotFoundError(f'Missing source file: {row["repository_path"]}')
            checked[path] = checksum(path)
        if checked[path] != expected[row['snapshot_path']]:
            raise ValueError(f'Source checksum changed: {row["repository_path"]}')
    return {'status': 'PASS', 'artifacts': len(layout['artifacts']),
            'original_source_paths': len(mapped), 'retained_source_files': len(checked)}


def stage_snapshot(layout: dict, stage: Path, repository: Path = REPOSITORY):
    """Restore original names/import paths without changing the public layout."""
    for row in layout['files']:
        source = inside(repository, row['repository_path'])
        destination = inside(stage, row['snapshot_path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def resolve_figure(selection: str, layout: dict) -> str:
    if selection == 'all':
        return 'all'
    for artifact in layout['artifacts']:
        if selection in (artifact['alias'], artifact['artifact_id']):
            return artifact['artifact_id']
    raise ValueError('Unknown figure/table. Run python reproduce.py --list for valid names.')


def export_environment(node: str | None) -> tuple[str, dict]:
    """Use explicit dependencies or the already installed bundled export runtime."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    profile = Path(os.environ.get('USERPROFILE', str(Path.home())))
    bundled = profile / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    bundled_node = bundled / 'bin/node.exe'
    selected = node or (str(bundled_node) if bundled_node.is_file() else 'node')
    if 'ARTIFACT_TOOL_NODE_MODULES' not in environment and (bundled/'node_modules').is_dir():
        environment['ARTIFACT_TOOL_NODE_MODULES'] = str(bundled/'node_modules')
    return selected, environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', default='all', help='1–5, ed1–ed10, table1–table2, or all (default)')
    parser.add_argument('--output-dir', type=Path, help='New output directory; default: outputs/<timestamp>')
    parser.add_argument('--list', action='store_true', help='List figure/table names without building')
    parser.add_argument('--check-only', action='store_true', help='Verify source integrity without building')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--statistics-only', action='store_true', help='Recalculate all 53 recorded statistical groups')
    mode.add_argument('--figures-only', action='store_true', help='Build artwork/tables without statistical replay')
    parser.add_argument('--node', help='Explicit Node.js executable for XLSX exports')
    args = parser.parse_args(argv)
    layout = read_layout()
    if args.list:
        for item in layout['artifacts']:
            print(f'{item["alias"]:7} {item["label"]}: {item["title"]}')
        return 0
    try:
        figure = resolve_figure(args.figure.lower(), layout)
    except ValueError as error:
        parser.error(str(error))
    if args.statistics_only and figure != 'all':
        parser.error('Statistical replay runs all families together; omit --figure in this mode.')
    report = verify_sources(layout)
    if args.check_only:
        print(json.dumps(report))
        return 0
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    output = (args.output_dir or REPOSITORY / 'outputs' / suffix).resolve()
    protected = [REPOSITORY/'figures', REPOSITORY/'tables', REPOSITORY/'reproduction', REPOSITORY/'docs']
    if output == REPOSITORY or any(output == path or output.is_relative_to(path) for path in protected):
        parser.error('Choose an output directory outside the source/docs folders, for example outputs/rebuild.')
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new or empty output directory: ' + str(output))
    node, environment = export_environment(args.node)
    print(f'Outputs: {output}', flush=True)
    with tempfile.TemporaryDirectory(prefix='maze_source_') as temporary:
        stage = Path(temporary).resolve()
        stage_snapshot(layout, stage)
        command = [sys.executable, '-B', str(stage/'reproduce.py'), '--no-compare',
                   '--artifact', figure, '--output-dir', str(output), '--node', node]
        if args.statistics_only:
            command.append('--statistics-only')
        if args.figures_only:
            command.append('--figures-only')
        return subprocess.call(command, cwd=stage, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
