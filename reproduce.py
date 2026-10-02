"""Build manuscript figures and tables from their included derived Source Data.

The public folder layout is optimized for browsing. This launcher verifies the
retained inputs and restores their paths in a temporary directory,
then calls the rendering/statistics engine. No raw-data processing runs.
"""
from __future__ import annotations

import argparse
import csv
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

# Only these inspected checksum inventories are projected in disposable stages.
# Their scientific settings and all retained entries remain unchanged.
HASH_INVENTORIES = {
    'main_figure_1/source/frozen_source/SOURCE_INPUTS.csv': ('csv', ''),
    'main_figure_3/source/SOURCE_INPUTS.json': ('files', ''),
    'main_figure_4/source/INPUT_HASHES.json': ('hashes', ''),
    'main_figure_5/source/SOURCE_INPUTS.json': ('sources', ''),
    'extended_data_figure_8/source/INPUT_MANIFEST.json': ('files_frozen', ''),
    'extended_data_figure_9/source/SOURCE_INPUTS.json': ('frozen_files', 'source_data/frozen'),
    'extended_data_figure_9/source/source_data/level_comparison/MANIFEST.json': ('outputs', ''),
    'extended_data_figure_9/source/source_data/first30_addition/manifest.json': ('outputs', ''),
    'extended_data_figure_9/source/source_data/population_vector_shuffle/manifest.json': ('outputs', ''),
    'extended_data_figure_9/source/source_data/population_vector_distribution/OUTPUT_HASHES.json': ('hashes', ''),
    'extended_data_figure_10/source/SOURCE_INPUTS.json': ('files', ''),
}


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


def source_catalog(layout: dict, repository: Path = REPOSITORY) -> tuple[dict, dict]:
    """Authenticate the current source authority and required runtime paths."""
    manifest_path = inside(repository, layout['snapshot_manifest'])
    if checksum(manifest_path) != layout['snapshot_manifest_sha256']:
        raise ValueError('Original snapshot manifest checksum changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']
                if row['role'] not in ('comparison_only', 'verification_only')}
    expected['RELEASE_MANIFEST.json'] = {'sha256': layout['snapshot_manifest_sha256'],
                                        'bytes': manifest_path.stat().st_size}
    originals = {entry['artifact_id']: entry for entry in manifest['artifacts']}
    artifacts = {entry['artifact_id']: entry for entry in layout['artifacts']}
    if (len(originals) != 17 or len(artifacts) != len(layout['artifacts'])
            or set(artifacts) != set(originals)):
        raise ValueError('Layout must retain all 17 original artifact identities')
    for artifact, entry in artifacts.items():
        if entry['outputs'] != originals[artifact]['accepted_outputs']:
            raise ValueError('Original artifact output metadata changed: ' + artifact)
        for key in ('strategy', 'record_name', 'accepted_outputs', 'authority', 'panels'):
            if key in entry and entry[key] != originals[artifact][key]:
                raise ValueError('Original artifact configuration changed: ' + artifact)
    mapped = {row['snapshot_path']: row for row in layout['files']}
    if len({path.casefold() for path in mapped}) != len(layout['files']):
        raise ValueError('Duplicate original source path in relocation map')
    required = {'RELEASE_MANIFEST.json', 'reproduce.py', 'recompute_statistics.py'}
    required.update(path for path in expected if path.startswith('shared/'))
    required.update(path for path in HASH_INVENTORIES if path in expected)
    if not required.issubset(mapped):
        raise ValueError('Required reproduction machinery is missing: ' + ', '.join(sorted(required-set(mapped))))
    for artifact in originals:
        if not any(path.startswith(artifact+'/source/') for path in mapped):
            raise ValueError('Artifact has no retained source inputs: ' + artifact)
    for snapshot, row in mapped.items():
        inside(repository, snapshot)
        inside(repository, row['repository_path'])
        if (snapshot not in expected or row['sha256'] != expected[snapshot]['sha256']
                or int(row['bytes']) != int(expected[snapshot]['bytes'])):
            raise ValueError('Source metadata differs from original manifest: ' + snapshot)
    if mapped['RELEASE_MANIFEST.json']['repository_path'] != layout['snapshot_manifest']:
        raise ValueError('Original manifest relocation differs from its authenticated path')
    return manifest, mapped


def verify_sources(layout: dict, repository: Path = REPOSITORY) -> dict:
    """Verify every retained byte against the authenticated current inventory."""
    _, mapped = source_catalog(layout, repository)
    checked = {}
    for row in layout['files']:
        path = inside(repository, row['repository_path'])
        if path not in checked:
            if not path.is_file():
                raise FileNotFoundError(f'Missing source file: {row["repository_path"]}')
            checked[path] = (checksum(path), path.stat().st_size)
        if checked[path] != (row['sha256'], int(row['bytes'])):
            raise ValueError(f'Source checksum changed: {row["repository_path"]}')
    return {'status': 'PASS', 'artifacts': len(layout['artifacts']),
            'original_source_paths': len(mapped), 'retained_source_files': len(checked)}


def project_input_inventories(stage: Path, retained: set[str]) -> list[dict]:
    """Drop omitted checksum entries only; preserve the original source metadata."""
    projections = []
    retained_paths = {inside(stage, name) for name in retained}
    for relative, (kind, base) in HASH_INVENTORIES.items():
        if relative not in retained:
            continue
        path = inside(stage, relative)
        original_hash = checksum(path)
        root = path.parent / base

        def keep(name: str) -> bool:
            target = inside(root.resolve(), name.replace('\\', '/'))
            return target in retained_paths

        if kind == 'csv':
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            selected = [row for row in rows if keep(row['path'])]
            if selected == rows:
                continue
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
            dropped = len(rows)-len(selected)
        else:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if kind == 'frozen_files':
                selected = [row for row in payload if keep(row['frozen_file'])]
                dropped = len(payload)-len(selected)
                payload = selected
            elif kind in ('files', 'files_frozen'):
                rows = payload['files']
                key = 'frozen' if kind == 'files_frozen' else 'path'
                selected = [row for row in rows if keep(row[key])]
                dropped = len(rows)-len(selected)
                payload['files'] = selected
            else:
                rows = payload if kind == 'hashes' else payload[kind]
                selected = {name: row for name, row in rows.items() if keep(name)}
                dropped = len(rows)-len(selected)
                if kind == 'hashes':
                    payload = selected
                else:
                    payload[kind] = selected
            if not dropped:
                continue
            path.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
        projections.append({'path': relative, 'original_sha256': original_hash,
                            'staged_sha256': checksum(path), 'omitted_entries': dropped})
    return projections


def stage_snapshot(layout: dict, stage: Path, repository: Path = REPOSITORY):
    """Stage retained originals and disposable inventories for the unchanged engine."""
    manifest, mapped = source_catalog(layout, repository)
    for row in layout['files']:
        source = inside(repository, row['repository_path'])
        destination = inside(stage, row['snapshot_path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if (checksum(destination), destination.stat().st_size) != (row['sha256'], int(row['bytes'])):
            raise ValueError('Staged source differs from original: ' + row['snapshot_path'])
    projections = project_input_inventories(stage, set(mapped))
    projected = {row['path'] for row in projections}
    staged = dict(manifest)
    staged['files'] = [dict(row) for row in manifest['files'] if row['path'] in mapped]
    for row in staged['files']:
        if row['path'] in projected:
            path = inside(stage, row['path'])
            row.update(sha256=checksum(path), bytes=path.stat().st_size)
    staged['staging'] = {'original_manifest_sha256': layout['snapshot_manifest_sha256'],
                         'retained_source_paths': len(mapped), 'checksum_inventory_projections': projections}
    (stage/'RELEASE_MANIFEST.json').write_text(json.dumps(staged, indent=2)+'\n', encoding='utf-8')
    return staged


def resolve_figure(selection: str, layout: dict) -> str:
    if selection == 'all':
        return 'all'
    for artifact in layout['artifacts']:
        if selection in (artifact['alias'], artifact['artifact_id']):
            return artifact['artifact_id']
    raise ValueError('Unknown figure/table. Run python reproduce.py --list for valid names.')


def export_environment() -> dict:
    """Set the public Python export environment; no private Node runtime is needed."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    return environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', default='all', help='1–5, ed1–ed10, table1–table2, or all (default)')
    parser.add_argument('--output-dir', type=Path, help='New output directory; default: outputs/<timestamp>')
    parser.add_argument('--list', action='store_true', help='List figure/table names without building')
    parser.add_argument('--check-only', action='store_true', help='Verify source integrity without building')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--statistics-only', action='store_true', help='Recalculate the included current statistical groups')
    mode.add_argument('--figures-only', action='store_true', help='Build artwork/tables without statistical replay')
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
    environment = export_environment()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'BUILD_SOURCE_AUTHORITY.json').write_text(json.dumps({
        'manifest_sha256': layout['snapshot_manifest_sha256'],
        'layout_sha256': checksum(REPOSITORY/'reproduction/layout.json'),
        'input_verification': report,
    }, indent=2)+'\n', encoding='utf-8')
    print(f'Outputs: {output}', flush=True)
    with tempfile.TemporaryDirectory(prefix='maze_source_') as temporary:
        stage = Path(temporary).resolve()
        stage_snapshot(layout, stage)
        command = [sys.executable, '-B', str(stage/'reproduce.py'), '--no-compare',
                   '--artifact', figure, '--output-dir', str(output)]
        if args.statistics_only:
            command.append('--statistics-only')
        if args.figures_only:
            command.append('--figures-only')
        return subprocess.call(command, cwd=stage, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
