"""Checks for reader-facing paths and authenticated source-only staging."""
from __future__ import annotations

import importlib.util
import copy
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_reproduction', ROOT/'reproduce.py')
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class ReaderLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = reproduction.read_layout(ROOT)

    def test_all_17_artifacts_have_data_code_guide_and_builder(self):
        self.assertEqual(len(self.layout['artifacts']), 17)
        for artifact in self.layout['artifacts']:
            base = ROOT/artifact['directory']
            for name in ['README.md', 'build.py', 'data/README.md', 'data/index.csv', 'code/README.md']:
                self.assertTrue((base/name).is_file(), str(base/name))

    def test_friendly_names_map_to_original_artifacts(self):
        self.assertEqual(reproduction.resolve_figure('2', self.layout), 'main_figure_2')
        self.assertEqual(reproduction.resolve_figure('ed10', self.layout), 'extended_data_figure_10')
        self.assertEqual(reproduction.resolve_figure('table2', self.layout), 'supplementary_table_2')
        self.assertEqual(reproduction.resolve_figure('all', self.layout), 'all')
        with self.assertRaises(ValueError):
            reproduction.resolve_figure('ed99', self.layout)

    def test_catalog_paths_cannot_escape(self):
        with self.assertRaises(ValueError):
            reproduction.inside(ROOT, '../outside')
        with self.assertRaises(ValueError):
            reproduction.inside(ROOT, '.')

    def test_retained_files_match_authenticated_original_manifest(self):
        report = reproduction.verify_sources(self.layout, ROOT)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['original_source_paths'], len(self.layout['files']))
        self.assertEqual(report['retained_source_files'],
                         len({reproduction.inside(ROOT, row['repository_path']) for row in self.layout['files']}))

    def test_subset_does_not_require_unused_original_paths(self):
        layout = copy.deepcopy(self.layout)
        layout['files'] = [row for row in layout['files'] if row['snapshot_path'] != 'README.md']
        self.assertEqual(reproduction.verify_sources(layout, ROOT)['original_source_paths'], len(layout['files']))

    def test_catalog_rejects_duplicate_new_and_changed_source_metadata(self):
        variants = []
        layout = copy.deepcopy(self.layout)
        layout['files'].append(copy.deepcopy(layout['files'][0]))
        variants.append(layout)
        layout = copy.deepcopy(self.layout)
        layout['files'][0]['snapshot_path'] = 'unregistered_source.csv'
        variants.append(layout)
        for key, value in [('sha256', '0'*64), ('bytes', -1)]:
            layout = copy.deepcopy(self.layout)
            layout['files'][0][key] = value
            variants.append(layout)
        for layout in variants:
            with self.subTest(layout=layout['files'][0]['snapshot_path']):
                with self.assertRaises(ValueError):
                    reproduction.source_catalog(layout, ROOT)

    def test_catalog_requires_original_manifest_machinery_and_artifacts(self):
        for removed in ['RELEASE_MANIFEST.json', 'reproduce.py', 'recompute_statistics.py',
                        'shared/source_release_main_statistics.py']:
            layout = copy.deepcopy(self.layout)
            layout['files'] = [row for row in layout['files'] if row['snapshot_path'] != removed]
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                reproduction.source_catalog(layout, ROOT)
        layout = copy.deepcopy(self.layout)
        layout['snapshot_manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'manifest checksum changed'):
            reproduction.source_catalog(layout, ROOT)
        for change in ['missing', 'duplicate', 'outputs', 'configuration']:
            layout = copy.deepcopy(self.layout)
            if change == 'missing':
                layout['artifacts'].pop()
            elif change == 'duplicate':
                layout['artifacts'].append(copy.deepcopy(layout['artifacts'][0]))
            elif change == 'outputs':
                layout['artifacts'][0]['outputs'][0]['sha256'] = '0'*64
            else:
                layout['artifacts'][0]['strategy'] = 'replacement_engine'
            with self.subTest(change=change), self.assertRaises(ValueError):
                reproduction.source_catalog(layout, ROOT)

    def test_generated_guides_do_not_overwrite_frozen_files(self):
        frozen = {row['repository_path'].casefold() for row in self.layout['files']}
        for artifact in self.layout['artifacts']:
            for name in ['README.md', 'build.py', 'data/README.md', 'data/index.csv', 'code/README.md']:
                self.assertNotIn((artifact['directory']+'/'+name).casefold(), frozen)

    def test_subset_stage_preserves_sources_configs_and_passes_engine_hash_check(self):
        layout = copy.deepcopy(self.layout)
        layout['files'] = [row for row in layout['files'] if row['snapshot_path'] != 'README.md']
        original_path = ROOT/layout['snapshot_manifest']
        original_bytes = original_path.read_bytes()
        original = json.loads(original_bytes)
        with tempfile.TemporaryDirectory(prefix='maze_layout_test_') as temporary:
            stage = Path(temporary).resolve()
            staged = reproduction.stage_snapshot(layout, stage, ROOT)
            projected = {row['path'] for row in staged['staging']['checksum_inventory_projections']}
            for row in layout['files']:
                if row['snapshot_path'] != 'RELEASE_MANIFEST.json' and row['snapshot_path'] not in projected:
                    self.assertEqual(reproduction.checksum(stage/row['snapshot_path']), row['sha256'], row['snapshot_path'])
            self.assertEqual(staged['artifacts'], original['artifacts'])
            self.assertEqual({row['path'] for row in staged['files']},
                             {row['snapshot_path'] for row in layout['files']}-{'RELEASE_MANIFEST.json'})
            originals = {row['path']: row for row in original['files']}
            for row in staged['files']:
                if row['path'] not in projected:
                    self.assertEqual(row, originals[row['path']])
                self.assertEqual(row['sha256'], reproduction.checksum(stage/row['path']))
            self.assertFalse((stage/'README.md').exists())
            result = subprocess.run([sys.executable, '-B', str(stage/'reproduce.py'), '--no-compare', '--check-only'],
                                    cwd=stage, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            source = next(row['path'] for row in staged['files'] if row['path'].endswith('.csv'))
            (stage/source).write_bytes(b'changed staged data')
            result = subprocess.run([sys.executable, '-B', str(stage/'reproduce.py'), '--no-compare', '--check-only'],
                                    cwd=stage, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Release hash check failed', result.stdout+result.stderr)
        self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_only_inspected_inventory_entries_are_projected(self):
        with tempfile.TemporaryDirectory(prefix='maze_inventory_test_') as temporary:
            stage = Path(temporary).resolve()
            retained, originals, keep_rows = set(), {}, {}
            for relative, (kind, base) in reproduction.HASH_INVENTORIES.items():
                path = stage/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                target = path.parent/('source_data' if kind == 'files_frozen' else base)/'retained.csv'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'observed numerical input')
                retained.update([relative, target.relative_to(stage).as_posix()])
                row = {'sha256': reproduction.checksum(target), 'bytes': target.stat().st_size,
                       'authority': 'preserved original authority'}
                omitted = dict(row, sha256='0'*64)
                if kind == 'csv':
                    rows = [dict(row, path='retained.csv'), dict(omitted, path='omitted.csv')]
                    with path.open('w', encoding='utf-8', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                        writer.writeheader(); writer.writerows(rows)
                    keep_rows[relative] = rows[0]
                elif kind == 'frozen_files':
                    payload = [dict(row, frozen_file='retained.csv'), dict(omitted, frozen_file='omitted.csv')]
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    keep_rows[relative] = payload[0]
                else:
                    if kind == 'files_frozen':
                        entries = [dict(row, frozen='source_data\\retained.csv'),
                                   dict(omitted, frozen='source_data\\omitted.csv')]
                    elif kind == 'files':
                        entries = [dict(row, path='retained.csv'), dict(omitted, path='omitted.csv')]
                    else:
                        entries = {'retained.csv': row['sha256'] if kind == 'hashes' else row,
                                   'omitted.csv': omitted['sha256'] if kind == 'hashes' else omitted}
                    field = 'files' if kind == 'files_frozen' else kind
                    payload = entries if kind == 'hashes' else {field: entries, 'settings': {'estimator': 'untouched'}}
                    if kind == 'files_frozen':
                        payload.update(schema='ed8-standardized-pv-frozen-inputs/v1',
                                       rule_version='unchanged rule', accepted_record='unchanged current record')
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    keep_rows[relative] = entries[0] if kind in ('files', 'files_frozen') else entries['retained.csv']
                originals[relative] = path.read_bytes()
            untouched = stage/'uninspected/MANIFEST.json'
            untouched.parent.mkdir()
            untouched.write_text('{"outputs":{"omitted.csv":"metadata only"}}', encoding='utf-8')
            before = untouched.read_bytes()
            projections = reproduction.project_input_inventories(stage, retained)
            self.assertEqual(len(projections), len(reproduction.HASH_INVENTORIES))
            self.assertTrue(all(row['omitted_entries'] == 1 for row in projections))
            for relative, (kind, _) in reproduction.HASH_INVENTORIES.items():
                path = stage/relative
                self.assertNotEqual(path.read_bytes(), originals[relative])
                if kind == 'csv':
                    with path.open(encoding='utf-8', newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows, [{key: str(value) for key, value in keep_rows[relative].items()}])
                else:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    if kind == 'frozen_files':
                        self.assertEqual(payload, [keep_rows[relative]])
                    elif kind == 'hashes':
                        self.assertEqual(payload, {'retained.csv': keep_rows[relative]})
                    else:
                        self.assertEqual(payload['settings'], {'estimator': 'untouched'})
                        field = 'files' if kind == 'files_frozen' else kind
                        self.assertEqual(payload[field], [keep_rows[relative]] if kind in ('files', 'files_frozen')
                                         else {'retained.csv': keep_rows[relative]})
                        if kind == 'files_frozen':
                            self.assertEqual(payload['schema'], 'ed8-standardized-pv-frozen-inputs/v1')
                            self.assertEqual(payload['rule_version'], 'unchanged rule')
                            self.assertEqual(payload['accepted_record'], 'unchanged current record')
            self.assertEqual(untouched.read_bytes(), before)
            self.assertEqual(reproduction.project_input_inventories(stage, retained), [])

    def test_reader_markdown_links_resolve(self):
        pages = [ROOT/'README.md', ROOT/'reproduction/README.md', *list((ROOT/'docs').glob('*.md'))]
        for artifact in self.layout['artifacts']:
            pages += list((ROOT/artifact['directory']).rglob('*.md'))
        for page in pages:
            contents = page.read_text(encoding='utf-8')
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', contents):
                if link.startswith(('http:', 'https:', '#', 'mailto:')):
                    continue
                target = unquote(link.split('#', 1)[0].strip('<>'))
                self.assertTrue((page.parent/target).exists(), f'{page.relative_to(ROOT)} -> {link}')

    def test_single_figure_entry_point_checks_sources(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'figures/figure_02/build.py'), '--check-only'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
