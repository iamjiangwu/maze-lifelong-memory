"""Checks for reader-facing paths and preservation of the frozen snapshot."""
from __future__ import annotations

import importlib.util
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

    def test_frozen_files_match_complete_original_manifest(self):
        report = reproduction.verify_sources(self.layout, ROOT)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['original_source_paths'], 970)
        self.assertEqual(report['retained_source_files'], 793)

    def test_generated_guides_do_not_overwrite_frozen_files(self):
        frozen = {row['repository_path'].casefold() for row in self.layout['files']}
        for artifact in self.layout['artifacts']:
            for name in ['README.md', 'build.py', 'data/README.md', 'data/index.csv', 'code/README.md']:
                self.assertNotIn((artifact['directory']+'/'+name).casefold(), frozen)

    def test_reconstructed_snapshot_is_byte_identical(self):
        with tempfile.TemporaryDirectory(prefix='maze_layout_test_') as temporary:
            stage = Path(temporary).resolve()
            reproduction.stage_snapshot(self.layout, stage, ROOT)
            for row in self.layout['files']:
                self.assertEqual(reproduction.checksum(stage/row['snapshot_path']), row['sha256'], row['snapshot_path'])
            result = subprocess.run([sys.executable, '-B', str(stage/'reproduce.py'), '--no-compare', '--check-only'],
                                    cwd=stage, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

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
