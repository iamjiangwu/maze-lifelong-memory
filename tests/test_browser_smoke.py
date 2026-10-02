"""The export check uses fallback browsers only when the preferred one fails."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import fitz

spec = importlib.util.spec_from_file_location('browser_smoke', Path(__file__).with_name('browser_smoke.py'))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ExportSelectionTests(unittest.TestCase):
    def run_check(self, successful_browser):
        calls = []

        def export(command, **kwargs):
            name = str(command[0])
            calls.append(name)
            if name == successful_browser:
                pdf = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--print-to-pdf='))
                with fitz.open() as document:
                    document.new_page().insert_text((50, 50), 'Maze export check')
                    document.save(pdf)
                return subprocess.CompletedProcess(command, 0, stderr=b'')
            return subprocess.CompletedProcess(command, 1, stderr=b'failed')

        with patch.dict(smoke.browser, {
            'find_browsers': lambda: [Path('preferred'), Path('fallback'), Path('unused')],
            'platform_headless_flags': lambda: [],
            'run_browser': export,
        }), contextlib.redirect_stdout(io.StringIO()):
            smoke.main()
        return calls

    def test_stop_after_preferred_success(self):
        self.assertEqual(self.run_check('preferred'), ['preferred'])

    def test_try_fallback_after_failure(self):
        self.assertEqual(self.run_check('fallback'), ['preferred', 'fallback'])

    def test_all_failures_are_reported(self):
        with self.assertRaises(AssertionError):
            self.run_check(None)


if __name__ == '__main__':
    unittest.main()
