"""Platform-independent checks for the browser discovery contract."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "browser_export", ROOT / "reproduction/runtime/ca_analysis/browser_export.py")
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)


class BrowserDiscoveryTests(unittest.TestCase):
    def test_override_is_used_before_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "browser"
            executable.touch()
            with patch.dict(browser.os.environ, {"MAZE_BROWSER": str(executable)}, clear=True), \
                    patch.object(browser.sys, "platform", "win32"):
                self.assertEqual(browser.find_browser(), executable.resolve())

    def test_invalid_override_does_not_silently_fall_back(self):
        with patch.dict(browser.os.environ, {"MAZE_BROWSER": "missing-browser"}, clear=True), \
                patch.object(browser.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "MAZE_BROWSER"):
                browser.find_browser()

    def test_linux_path_discovery(self):
        with patch.dict(browser.os.environ, {}, clear=True), \
                patch.object(browser.sys, "platform", "linux"), \
                patch.object(browser.shutil, "which", side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None), \
                patch.object(browser, "_executable", side_effect=lambda path: path.as_posix() == "/usr/bin/chromium"):
            # Some Linux distributions make chromium a symlink to chrome.
            self.assertEqual(browser.find_browser(), Path("/usr/bin/chromium").absolute())

    def test_macos_application_discovery(self):
        expected = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        with patch.dict(browser.os.environ, {}, clear=True), \
                patch.object(browser.sys, "platform", "darwin"), \
                patch.object(browser.Path, "home", return_value=Path("/test-home")), \
                patch.object(browser.shutil, "which", return_value=None), \
                patch.object(browser, "_executable", side_effect=lambda path: path == expected):
            self.assertEqual(browser.find_browser(), expected.resolve())

    def test_windows_preference_is_preserved(self):
        with patch.dict(browser.os.environ, {"PROGRAMFILES": "C:/Program Files"}, clear=True), \
                patch.object(browser.sys, "platform", "win32"), \
                patch.object(browser.shutil, "which", return_value=None), \
                patch.object(browser, "_executable", return_value=True):
            self.assertEqual(browser.find_browser(preferred=("edge", "chrome")).name, "msedge.exe")

    def test_missing_browser_has_installation_guidance(self):
        with patch.dict(browser.os.environ, {}, clear=True), \
                patch.object(browser.shutil, "which", return_value=None), \
                patch.object(browser, "_executable", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Install one"):
                browser.find_browser()

    def test_multiple_exporters_keep_preference_for_failure_fallback(self):
        with patch.dict(browser.os.environ, {}, clear=True), \
                patch.object(browser.shutil, "which", side_effect=lambda name:
                    "/test/" + name if name in ("google-chrome", "msedge") else None), \
                patch.object(browser, "_application_paths", return_value=[]), \
                patch.object(browser, "_executable", return_value=True):
            self.assertEqual(browser.find_browsers(preferred=("edge", "chrome")),
                             [Path('/test/msedge').resolve(), Path('/test/google-chrome').resolve()])

    def test_macos_flags_do_not_change_windows_export_commands(self):
        with patch.object(browser.sys, 'platform', 'win32'):
            self.assertEqual(browser.platform_headless_flags(), [])
        with patch.object(browser.sys, 'platform', 'darwin'):
            self.assertIn('--use-mock-keychain', browser.platform_headless_flags())

    def test_confined_override_is_rejected(self):
        with patch.dict(browser.os.environ, {'MAZE_BROWSER': '/snap/bin/chromium'}, clear=True), \
                patch.object(browser.shutil, 'which', return_value=None), \
                patch.object(browser, '_executable', return_value=True):
            with self.assertRaisesRegex(ValueError, 'native browser package'):
                browser.find_browser()

    def test_discovery_preserves_launcher_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'native-browser'
            target.touch()
            launcher = Path(temporary) / 'chromium'
            try:
                launcher.symlink_to(target)
            except OSError:
                self.skipTest('Creating symlinks requires permission on this host')
            with patch.dict(browser.os.environ, {'MAZE_BROWSER': str(launcher)}, clear=True), \
                    patch.object(browser, '_executable', return_value=True):
                self.assertEqual(browser.find_browser(), launcher.absolute())

    def test_timeout_preserves_diagnostics(self):
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            browser.run_browser([sys.executable, '-c',
                                 'import sys,time; print("started",file=sys.stderr,flush=True); time.sleep(3)'],
                                timeout=0.5)
        self.assertIn(b'started', caught.exception.stderr)

    def test_export_does_not_wait_for_inherited_diagnostic_handles(self):
        # The parent exits immediately; its background child holds stderr open.
        command = [sys.executable, '-c',
                   'import subprocess,sys; '
                   'subprocess.Popen([sys.executable,"-c","import time; time.sleep(2)"]); '
                   'print("export complete",file=sys.stderr)']
        result = browser.run_browser(command, check=True, text=True, timeout=1.5)
        self.assertEqual(result.returncode, 0)
        self.assertIn('export complete', result.stderr)

    def test_failed_export_is_not_accepted(self):
        with self.assertRaises(subprocess.CalledProcessError):
            browser.run_browser([sys.executable, '-c', 'raise SystemExit(2)'], check=True)


if __name__ == "__main__":
    unittest.main()
