"""Platform-independent checks for the browser discovery contract."""
import importlib.util
from pathlib import Path
import tempfile
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
            self.assertEqual(browser.find_browser(), Path("/usr/bin/chromium").resolve())

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


if __name__ == "__main__":
    unittest.main()
