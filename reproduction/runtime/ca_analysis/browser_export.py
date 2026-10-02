"""Find an installed Chromium browser for local SVG/HTML-to-PDF export.

Chrome, Chromium and Edge share the headless printing interface used by the
figure renderers. MAZE_BROWSER can select a particular executable without
changing source code; otherwise discovery uses PATH and standard app folders.
No browser is downloaded or installed by the reproduction scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_COMMANDS = {
    "chrome": ("google-chrome", "google-chrome-stable", "chrome", "chrome.exe"),
    "edge": ("microsoft-edge", "microsoft-edge-stable", "msedge", "msedge.exe"),
    "chromium": ("chromium", "chromium-browser"),
}


def _application_paths(browser: str) -> list[Path]:
    """Return conventional installation locations for the current platform."""
    if sys.platform == "win32":
        suffix = {
            "chrome": Path("Google/Chrome/Application/chrome.exe"),
            "edge": Path("Microsoft/Edge/Application/msedge.exe"),
            "chromium": Path("Chromium/Application/chrome.exe"),
        }[browser]
        # Keep existing Windows browser ordering while accepting per-user installs.
        return [Path(root) / suffix for key in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")
                if (root := os.environ.get(key))]
    if sys.platform == "darwin":
        app, executable = {
            "chrome": ("Google Chrome", "Google Chrome"),
            "edge": ("Microsoft Edge", "Microsoft Edge"),
            "chromium": ("Chromium", "Chromium"),
        }[browser]
        suffix = Path(f"{app}.app/Contents/MacOS/{executable}")
        return [Path("/Applications") / suffix, Path.home() / "Applications" / suffix]
    # Distribution packages normally expose an executable on PATH. Include
    # Common native-package locations; confined Snap/Flatpak builds cannot read
    # the disposable export folders used here.
    return [Path(root) / command for root in ("/usr/bin", "/usr/local/bin")
            for command in _COMMANDS[browser] if not command.endswith(".exe")]


def _executable(path: Path) -> bool:
    return path.is_file() and (sys.platform == "win32" or os.access(path, os.X_OK))


def _confined_browser(path: Path) -> bool:
    """Recognize package launchers whose confinement hides temporary inputs."""
    resolved = path.resolve()
    return (any(part in ('snap', '.flatpak', 'flatpak') for part in path.parts)
            or any(part in ('snap', '.flatpak', 'flatpak') for part in resolved.parts)
            or resolved.name in ('snap', 'flatpak'))


def find_browsers(preferred: tuple[str, ...] = ("chrome", "edge", "chromium")) -> list[Path]:
    """Resolve installed exporters in preference order, or only MAZE_BROWSER.

    An invalid override fails clearly rather than silently using another browser.
    The preference order lets established renderers retain their export engine.
    """
    override = os.environ.get("MAZE_BROWSER", "").strip().strip('"')
    if override:
        candidate = Path(shutil.which(override) or override).expanduser()
        if _executable(candidate):
            if _confined_browser(candidate):
                raise ValueError('MAZE_BROWSER selects a Snap/Flatpak browser; install a native browser package instead.')
            return [candidate.absolute()]
        raise FileNotFoundError("MAZE_BROWSER must name an executable Chrome, Chromium or Edge browser.")
    found, resolved_paths = [], set()
    for browser in dict.fromkeys((*preferred, *_COMMANDS)):
        if browser not in _COMMANDS:
            raise ValueError(f"Unknown browser preference: {browser}")
        candidates = [Path(location) for command in _COMMANDS[browser]
                      if (location := shutil.which(command))]
        for candidate in [*candidates, *_application_paths(browser)]:
            if _executable(candidate) and not _confined_browser(candidate):
                resolved = candidate.resolve()
                if resolved not in resolved_paths:
                    # Preserve launcher identity; resolving a symlink can change
                    # argv[0] or bypass a package's executable wrapper.
                    found.append(candidate.absolute())
                    resolved_paths.add(resolved)
                break
    if found:
        return found
    raise FileNotFoundError(
        "Figure PDF export requires Google Chrome, Chromium or Microsoft Edge. "
        "Install one as a native package (not Snap/Flatpak), place its executable on PATH, "
        "or set MAZE_BROWSER to its executable path."
    )


def find_browser(preferred: tuple[str, ...] = ("chrome", "edge", "chromium")) -> Path:
    """Return the preferred installed exporter for single-browser callers."""
    return find_browsers(preferred)[0]


def platform_headless_flags() -> list[str]:
    """Avoid interactive macOS keychain/first-run dialogs in disposable profiles.

    Chromium's own test launcher uses the mock keychain to prevent blocking
    permissions dialogs. These flags never access an existing browser profile:
    https://github.com/GoogleChrome/chrome-launcher/blob/main/docs/chrome-flags-for-tools.md
    """
    if sys.platform == 'darwin':
        return ['--use-mock-keychain', '--password-store=basic',
                '--no-first-run', '--no-default-browser-check']
    return []


def run_browser(command, *, check=False, capture_output=True, text=False, timeout=180):
    """Wait for the exporter, not background updaters that inherit its stderr.

    On macOS a browser's updater can retain a pipe after the exporting process
    exits. Capture combined diagnostics in a temporary file instead: process
    exit, timeout and caller checks of the produced PDF/PNG remain mandatory.
    Only this explicitly launched process is waited for; no other browser or
    updater is terminated. The temporary log is not a published source input.
    """
    if not capture_output:
        raise ValueError('Browser diagnostics must be captured')
    with tempfile.TemporaryFile() as diagnostics:
        try:
            completed = subprocess.run(command, stdout=diagnostics, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            diagnostics.seek(0)
            error.stderr = diagnostics.read()
            raise
        diagnostics.seek(0)
        logs = diagnostics.read()
    if text:
        logs = logs.decode('utf-8', errors='replace')
    result = subprocess.CompletedProcess(command, completed.returncode,
                                         stdout='' if text else b'', stderr=logs)
    if check:
        result.check_returncode()
    return result
