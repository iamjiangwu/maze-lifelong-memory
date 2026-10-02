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
import sys


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
    # common package/Snap locations for minimal shell environments as well.
    return [Path(root) / command for root in ("/usr/bin", "/usr/local/bin", "/snap/bin")
            for command in _COMMANDS[browser] if not command.endswith(".exe")]


def _executable(path: Path) -> bool:
    return path.is_file() and (sys.platform == "win32" or os.access(path, os.X_OK))


def find_browsers(preferred: tuple[str, ...] = ("chrome", "edge", "chromium")) -> list[Path]:
    """Resolve installed exporters in preference order, or only MAZE_BROWSER.

    An invalid override fails clearly rather than silently using another browser.
    The preference order lets established renderers retain their export engine.
    """
    override = os.environ.get("MAZE_BROWSER", "").strip().strip('"')
    if override:
        candidate = Path(shutil.which(override) or override).expanduser()
        if _executable(candidate):
            return [candidate.resolve()]
        raise FileNotFoundError("MAZE_BROWSER must name an executable Chrome, Chromium or Edge browser.")
    found = []
    for browser in dict.fromkeys((*preferred, *_COMMANDS)):
        if browser not in _COMMANDS:
            raise ValueError(f"Unknown browser preference: {browser}")
        candidates = [Path(location) for command in _COMMANDS[browser]
                      if (location := shutil.which(command))]
        for candidate in [*candidates, *_application_paths(browser)]:
            if _executable(candidate):
                resolved = candidate.resolve()
                if resolved not in found:
                    found.append(resolved)
                break
    if found:
        return found
    raise FileNotFoundError(
        "Figure PDF export requires Google Chrome, Chromium or Microsoft Edge. "
        "Install one, place its executable on PATH, or set MAZE_BROWSER to its executable path."
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
