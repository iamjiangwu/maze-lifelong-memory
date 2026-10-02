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


def find_browser(preferred: tuple[str, ...] = ("chrome", "edge", "chromium")) -> Path:
    """Resolve an export browser, honoring an explicit MAZE_BROWSER first.

    An invalid override fails clearly rather than silently using another browser.
    The preference order lets established renderers retain their export engine.
    """
    override = os.environ.get("MAZE_BROWSER", "").strip().strip('"')
    if override:
        candidate = Path(shutil.which(override) or override).expanduser()
        if _executable(candidate):
            return candidate.resolve()
        raise FileNotFoundError("MAZE_BROWSER must name an executable Chrome, Chromium or Edge browser.")
    for browser in dict.fromkeys((*preferred, *_COMMANDS)):
        if browser not in _COMMANDS:
            raise ValueError(f"Unknown browser preference: {browser}")
        for command in _COMMANDS[browser]:
            if (location := shutil.which(command)) and _executable(Path(location)):
                return Path(location).resolve()
        for candidate in _application_paths(browser):
            if _executable(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        "Figure PDF export requires Google Chrome, Chromium or Microsoft Edge. "
        "Install one, place its executable on PATH, or set MAZE_BROWSER to its executable path."
    )
