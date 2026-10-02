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
import socket
import subprocess
import sys
import tempfile
import time


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


def _run_macos_export(command, diagnostics, log_path, timeout):
    """Request a normal browser shutdown once the CLI reports its saved export.

    Some macOS builds stay in the application event loop after printing. A
    private DevTools pipe lets us send Browser.close to only this disposable
    instance, without a network port, user profile, or process-name kill.
    Successful process exit and the caller's PDF/PNG checks are still required.
    """
    exports = [Path(str(arg).split('=', 1)[1]) for arg in command
               if str(arg).startswith(('--print-to-pdf=', '--screenshot='))]
    if len(exports) != 1 or not any(str(arg).startswith('--user-data-dir=') for arg in command):
        raise ValueError('macOS export requires one output and a disposable browser profile')
    if exports[0].exists():
        raise FileExistsError('Use a fresh browser export path: ' + str(exports[0]))
    # Chromium reads CDP on descriptor 3 and writes on 4. A socket pair carries
    # the same null-terminated messages in both directions. The tiny exec
    # wrapper maps only inherited descriptors; it becomes the browser process.
    setup = ('import os,sys,fcntl; '
             'fd=fcntl.fcntl(int(sys.argv[1]),fcntl.F_DUPFD,5); '
             'os.dup2(fd,3); os.dup2(fd,4); os.close(fd); '
             'os.execv(sys.argv[2],sys.argv[2:])')
    parent, child = socket.socketpair()
    with parent, child:
        launched = [sys.executable, '-c', setup, str(child.fileno()),
                    str(command[0]), '--remote-debugging-pipe', *command[1:]]
        process = subprocess.Popen(launched, stdout=diagnostics, stderr=subprocess.STDOUT,
                                   pass_fds=(child.fileno(),))
        child.close()
        deadline = time.monotonic() + timeout
        close_sent = False
        marker = b' bytes written to file ' + os.fsencode(exports[0])
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                if not close_sent and marker in log_path.read_bytes() and exports[0].is_file():
                    try:
                        parent.sendall(b'{"id":1,"method":"Browser.close"}\0')
                    except BrokenPipeError:
                        # A browser that has already exited needs no shutdown
                        # request. Its actual exit code is checked below.
                        if process.poll() is None:
                            raise
                    close_sent = True
                time.sleep(0.1)
            return subprocess.CompletedProcess(command, process.returncode)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def run_browser(command, *, check=False, capture_output=True, text=False, timeout=180):
    """Wait for the exporter, not background updaters that inherit its stderr.

    Capture combined diagnostics in a temporary file so inherited updater pipes
    cannot hold a completed export open. macOS receives Browser.close after the
    CLI reports the saved file. Normal process exit, timeout and caller checks
    of the produced PDF/PNG remain mandatory. The log is not a source input.
    """
    if not capture_output:
        raise ValueError('Browser diagnostics must be captured')
    log_factory = tempfile.NamedTemporaryFile if sys.platform == 'darwin' else tempfile.TemporaryFile
    with log_factory() as diagnostics:
        try:
            if sys.platform == 'darwin' and any(str(arg).startswith(('--print-to-pdf=', '--screenshot=')) for arg in command):
                completed = _run_macos_export(command, diagnostics, Path(diagnostics.name), timeout)
            else:
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
