"""Export the Figure 5 source-rendered SVG through Chrome or Edge.

rebuild_figure5 assembles the panels and passes its SVG to this helper.
The historical filename identifies the frozen export implementation.
"""

from __future__ import annotations

import os

import subprocess

import tempfile

import time

from pathlib import Path

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

CHROME_CANDIDATES = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
)

BROWSER_EXPORT_TIMEOUT_SECONDS = 180

SVG_NS = "http://www.w3.org/2000/svg"

XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)

ET.register_namespace("xlink", XLINK_NS)

def _find_chrome_executable() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find Chrome/Edge for SVG export. Checked: "
        + ", ".join(str(path) for path in CHROME_CANDIDATES)
    )

def _write_svg_browser_wrapper(svg_path: Path, html_path: Path, *, width_mm: float, height_mm: float) -> None:
    svg_text = svg_path.read_text(encoding="utf-8")
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: {width_mm:.6f}mm {height_mm:.6f}mm; margin: 0; }}
html, body {{ width: {width_mm:.6f}mm; height: {height_mm:.6f}mm; margin: 0; padding: 0; background: white; overflow: hidden; }}
svg {{ display: block; width: {width_mm:.6f}mm !important; height: {height_mm:.6f}mm !important; }}
* {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
</style>
</head>
<body>
{svg_text}
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8", newline="\n")

def _export_svg_with_chrome(
    svg_path: Path,
    png_path: Path,
    pdf_path: Path,
    *,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> Path:
    browsers = [path for path in CHROME_CANDIDATES if path.exists()]
    if not browsers:
        _find_chrome_executable()  # Raise the maintained, path-specific error.
        raise AssertionError("unreachable")
    html_path = svg_path.with_suffix(".html")
    _write_svg_browser_wrapper(svg_path, html_path, width_mm=width_mm, height_mm=height_mm)
    css_width_px = max(1, int(round(width_mm / 25.4 * 96.0)))
    css_height_px = max(1, int(round(height_mm / 25.4 * 96.0)))
    device_scale = max(0.1, float(dpi) / 96.0)
    errors: list[str] = []
    for browser in browsers:
        tmp_png = png_path.with_name(
            f".{png_path.stem}.{browser.stem}.tmp{png_path.suffix}"
        )
        tmp_pdf = pdf_path.with_name(
            f".{pdf_path.stem}.{browser.stem}.tmp{pdf_path.suffix}"
        )
        tmp_png.unlink(missing_ok=True)
        tmp_pdf.unlink(missing_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"figure_export_{browser.stem}_") as profile_dir:
                common = [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=1000",
                    f"--user-data-dir={profile_dir}",
                ]
                screenshot_cmd = [
                    *common,
                    f"--window-size={css_width_px},{css_height_px}",
                    f"--force-device-scale-factor={device_scale:.6f}",
                    f"--screenshot={tmp_png.resolve()}",
                    html_path.resolve().as_uri(),
                ]
                subprocess.run(
                    screenshot_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
                )
                _wait_for_browser_output(tmp_png)
                pdf_cmd = [
                    *common,
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={tmp_pdf.resolve()}",
                    html_path.resolve().as_uri(),
                ]
                subprocess.run(
                    pdf_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
                )
                _wait_for_browser_output(tmp_pdf)
            if not tmp_png.exists() or not tmp_pdf.exists():
                raise RuntimeError("browser exited without both PNG and PDF outputs")
            os.replace(tmp_png, png_path)
            os.replace(tmp_pdf, pdf_path)
            return html_path
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"{browser}: {type(exc).__name__}: {exc}")
            tmp_png.unlink(missing_ok=True)
            tmp_pdf.unlink(missing_ok=True)
    raise RuntimeError(
        "All available Chrome/Edge exporters failed:\n" + "\n".join(errors)
    )

def _wait_for_browser_output(path: Path, *, timeout_seconds: float = 15.0) -> None:
    """Wait for Chromium's detached writer to finish the requested export.

    Current Windows Chromium builds can return from the headless command before
    the screenshot/PDF child process has created its output.  Requiring a
    non-empty, stable file keeps the temporary browser profile alive until the
    asynchronous write is complete.
    """
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable_polls = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1
        if size > 0 and size == previous_size:
            stable_polls += 1
            if stable_polls >= 3:
                return
        else:
            stable_polls = 0
        previous_size = size
        time.sleep(0.1)
    raise RuntimeError(f"browser output did not settle within {timeout_seconds:g} s: {path}")
