"""Small plotting primitives shared by publication figure notebooks.

Three groups of helpers here keep the figure files portable and repeatable:

- :func:`figure_font_family` chooses the figure font: Arial, the accepted
  typography, when it is installed, otherwise a substitute (see its notes);
- :func:`save_figure` and :func:`savefig_reproducibly` are the only places
  that write Matplotlib figures, so every SVG and PDF is byte-identical from
  run to run on the same computer;
- :func:`chromium_export_browser` finds Google Chrome, Chromium or Microsoft
  Edge for the two pages that a browser prints (Figures 2 and 3); without one
  those page exports are skipped with a message.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont
import pymupdf
from PIL import Image

from ._validation import positive_integer, strict_boolean


PUBLICATION_FONT_FAMILY = "DejaVu Sans"
_PUBLICATION_FONT_FILENAME = "DejaVuSans.ttf"
_PROTECTED_FONT_KEYS = frozenset({"font.family", "font.sans-serif", "mathtext.fontset"})
VERSION_RECORD_FONT_FAMILY = "Arial"
"""Font family of the accepted publication figures."""

ARIAL_METRIC_SUBSTITUTES: tuple[str, ...] = ("Liberation Sans", "Arimo")
"""Free fonts with Arial's character widths, tried in this order without Arial."""

LAST_RESORT_FONT_FAMILY = "DejaVu Sans"
"""Font used when neither Arial nor a substitute is installed; ships with Matplotlib."""

HIDE_FONTS_VARIABLE = "MAZE_LIFELONG_MEMORY_HIDE_FONTS"
"""Testing only: comma-separated font families to treat as not installed."""

BROWSER_VARIABLE = "MAZE_LIFELONG_MEMORY_BROWSER"
"""Browser executable for the page exports, or ``none`` to skip them."""

SVG_HASH_SALT = "maze-lifelong-memory"
"""Fixed salt for Matplotlib's SVG element ids, so the ids repeat run to run."""

_FIXED_PDF_DATE = b"20000101000000"
_PDF_DATE_PATTERN = re.compile(rb"/(CreationDate|ModDate)\s*\(D:(\d{14})")
_MATPLOTLIB_SVG_DOCTYPE = re.compile(
    r'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1\.1//EN"\s*'
    r'"http://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd">\s*',
    flags=re.IGNORECASE,
)

PUBLICATION_RC: dict[str, object] = {
    "axes.linewidth": 0.8,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "font.family": [PUBLICATION_FONT_FAMILY],
    "font.sans-serif": [PUBLICATION_FONT_FAMILY],
    "font.size": 7,
    "legend.fontsize": 6,
    "lines.linewidth": 1.0,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
    "svg.fonttype": "none",
    "xtick.labelsize": 6,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.labelsize": 6,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.7,
}
_VECTOR_EXPORT_RC = {
    key: PUBLICATION_RC[key]
    for key in ("pdf.fonttype", "ps.fonttype", "svg.fonttype")
}


@contextmanager
def publication_style(overrides: Mapping[str, object] | None = None) -> Iterator[None]:
    """Apply the shared journal style inside a local matplotlib context."""

    resolved_publication_font()
    settings = dict(PUBLICATION_RC)
    if overrides is not None:
        protected = sorted(_PROTECTED_FONT_KEYS.intersection(overrides))
        if protected:
            raise ValueError(
                "Publication font settings are fixed and cannot be overridden: "
                f"{protected}"
            )
        settings.update(overrides)
    with mpl.rc_context(settings):
        yield


def font_is_installed(family: str) -> bool:
    """Return whether Matplotlib knows an installed font file of ``family``."""

    wanted = family.casefold()
    return any(
        entry.name.casefold() == wanted and os.path.isfile(entry.fname)
        for entry in font_manager.fontManager.ttflist
    )


def choose_figure_font_family(
    *,
    installed: Callable[[str], bool] | None = None,
    hidden: Iterable[str] = (),
) -> str:
    """Return the first available family of Arial, its substitutes, DejaVu Sans.

    ``installed`` tells whether a family is installed (default
    :func:`font_is_installed`); ``hidden`` names families to treat as not
    installed.  DejaVu Sans ships with Matplotlib, so it is returned even when
    it is hidden.
    """

    installed = font_is_installed if installed is None else installed
    hidden_names = {name.strip().casefold() for name in hidden}
    for family in (VERSION_RECORD_FONT_FAMILY, *ARIAL_METRIC_SUBSTITUTES):
        if family.casefold() not in hidden_names and installed(family):
            return family
    return LAST_RESORT_FONT_FAMILY


def figure_font_notice(family: str, *, hidden: Iterable[str] = ()) -> str:
    """Explain in one paragraph why the figures use ``family`` instead of Arial."""

    hidden_names = {name.strip().casefold() for name in hidden}

    def missing(name: str) -> str:
        if name.casefold() in hidden_names:
            return f"{name} is hidden by {HIDE_FONTS_VARIABLE}"
        return f"{name} is not installed"

    if family in ARIAL_METRIC_SUBSTITUTES:
        return (
            f"Font notice: {missing(VERSION_RECORD_FONT_FAMILY)}, so the figures "
            f"use {family}. It has the same character widths as Arial, so text "
            "takes the same space as in the published figures, but the letter "
            "shapes differ slightly. Calculated values are unchanged."
        )
    reasons = "; ".join(
        missing(name) for name in (VERSION_RECORD_FONT_FAMILY, *ARIAL_METRIC_SUBSTITUTES)
    )
    return (
        f"Font notice: {reasons}. The figures therefore use {family}, which ships "
        "with Matplotlib. It is wider than Arial, so some labels sit or fit "
        "differently from the published figures. Calculated values are "
        "unchanged. Install Arial or Liberation Sans to match the published "
        "typography."
    )


def _hidden_font_families() -> tuple[str, ...]:
    """Read the testing-only list of font families to treat as missing."""

    value = os.environ.get(HIDE_FONTS_VARIABLE, "")
    return tuple(sorted({part.strip() for part in value.split(",") if part.strip()}))


@functools.lru_cache(maxsize=None)
def _resolved_figure_font(hidden: tuple[str, ...]) -> str:
    """Choose the figure font once per setting and print any fallback notice."""

    family = choose_figure_font_family(hidden=hidden)
    if family != VERSION_RECORD_FONT_FAMILY:
        print(figure_font_notice(family, hidden=hidden), flush=True)
    return family


def figure_font_family() -> str:
    """Return the font family for figure text.

    The accepted figures use Arial.  Without Arial the figures use the first
    installed of Liberation Sans and Arimo, which have Arial's character
    widths, and otherwise DejaVu Sans.  A fallback prints one notice per
    process and never stops a notebook.  For testing, the environment
    variable ``MAZE_LIFELONG_MEMORY_HIDE_FONTS`` (for example ``Arial``)
    treats the listed families as not installed.
    """

    return _resolved_figure_font(_hidden_font_families())


def with_figure_font(settings: Mapping[str, object]) -> dict[str, object]:
    """Return Matplotlib settings with Arial replaced by :func:`figure_font_family`.

    Figure styles name Arial, the accepted typography, in ``font.*`` and
    ``mathtext.*`` settings (``"Arial"``, ``"Arial:bold"`` or a list of
    families).  When Arial is installed the settings are returned unchanged.
    """

    family = figure_font_family()
    resolved = dict(settings)
    if family == VERSION_RECORD_FONT_FAMILY:
        return resolved
    accepted = VERSION_RECORD_FONT_FAMILY

    def substitute(value: object) -> object:
        if isinstance(value, str):
            if value == accepted or value.startswith(f"{accepted}:"):
                return family + value[len(accepted):]
            return value
        if isinstance(value, (list, tuple)):
            families: list[object] = []
            for item in value:
                item = substitute(item)
                if item not in families:
                    families.append(item)
            return families
        return value

    for key, value in resolved.items():
        if key.startswith(("font.", "mathtext.")):
            resolved[key] = substitute(value)
    return resolved


@contextmanager
def version_record_style(
    overrides: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Apply the typography used by the accepted version of record figures.

    The text uses :func:`figure_font_family`: Arial when it is installed.
    """

    family = figure_font_family()
    settings = dict(PUBLICATION_RC)
    settings.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family],
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
        }
    )
    if overrides is not None:
        protected = sorted(_PROTECTED_FONT_KEYS.intersection(overrides))
        if protected:
            raise ValueError(
                "Version of record font settings are fixed and cannot be "
                f"overridden: {protected}"
            )
        settings.update(overrides)
    with mpl.rc_context(settings):
        yield


@contextmanager
def vector_export_style() -> Iterator[None]:
    """Keep text editable when Matplotlib writes vector output."""

    with mpl.rc_context(_VECTOR_EXPORT_RC):
        yield


def resolved_publication_font() -> Path:
    """Return the exact DejaVu Sans TTF shipped by this Matplotlib install.

    The direct package-data path avoids an operating-system font search and
    the explicit resolver check prevents a silent substitution by Matplotlib.
    """

    packaged = (
        Path(mpl.get_data_path()) / "fonts" / "ttf" / _PUBLICATION_FONT_FILENAME
    ).resolve()
    if not packaged.is_file():
        raise RuntimeError(f"Matplotlib publication font is missing: {packaged}")
    family = FontProperties(fname=str(packaged)).get_name()
    if family != PUBLICATION_FONT_FAMILY:
        raise RuntimeError(
            "Matplotlib publication font has an unexpected family: " f"{family!r}"
        )
    selected = Path(
        findfont(
            FontProperties(family=[PUBLICATION_FONT_FAMILY]),
            fallback_to_default=False,
        )
    ).resolve()
    if selected != packaged:
        raise RuntimeError(
            "Matplotlib selected a non-packaged DejaVu Sans font; expected "
            f"{packaged.name}, found {selected.name}"
        )
    return packaged


def despine(axis: Axes, *, keep: Sequence[str] = ("left", "bottom")) -> None:
    """Hide axis spines except those named in ``keep``."""

    keep_set = set(keep)
    unknown = keep_set.difference(axis.spines)
    if unknown:
        raise ValueError(f"Unknown spine names: {sorted(unknown)}")
    for name, spine in axis.spines.items():
        spine.set_visible(name in keep_set)
    axis.tick_params(
        top="top" in keep_set,
        right="right" in keep_set,
        bottom="bottom" in keep_set,
        left="left" in keep_set,
    )


def draw_water_port_icon(axis: Axes, *, y_axes: float) -> None:
    """Draw a 12 point water icon at the right edge, at an axes-relative height."""

    size = 12.0
    drawing = DrawingArea(size, size, 0, 0)
    center = size / 2.0
    scale = size * 0.74
    vertices = (
        (center, center + 0.66 * scale),
        (center - 0.60 * scale, center + 0.05 * scale),
        (center - 0.48 * scale, center - 0.52 * scale),
        (center, center - 0.58 * scale),
        (center + 0.48 * scale, center - 0.52 * scale),
        (center + 0.60 * scale, center + 0.05 * scale),
        (center, center + 0.66 * scale),
        (center, center + 0.66 * scale),
    )
    codes = (
        MatplotlibPath.MOVETO,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CLOSEPOLY,
    )
    drawing.add_artist(
        PathPatch(
            MatplotlibPath(vertices, codes),
            facecolor="#36a3ff",
            edgecolor="#0866b2",
            linewidth=0.65,
            joinstyle="round",
        )
    )
    drawing.add_artist(
        Ellipse(
            (center - 0.13 * scale, center + 0.02 * scale),
            width=0.15 * scale,
            height=0.30 * scale,
            angle=28,
            facecolor="#c9ecff",
            edgecolor="none",
            alpha=0.9,
        )
    )
    axis.add_artist(
        AnnotationBbox(
            drawing,
            (1.0, y_axes),
            xycoords=axis.transAxes,
            xybox=(6.0, 0.0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
            annotation_clip=False,
            zorder=20,
        )
    )


def _figure_axes(figure: Figure | SubFigure) -> Iterator[Axes]:
    """Yield the axes of a figure and of all its subfigures."""

    yield from figure.axes
    for subfigure in getattr(figure, "subfigs", ()):
        yield from _figure_axes(subfigure)


def _match_new_ticks_to_existing(figure: Figure) -> None:
    """Give ticks that Matplotlib creates while saving the existing ticks' geometry.

    Matplotlib creates tick objects lazily.  A panel is styled inside its
    style context, but Matplotlib often adds the remaining ticks only when the
    figure is saved, after that context has closed, and those ticks take the
    global tick length, width and padding.  The drawn tick marks still copy
    the first tick, so they look identical.  The layout code, however,
    measures the room for tick marks from one tick that it picks in
    memory-address order.  With mixed tick lengths a panel that uses
    constrained layout or a tight bounding box then moved by a fraction of a
    point between runs.  Recording the first tick's geometry as the axis
    default before saving makes every new tick agree with the existing ones.
    Explicit ``tick_params`` settings already stored on the axis are kept.
    """

    for axes in _figure_axes(figure):
        axis_map = getattr(axes, "_axis_map", None)
        if axis_map is None:
            axis_map = {"x": axes.xaxis, "y": axes.yaxis}
        for axis in axis_map.values():
            for attribute, keywords_name in (
                ("majorTicks", "_major_tick_kw"),
                ("minorTicks", "_minor_tick_kw"),
            ):
                # Read the instance dictionary so that an unused lazy tick
                # list is not created here under the current settings.
                ticks = vars(axis).get(attribute)
                keywords = getattr(axis, keywords_name, None)
                if not ticks or keywords is None:
                    continue
                template = ticks[0]
                for keyword, value in (
                    ("size", getattr(template, "_size", None)),
                    ("width", getattr(template, "_width", None)),
                    ("pad", getattr(template, "_base_pad", None)),
                    ("tickdir", getattr(template, "_tickdir", None)),
                ):
                    if value is not None:
                        keywords.setdefault(keyword, value)


def savefig_reproducibly(
    figure: Figure,
    target: str | os.PathLike[str] | IO[bytes] | IO[str],
    *,
    format: str | None = None,
    **savefig_options: object,
) -> None:
    """Write ``figure`` with :meth:`~matplotlib.figure.Figure.savefig` repeatably.

    Every Matplotlib figure file of the package is written here, directly or
    through :func:`save_figure`.  The options are passed on unchanged, with
    three additions that do not change how the figure looks:

    - PDF files omit the creation date and SVG files the date;
    - SVG element ids use the fixed salt :data:`SVG_HASH_SALT` instead of a
      random one;
    - ticks that Matplotlib creates while saving get the geometry of the
      axis's existing ticks (see :func:`_match_new_ticks_to_existing`).

    PNG and TIFF files keep Matplotlib's default content.
    """

    fmt = format
    if fmt is None:
        if isinstance(target, (str, os.PathLike)):
            fmt = Path(target).suffix.lstrip(".") or None
        fmt = fmt or str(mpl.rcParams["savefig.format"])
    fmt = fmt.lower()
    options = dict(savefig_options)
    if fmt in {"pdf", "svg"}:
        metadata = dict(options.pop("metadata", None) or {})
        metadata.setdefault("CreationDate" if fmt == "pdf" else "Date", None)
        options["metadata"] = metadata
    _match_new_ticks_to_existing(figure)
    with mpl.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
        figure.savefig(target, format=format, **options)


def fix_pdf_dates(path: str | Path) -> None:
    """Replace the creation and modification dates of a PDF by 1 January 2000.

    Browsers stamp the time of printing into the PDF information dictionary.
    The replacement has the same number of bytes, so the cross-reference
    table stays valid and nothing else in the file changes.
    """

    pdf = Path(path)
    content = pdf.read_bytes()
    fixed = _PDF_DATE_PATTERN.sub(
        lambda match: match.group(0)[: -len(match.group(2))] + _FIXED_PDF_DATE,
        content,
    )
    if fixed != content:
        pdf.write_bytes(fixed)


def save_figure(
    figure: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("svg", "pdf", "png", "tiff"),
    raster_dpi: int = 600,
    transparent: bool = False,
    preserve_canvas: bool = True,
) -> tuple[Path, ...]:
    """Export editable vector files and requested high-resolution rasters.

    ``output_stem`` must not carry an extension.  SVG/PDF preserve editable
    text and vector artists; PNG/TIFF use ``raster_dpi``.  Full publication
    figures preserve the declared canvas by default.  Set ``preserve_canvas``
    to ``False`` only for a component whose contract explicitly calls for a
    tightly cropped bounding box.
    """

    stem = Path(output_stem)
    if stem.suffix:
        raise ValueError("output_stem must not include a file extension")
    raster_dpi = positive_integer(raster_dpi, name="raster_dpi")
    transparent = strict_boolean(transparent, name="transparent")
    preserve_canvas = strict_boolean(preserve_canvas, name="preserve_canvas")
    normalized = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    allowed = {"svg", "pdf", "png", "tif", "tiff"}
    invalid = sorted(set(normalized).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported figure formats: {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("formats must not contain duplicates")
    stem.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for fmt in normalized:
        path = stem.with_suffix(f".{fmt}")
        kwargs: dict[str, object] = {"transparent": transparent}
        if not preserve_canvas:
            kwargs["bbox_inches"] = "tight"
        if fmt in {"png", "tif", "tiff"}:
            kwargs["dpi"] = raster_dpi
        if fmt in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        if fmt in {"svg", "pdf"}:
            with vector_export_style():
                savefig_reproducibly(figure, path, format=fmt, **kwargs)
        else:
            savefig_reproducibly(figure, path, format=fmt, **kwargs)
        if fmt == "svg":
            svg = path.read_text(encoding="utf-8")
            sanitized = _MATPLOTLIB_SVG_DOCTYPE.sub("", svg)
            if sanitized != svg:
                path.write_text(sanitized, encoding="utf-8")
        outputs.append(path)
    return tuple(outputs)


def declare_svg_canvas_size(
    svg_path: str | Path,
    *,
    width_mm: float,
    height_mm: float,
    view_box: Sequence[float],
) -> None:
    """Set an SVG's physical size while preserving its drawing coordinates."""

    path = Path(svg_path)
    if path.suffix.casefold() != ".svg":
        raise ValueError("svg_path must end in .svg")
    if width_mm <= 0.0 or height_mm <= 0.0:
        raise ValueError("SVG dimensions must be positive")
    expected = tuple(float(value) for value in view_box)
    if len(expected) != 4:
        raise ValueError("view_box must contain x, y, width, and height")
    svg = path.read_text(encoding="utf-8")
    root_match = re.search(r"<svg\b[^>]*>", svg, flags=re.IGNORECASE)
    if root_match is None:
        raise ValueError(f"No SVG root element found in {path}")
    root_tag = root_match.group(0)
    view_match = re.search(r'\bviewBox="([^"]+)"', root_tag)
    if view_match is None:
        raise ValueError(f"No viewBox found in {path}")
    observed = tuple(float(value) for value in view_match.group(1).split())
    if len(observed) != 4 or any(
        abs(left - right) > 1e-6 for left, right in zip(observed, expected)
    ):
        raise ValueError(
            f"Unexpected SVG viewBox in {path}: {observed}; expected {expected}"
        )

    def replace_attribute(tag: str, name: str, value: str) -> str:
        """Replace one required root attribute without reserializing the SVG."""

        pattern = rf'\b{name}="[^"]*"'
        if re.search(pattern, tag) is None:
            raise ValueError(f"SVG root is missing {name}")
        return re.sub(pattern, f'{name}="{value}"', tag, count=1)

    root_tag = replace_attribute(root_tag, "width", f"{width_mm:.3f}mm")
    root_tag = replace_attribute(root_tag, "height", f"{height_mm:.3f}mm")
    root_tag = replace_attribute(
        root_tag,
        "viewBox",
        f"0 0 {expected[2]:.6f} {expected[3]:.6f}",
    )
    rewritten = svg[: root_match.start()] + root_tag + svg[root_match.end() :]
    path.write_text(rewritten, encoding="utf-8", newline="\n")


BROWSER_COMMANDS: tuple[str, ...] = (
    "chrome",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "msedge",
    "microsoft-edge",
    "microsoft-edge-stable",
)
"""Browser commands searched on ``PATH``, in this order."""


def browser_install_locations(
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return the standard browser install locations searched after ``PATH``.

    Windows: Google Chrome under Program Files, Program Files (x86) and the
    user's local application data, then Microsoft Edge under the same roots.
    macOS: Google Chrome, Chromium and Microsoft Edge in ``/Applications`` and
    ``~/Applications``.  Linux browsers are found through ``PATH`` only.
    """

    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if platform.startswith("win"):
        roots = [
            environ.get("PROGRAMFILES") or r"C:\Program Files",
            environ.get("PROGRAMFILES(X86)") or r"C:\Program Files (x86)",
        ]
        local = environ.get("LOCALAPPDATA")
        if local:
            roots.append(local)
        chrome = Path("Google", "Chrome", "Application", "chrome.exe")
        edge = Path("Microsoft", "Edge", "Application", "msedge.exe")
        return tuple(Path(root) / program for program in (chrome, edge) for root in roots)
    if platform == "darwin":
        bundles = (
            "Google Chrome.app/Contents/MacOS/Google Chrome",
            "Chromium.app/Contents/MacOS/Chromium",
            "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        )
        roots = [Path("/Applications")]
        home = environ.get("HOME")
        if home:
            roots.append(Path(home) / "Applications")
        return tuple(root / bundle for bundle in bundles for root in roots)
    return ()


def chromium_export_browser(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path | None:
    """Return the browser for the Figure 2 and Figure 3 page exports, or None.

    ``MAZE_LIFELONG_MEMORY_BROWSER`` overrides the search: a path to the
    executable or a command on ``PATH`` selects that browser, and ``none``
    selects no browser, which skips the browser exports.  Without the
    variable the first of :data:`BROWSER_COMMANDS` on ``PATH`` is used, then
    the first existing entry of :func:`browser_install_locations`.
    """

    environ = os.environ if environ is None else environ
    override = environ.get(BROWSER_VARIABLE, "").strip()
    if override:
        if override.casefold() == "none":
            return None
        path = Path(override).expanduser()
        if path.is_file():
            return path
        found = which(override)
        return Path(found) if found else None
    candidates = [Path(found) for name in BROWSER_COMMANDS if (found := which(name))]
    candidates.extend(browser_install_locations(platform, environ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def browser_unavailable_reason(environ: Mapping[str, str] | None = None) -> str:
    """Explain why :func:`chromium_export_browser` found no browser."""

    environ = os.environ if environ is None else environ
    override = environ.get(BROWSER_VARIABLE, "").strip()
    if override.casefold() == "none":
        return f"{BROWSER_VARIABLE} is set to 'none'"
    if override:
        return (
            f"{BROWSER_VARIABLE} is set to '{override}', which is neither an "
            "executable file nor a command on PATH"
        )
    return (
        "no Google Chrome, Chromium or Microsoft Edge was found on PATH or in "
        "the standard install locations"
    )


def report_skipped_browser_export(
    page: str,
    *,
    skipped: Sequence[Path],
    written: Sequence[Path],
) -> None:
    """Print why a browser page export was skipped and what was still written.

    Files from an earlier export in ``skipped`` are removed, because they no
    longer match the page written now.
    """

    removed = [path for path in skipped if path.is_file()]
    for path in removed:
        path.unlink()
    message = (
        f"Skipped the browser export of {page}: {browser_unavailable_reason()}. "
        f"Not written: {', '.join(path.name for path in skipped)}. "
        f"Written: {', '.join(path.name for path in written)}, which holds the "
        "complete page and opens in any web browser; every panel file and value "
        "table of this notebook is written as usual. To export the page, install "
        f"Google Chrome, Chromium or Microsoft Edge, or set {BROWSER_VARIABLE} to "
        "the browser executable, and rerun this cell."
    )
    if removed:
        message += (
            f" Removed the earlier {', '.join(path.name for path in removed)}, "
            "which no longer matched the new page."
        )
    print(message, flush=True)


def wait_for_browser_file(path: Path, *, timeout_seconds: float = 15.0) -> None:
    """Wait until Chromium's detached writer finishes one output file."""

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
    raise RuntimeError(f"Browser export did not finish: {path}")


def export_svg_with_chromium(
    svg_path: str | Path,
    png_path: str | Path,
    pdf_path: str | Path,
    *,
    width_mm: float,
    height_mm: float,
    raster_dpi: int,
    tiff_path: str | Path | None = None,
    expected_pdf_size_points: Sequence[float] | None = None,
    expected_raster_size: Sequence[int] | None = None,
    page_name: str | None = None,
) -> Path | None:
    """Print an SVG with Chromium, then rasterize that exact PDF page.

    Returns the browser used.  Without a browser (see
    :func:`chromium_export_browser`) the PDF, PNG and TIFF are skipped with a
    printed message naming ``page_name`` and None is returned; the SVG is
    left as it is.  The PDF's printing date is fixed, so repeated exports
    are byte-identical.
    """

    source = Path(svg_path)
    png = Path(png_path)
    pdf = Path(pdf_path)
    tiff = None if tiff_path is None else Path(tiff_path)
    raster_dpi = positive_integer(raster_dpi, name="raster_dpi")
    if not source.is_file() or source.suffix.casefold() != ".svg":
        raise FileNotFoundError(source)
    browser = chromium_export_browser()
    if browser is None:
        report_skipped_browser_export(
            page_name or source.stem,
            skipped=(pdf, png, *(() if tiff is None else (tiff,))),
            written=(source,),
        )
        return None
    for path in (png, pdf, *(tuple() if tiff is None else (tiff,))):
        path.parent.mkdir(parents=True, exist_ok=True)

    svg_text = source.read_text(encoding="utf-8")
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: {width_mm:.6f}mm {height_mm:.6f}mm; margin: 0; }}
html, body {{ width: {width_mm:.6f}mm; height: {height_mm:.6f}mm; margin: 0; padding: 0; background: white; overflow: hidden; }}
svg {{ display: block; width: {width_mm:.6f}mm !important; height: {height_mm:.6f}mm !important; }}
* {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
</style></head><body>{svg_text}</body></html>
"""
    with tempfile.TemporaryDirectory(prefix=".maze_svg_export_", dir=png.parent) as temporary:
        temporary_path = Path(temporary)
        html_path = temporary_path / "figure.html"
        normalized_png = temporary_path / "figure.png"
        temporary_pdf = temporary_path / "figure.pdf"
        temporary_tiff = temporary_path / "figure.tiff"
        profile_path = temporary_path / "browser_profile"
        html_path.write_text(html_text, encoding="utf-8", newline="\n")
        common = (
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--user-data-dir={profile_path}",
        )
        subprocess.run(
            (
                *common,
                "--no-pdf-header-footer",
                f"--print-to-pdf={temporary_pdf}",
                html_path.as_uri(),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        wait_for_browser_file(temporary_pdf)

        with pymupdf.open(temporary_pdf) as document:
            if document.page_count != 1:
                raise RuntimeError(
                    f"Expected a one page browser PDF, found {document.page_count}"
                )
            page = document[0]
            observed_page_size = (float(page.rect.width), float(page.rect.height))
            if expected_pdf_size_points is not None:
                expected_page_size = tuple(
                    float(value) for value in expected_pdf_size_points
                )
                if len(expected_page_size) != 2 or any(
                    abs(observed - expected) > 1e-6
                    for observed, expected in zip(
                        observed_page_size, expected_page_size, strict=True
                    )
                ):
                    raise RuntimeError(
                        "Browser PDF page size differs from the accepted geometry: "
                        f"{observed_page_size}; expected {expected_page_size}"
                    )
            calculated_raster_size = tuple(
                round(points / 72.0 * raster_dpi) for points in observed_page_size
            )
            if expected_raster_size is not None:
                declared_raster_size = tuple(int(value) for value in expected_raster_size)
                if (
                    len(declared_raster_size) != 2
                    or calculated_raster_size != declared_raster_size
                ):
                    raise RuntimeError(
                        "Accepted raster size is inconsistent with the browser PDF: "
                        f"{calculated_raster_size}; expected {declared_raster_size}"
                    )
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(raster_dpi / 72.0, raster_dpi / 72.0),
                alpha=False,
            )
        raster_size = (pixmap.width, pixmap.height)
        if raster_size != calculated_raster_size:
            raise RuntimeError(
                "PDF raster geometry mismatch: "
                f"{raster_size}; expected {calculated_raster_size}"
            )
        image = Image.frombytes("RGB", raster_size, pixmap.samples)
        image.save(normalized_png, dpi=(raster_dpi, raster_dpi), optimize=True)
        if tiff is not None:
            image.save(
                temporary_tiff,
                compression="tiff_lzw",
                dpi=(raster_dpi, raster_dpi),
            )

        fix_pdf_dates(temporary_pdf)
        os.replace(temporary_pdf, pdf)
        os.replace(normalized_png, png)
        if tiff is not None:
            os.replace(temporary_tiff, tiff)
    return browser


def close_figure(figure: Figure) -> None:
    """Close an exported figure to release memory during batch reproduction."""

    plt.close(figure)


__all__ = [
    "ARIAL_METRIC_SUBSTITUTES",
    "BROWSER_COMMANDS",
    "BROWSER_VARIABLE",
    "HIDE_FONTS_VARIABLE",
    "LAST_RESORT_FONT_FAMILY",
    "PUBLICATION_FONT_FAMILY",
    "PUBLICATION_RC",
    "SVG_HASH_SALT",
    "VERSION_RECORD_FONT_FAMILY",
    "browser_install_locations",
    "browser_unavailable_reason",
    "choose_figure_font_family",
    "chromium_export_browser",
    "close_figure",
    "declare_svg_canvas_size",
    "despine",
    "export_svg_with_chromium",
    "figure_font_family",
    "figure_font_notice",
    "fix_pdf_dates",
    "font_is_installed",
    "publication_style",
    "report_skipped_browser_export",
    "resolved_publication_font",
    "save_figure",
    "savefig_reproducibly",
    "vector_export_style",
    "version_record_style",
    "wait_for_browser_file",
    "with_figure_font",
]
