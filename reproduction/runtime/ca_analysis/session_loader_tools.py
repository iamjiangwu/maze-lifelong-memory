from __future__ import annotations

# Auto-migrated local copy from legacy source.
# This module is self-contained within the repository.

# session_loaders.py â€” single & cross-session pickle loaders
#
# A small, battle-tested loader suite for neuroscience pipelines.
# It supports:
#   1) Singleâ€‘session files like:  "wtn_{name}_spatial.pkl"
#   2) Crossâ€‘session files like:   "tuning_{src}_to_{dst}.pkl"
#
# Highlights
# - Searches across multiple root directories (first match wins, warns on dups)
# - Exact filename matching (no accidental *partial* matches)
# - Optional threaded I/O (workers>0) for speedup
# - Friendly summaries and object inspection helpers
# - Robust name templating: str / (src,dst) tuple / {src:..., dst:...}
# - Slug parsing: "A_to_B" â†’ (A, B)
#
# Usage examples are at the bottom of this file.


from dataclasses import dataclass
import pickle
import warnings
import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Types
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Name = str
Pair = Tuple[str, str]
NameLike = Union[Name, Pair, Mapping[str, str]]

ResultDict = Dict[str, Dict[str, Any]]  # { group: { key: object } }

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Core I/O utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _default_pickle_loader(p: Path) -> Any:
    with p.open("rb") as f:
        return pickle.load(f)


def _try_pandas_loader() -> Optional[Callable[[Path], Any]]:
    """Prefer pandas.read_pickle if available (preserves pd metadata)."""
    try:
        import pandas as pd  # type: ignore
        return lambda p: pd.read_pickle(p)
    except Exception:
        return None


def _format_pattern(pattern_template: str, name_obj: NameLike) -> str:
    """Allow flexible templating:
    - str   â†’ pattern_template.format(name=...)
    - tuple â†’ pattern_template.format(*name_obj)
    - dict  â†’ pattern_template.format(**name_obj)
    """
    if isinstance(name_obj, Mapping):
        return pattern_template.format(**name_obj)
    elif isinstance(name_obj, tuple):
        return pattern_template.format(*name_obj)
    else:
        return pattern_template.format(name=name_obj)


def _find_one_path(
    name_obj: NameLike,
    roots: Iterable[Path],
    *,
    pattern_template: str,
) -> Optional[Path]:
    """Search for an *exact* filename match derived from the template.

    Strategy per root:
      1) Direct join (fast path)
      2) rglob the exact filename (not a pattern) if not found directly
    """
    exact_rel = _format_pattern(pattern_template, name_obj)
    found_paths: List[Path] = []

    for root in roots:
        root = Path(root)
        # 1) direct
        direct = root / exact_rel
        if direct.exists():
            found_paths.append(direct)
            continue

        # 2) exact-name rglob (avoid partial matches)
        target_name = Path(exact_rel).name
        matches = [p for p in root.rglob(target_name) if p.name == target_name]
        found_paths.extend(matches)

    if not found_paths:
        return None

    # Deduplicate identical files reachable through multiple paths (e.g., symlinks)
    unique: List[Path] = []
    seen: set[str] = set()
    for p in found_paths:
        key = p.resolve().as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if len(unique) > 1:
        warnings.warn(
            "Multiple matches for '{}':\n  {}\nUsing the first one. Consider narrowing your roots.".format(
                exact_rel, "\n  ".join(str(p) for p in unique)
            )
        )
    return unique[0]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Crossâ€‘session helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_slug_re = re.compile(r"^(?P<src>.+?)_to_(?P<dst>.+)$")


def parse_pair_slug(slug: str) -> Pair:
    """Parse "foo_to_bar" â†’ ("foo", "bar").
    The separator is the literal substring "_to_".
    """
    m = _slug_re.match(slug)
    if not m:
        raise ValueError(f"Cannot parse pair slug: {slug!r} (expected 'SRC_to_DST')")
    return m.group("src"), m.group("dst")


def canonical_pair_key(pair: NameLike) -> str:
    """Produce a pretty, stable dictionary key for a crossâ€‘session pair.
    Accepts tuple, mapping with src/dst, or a slug string.
    """
    if isinstance(pair, Mapping):
        src, dst = pair.get("src"), pair.get("dst")
        if not (isinstance(src, str) and isinstance(dst, str)):
            raise ValueError("Mapping must contain 'src' and 'dst' strings")
    elif isinstance(pair, tuple):
        if len(pair) != 2:
            raise ValueError("Pair tuple must be length 2: (src, dst)")
        src, dst = pair  # type: ignore[misc]
    else:  # slug
        src, dst = parse_pair_slug(pair)
    return f"{src}â†’{dst}"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Public API â€” Singleâ€‘session loader
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_single_session_pickles(
    groups: Dict[str, List[Name]],
    data_roots: Iterable[Union[str, Path]],
    *,
    pattern_template: str = "wtn_{name}_spatial.pkl",
    loader: Optional[Callable[[Path], Any]] = None,
    strict: bool = False,
    workers: int = 0,
) -> ResultDict:
    """Load singleâ€‘session files named like 'wtn_{name}_spatial.pkl'.

    Parameters
    ----------
    groups : dict
        { group_label: [name1, name2, ...] }
    data_roots : iterable of paths
        One or more directories to search (first match wins).
    pattern_template : str
        Filename template; must include '{name}'.
    loader : callable
        Function to deserialize a file. Default: pandas.read_pickle if available,
        else pickle.load.
    strict : bool
        If True, missing or failed loads raise. If False, they warn and skip.
    workers : int
        If > 0, use a ThreadPoolExecutor with this many worker threads.

    Returns
    -------
    { group: { name: object } }
    """
    roots = [Path(r) for r in data_roots]
    loader = loader or _try_pandas_loader() or _default_pickle_loader

    # Plan
    jobs: List[Tuple[str, Name, Optional[Path]]] = []
    for g, name_list in groups.items():
        for nm in name_list:
            p = _find_one_path(nm, roots, pattern_template=pattern_template)
            jobs.append((g, nm, p))

    missing = [(g, nm) for (g, nm, p) in jobs if p is None]
    if missing:
        msg = "Missing files: " + ", ".join([f"{g}:{nm}" for g, nm in missing])
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)

    # Load
    result: ResultDict = {g: {} for g in groups}
    to_load = [(g, nm, p) for (g, nm, p) in jobs if p is not None]

    if workers and workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2key = {ex.submit(loader, p): (g, nm, p) for (g, nm, p) in to_load}
            for fut in as_completed(fut2key):
                g, nm, p = fut2key[fut]
                try:
                    obj = fut.result()
                except Exception as e:  # noqa: BLE001
                    if strict:
                        raise
                    warnings.warn(f"Failed to load {p} ({g}:{nm}): {e}")
                    continue
                result[g][nm] = obj
    else:
        for g, nm, p in to_load:
            try:
                result[g][nm] = loader(p)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                warnings.warn(f"Failed to load {p} ({g}:{nm}): {e}")

    return result

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Public API â€” Crossâ€‘session loader
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_cross_session_pickles(
    groups: Dict[str, List[NameLike]],
    data_roots: Iterable[Union[str, Path]],
    *,
    pattern_template: str = "tuning_{src}_to_{dst}.pkl",
    loader: Optional[Callable[[Path], Any]] = None,
    strict: bool = False,
    workers: int = 0,
    key_fn: Optional[Callable[[NameLike], str]] = canonical_pair_key,
) -> ResultDict:
    """Load crossâ€‘session files like 'tuning_{src}_to_{dst}.pkl'.

    `groups` may contain items as:
      - (src, dst) tuples
      - {"src": ..., "dst": ...} mappings
      - slug strings "SRC_to_DST"

    The dictionary keys for each group's results default to a pretty
    canonical form 'SRCâ†’DST'. You can override with `key_fn`.
    """
    roots = [Path(r) for r in data_roots]
    loader = loader or _try_pandas_loader() or _default_pickle_loader

    # Normalize names for path formatting (always provide src/dst for the template)
    def as_mapping(nm: NameLike) -> Mapping[str, str]:
        if isinstance(nm, Mapping):
            if "src" in nm and "dst" in nm:
                return nm  # type: ignore[return-value]
            raise ValueError("Mapping names must include 'src' and 'dst'.")
        if isinstance(nm, tuple):
            if len(nm) != 2:
                raise ValueError("Pair tuple must be length 2: (src, dst)")
            return {"src": nm[0], "dst": nm[1]}
        # str slug â†’ parse
        src, dst = parse_pair_slug(nm)
        return {"src": src, "dst": dst}

    # Plan jobs
    planned: List[Tuple[str, NameLike, Optional[Path]]] = []
    for g, name_list in groups.items():
        for nm in name_list:
            nm_map = as_mapping(nm)
            p = _find_one_path(nm_map, roots, pattern_template=pattern_template)
            planned.append((g, nm, p))

    # Missing
    missing = [(g, nm) for (g, nm, p) in planned if p is None]
    if missing:
        def _to_str(x: NameLike) -> str:
            try:
                return canonical_pair_key(x)
            except Exception:
                return str(x)
        msg = "Missing crossâ€‘session files: " + \
              ", ".join([f"{g}:{_to_str(nm)}" for g, nm in missing])
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)

    # Load
    result: ResultDict = {g: {} for g in groups}
    to_load = [(g, nm, p) for (g, nm, p) in planned if p is not None]

    def key_for(nm: NameLike) -> str:
        return key_fn(nm) if key_fn else canonical_pair_key(nm)

    if workers and workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2key = {ex.submit(loader, p): (g, nm, p) for (g, nm, p) in to_load}
            for fut in as_completed(fut2key):
                g, nm, p = fut2key[fut]
                try:
                    obj = fut.result()
                except Exception as e:  # noqa: BLE001
                    if strict:
                        raise
                    warnings.warn(f"Failed to load {p} ({g}:{canonical_pair_key(nm)}): {e}")
                    continue
                result[g][key_for(nm)] = obj
    else:
        for g, nm, p in to_load:
            try:
                result[g][key_for(nm)] = loader(p)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                warnings.warn(f"Failed to load {p} ({g}:{canonical_pair_key(nm)}): {e}")

    return result

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convenience: flatten, summarize, and inspect
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def flatten_loaded(groups_loaded: ResultDict) -> List[Tuple[str, str, Any]]:
    """Convert {group:{key:obj}} â†’ [(group, key, obj), ...] for easy iteration."""
    out: List[Tuple[str, str, Any]] = []
    for g, d in groups_loaded.items():
        for key, obj in d.items():
            out.append((g, key, obj))
    return out


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Enhanced summary: unified for single and cross-session files
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_load_summary(
    groups_loaded: dict,
    requested: dict,
    *,
    show_keys: bool = True,
    max_keys: int = 40,
    key_freq: float = 0.0,
) -> None:
    """Print load status, missing entries, and variable availability.

    Works for both single-session and cross-session files.

    - Detects cross-session pairs (tuple, dict with src/dst, or string with '_to_' or 'â†’').
    - Prints missing files and the available variable names (dict keys, DataFrame columns, etc.).
    - Aggregates variable counts across all successfully loaded files in each group.
    """
    import warnings

    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    def _is_pair_like(x):
        return (
            isinstance(x, tuple) and len(x) == 2
        ) or (
            isinstance(x, dict) and {'src', 'dst'} <= set(x)
        ) or (
            isinstance(x, str) and ('_to_' in x or 'â†’' in x)
        )

    def _canonical_pair_key(x):
        if isinstance(x, str):
            if 'â†’' in x:
                return x
            elif '_to_' in x:
                return x.replace('_to_', 'â†’')
            else:
                return x
        if isinstance(x, tuple) and len(x) == 2:
            return f"{x[0]}â†’{x[1]}"
        if isinstance(x, dict) and 'src' in x and 'dst' in x:
            return f"{x['src']}â†’{x['dst']}"
        return str(x)

    def _vars_from_obj(obj):
        if isinstance(obj, dict):
            return list(obj.keys())
        if pd is not None and isinstance(obj, pd.DataFrame):
            return list(obj.columns)
        if np is not None and isinstance(obj, np.ndarray) and getattr(obj.dtype, 'names', None):
            return list(obj.dtype.names or [])
        return []

    for g, req in requested.items():
        got_dict = groups_loaded.get(g, {})
        got_keys = list(got_dict.keys())

        # normalize requested names (handle cross-session)
        norm_req = []
        for x in req:
            if _is_pair_like(x):
                norm_req.append(_canonical_pair_key(x))
            else:
                norm_req.append(str(x))

        missing = [x for x in norm_req if x not in got_keys]
        print(f"{g}: loaded {len(got_keys)}/{len(norm_req)}")

        if missing:
            print(f"  Missing ({len(missing)}): {', '.join(missing)}")
        else:
            print("  Missing: none")

        if show_keys and got_keys:
            counts = {}
            for nm, obj in got_dict.items():
                vars_ = _vars_from_obj(obj)
                for v in vars_:
                    counts[v] = counts.get(v, 0) + 1

            if counts:
                n_items = len(got_keys)
                sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                if key_freq > 0:
                    sorted_items = [kv for kv in sorted_items if kv[1] / n_items >= key_freq]
                sorted_items = sorted_items[:max_keys]
                print("  Variables (present/total):")
                for name, c in sorted_items:
                    print(f"    - {name}: {c}/{n_items}")
            else:
                print("  Variables: (no keys, columns, or fields detected)")
        print("")


@dataclass
class ObjectPreview:
    typ: str
    shape: Optional[Tuple[int, ...]]
    summary: str


def inspect_loaded_object(obj: Any, *, head: int = 5) -> ObjectPreview:
    """Return a compact preview describing a loaded object.

    - pandas.DataFrame: shape, head() text
    - pandas.Series: length, head() text
    - numpy.ndarray: dtype + shape
    - dict: topâ€‘level keys
    - list/tuple: length + first few types
    - other: type name + truncated repr
    """
    # Lazy imports
    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - optional
        pd = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional
        np = None  # type: ignore

    tname = type(obj).__name__

    if pd is not None and isinstance(obj, pd.DataFrame):
        with pd.option_context("display.max_rows", head, "display.max_columns", 20):
            return ObjectPreview(
                typ="pandas.DataFrame",
                shape=tuple(obj.shape),
                summary=str(obj.head(head)),
            )

    if pd is not None and isinstance(obj, pd.Series):
        with pd.option_context("display.max_rows", head):
            return ObjectPreview(
                typ="pandas.Series",
                shape=(obj.shape[0],),
                summary=str(obj.head(head)),
            )

    if np is not None and isinstance(obj, np.ndarray):
        return ObjectPreview(
            typ=f"numpy.ndarray[{obj.dtype}]",
            shape=tuple(obj.shape),
            summary=f"min={obj.min():.4g}, max={obj.max():.4g}" if obj.size else "empty",
        )

    if isinstance(obj, dict):
        keys = list(obj.keys())
        show = keys[:10]
        extra = "" if len(keys) <= 10 else f" (+{len(keys)-10} more)"
        return ObjectPreview(typ="dict", shape=None, summary="keys: " + ", ".join(map(str, show)) + extra)

    if isinstance(obj, (list, tuple)):
        inner = ", ".join(type(x).__name__ for x in obj[:10])
        extra = "" if len(obj) <= 10 else f" (+{len(obj)-10} more)"
        return ObjectPreview(typ=f"{tname}", shape=(len(obj),), summary=f"first types: [{inner}]{extra}")

    # Fallback
    s = repr(obj)
    if len(s) > 500:
        s = s[:500] + " â€¦"
    return ObjectPreview(typ=tname, shape=None, summary=s)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cross-session variant orchestration (day1â†’day1, day1â†’day2, day2â†’day1, day2â†’day2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from typing import Dict, List, Tuple, Callable, Any, Optional
import re as _re

Pair      = Tuple[str, str]
Groups    = Dict[str, List[Pair]]                 # e.g., {"once":[(src,dst),...], "twice":[...]}
Dataset   = Dict[str, Dict[str, Dict[str, Any]]]  # variant -> group -> "SRCâ†’DST" -> obj or {"data","shuffle"}

def default_pv_variant_base_groups(*, include_new_animals: bool = False) -> Groups:
    """Return the maintained day1-to-day1 base cohort for PV/place variants."""
    groups: Groups = {
        "once": [
            ("1616day1", "1616bday1"),
            ("418RRday1", "418-3day1"),
            ("434RRday1", "434-3day1"),
            ("451Rday1", "451-3day1"),
            ("488Rday1", "488-3day1"),
            ("489Lday1", "489-3day1"),
            ("579RLday1", "579-3day1"),
            ("468Lday1", "468-3day1"),
            ("422RRday1", "422-3day1"),
        ],
        "twice": [
            ("1616bday1", "1616b-3day1"),
            ("418-3day1", "418-6day1"),
            ("451-3day1", "451-6day1"),
            ("488-3day1", "488-6day1"),
            ("489-3day1", "489-6day1"),
            ("468-3day1", "468-6day1"),
            ("422-3day1", "422-6day1"),
        ],
        "three times": [
            ("418-6day1", "418-9day1"),
            ("451-6day1", "451-9day1"),
            ("468-6day1", "468-9day1"),
        ],
        "four times": [
            ("468-9day1", "468-12day1"),
        ],
    }
    if include_new_animals:
        new_animals_with_9mo = ("837", "838", "839", "842")
        new_animals_with_6mo = (*new_animals_with_9mo, "862")
        groups["once"].extend((f"{animal}day1", f"{animal}-3day1") for animal in new_animals_with_6mo)
        groups["twice"].extend((f"{animal}-3day1", f"{animal}-6day1") for animal in new_animals_with_6mo)
        groups["three times"].extend((f"{animal}-6day1", f"{animal}-9day1") for animal in new_animals_with_9mo)
    return {name: list(pairs) for name, pairs in groups.items()}

_DAY_RE = _re.compile(r"day(\d+)$")

def replace_day(name: str, to_day: int) -> str:
    """Replace trailing 'dayN' with 'day{to_day}' if present; else append."""
    if _DAY_RE.search(name):
        return _DAY_RE.sub(f"day{to_day}", name)
    return f"{name}day{to_day}"

def make_variant_transform(src_day: int, dst_day: int) -> Callable[[Pair], Pair]:
    """Return a function that rewrites (src,dst) to desired days."""
    def _xf(pair: Pair) -> Pair:
        s, d = pair
        return replace_day(s, src_day), replace_day(d, dst_day)
    return _xf

# Default 4 variants
DEFAULT_VARIANTS: Dict[str, Callable[[Pair], Pair]] = {
    "d1v1": make_variant_transform(1, 1),  # day1 â†’ day1
    "d1v2": make_variant_transform(1, 2),  # day1 â†’ day2
    "d2v1": make_variant_transform(2, 1),  # day2 â†’ day1
    "d2v2": make_variant_transform(2, 2),  # day2 â†’ day2
}

def build_variant_groups(base: Groups,
                         variants: Optional[Dict[str, Callable[[Pair], Pair]]] = None
                         ) -> Dict[str, Groups]:
    """Derive {variant: {group: [(src,dst), ...]}} from a base group dict."""
    variants = variants or DEFAULT_VARIANTS
    out: Dict[str, Groups] = {}
    for vname, xf in variants.items():
        out[vname] = {grp: [xf(p) for p in pairs] for grp, pairs in base.items()}
    return out

def _merge_main_and_shuffle(main: Dict[str, Dict[str, Any]],
                            shuf: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return {group: {pair_key: {'data': obj|None, 'shuffle': obj|None}}}."""
    merged: Dict[str, Dict[str, Any]] = {g: {} for g in set(main) | set(shuf)}
    for g in merged:
        all_keys = set(main.get(g, {})) | set(shuf.get(g, {}))
        for k in all_keys:
            merged[g][k] = {"data": main.get(g, {}).get(k), "shuffle": shuf.get(g, {}).get(k)}
    return merged

def load_cross_session_variants(
    base_groups: Groups,
    data_roots,
    *,
    pattern_template: str = "tuning_{src}_to_{dst}.pkl",
    shuffle_pattern_template: Optional[str] = None,   # e.g., "tuning_{src}_to_{dst}_shuffle.pkl"
    strict: bool = False,
    workers: int = 0,
    variants: Optional[Dict[str, Callable[[Pair], Pair]]] = None,
    key_fn: Optional[Callable[[Any], str]] = canonical_pair_key,  # keep 'SRCâ†’DST'
) -> Dataset:
    """
    Build day-variants from base pairs and load each variant into a registry.

      Returns:
        registry[variant][group]['SRCâ†’DST'] = obj
        or, if shuffle_pattern_template is provided:
        registry[variant][group]['SRCâ†’DST'] = {'data': obj, 'shuffle': shuf_obj}
    """
    variant_groups = build_variant_groups(base_groups, variants=variants)
    registry: Dataset = {}
    for vname, groups in variant_groups.items():
        loaded = load_cross_session_pickles(
            groups, data_roots,
            pattern_template=pattern_template,
            strict=strict, workers=workers,
            key_fn=key_fn or canonical_pair_key
        )
        if shuffle_pattern_template:
            shuf = load_cross_session_pickles(
                groups, data_roots,
                pattern_template=shuffle_pattern_template,
                strict=False, workers=workers,
                key_fn=key_fn or canonical_pair_key
            )
            registry[vname] = _merge_main_and_shuffle(loaded, shuf)
        else:
            registry[vname] = loaded
    return registry

def get_entry(registry: Dataset, variant: str, group: str, src: str, dst: str):
    """Convenience accessor by raw (src,dst) to canonical key 'SRCâ†’DST'."""
    key = canonical_pair_key((src, dst))
    return registry[variant][group].get(key)

# Animal indexing helpers (override animal_id_fn as needed)
import re as _aid_re

# Digits-only animal id (1616 and 1616b â†’ 1616; 418RRday1 â†’ 418; 468-12day1 â†’ 468)
_ANIMAL_ID_RE = _aid_re.compile(r"^(\d+)")

def animal_id_default(session_name: str) -> str:
    """Extract animal id as leading digits."""
    if not isinstance(session_name, str):
        return str(session_name)
    m = _ANIMAL_ID_RE.match(session_name)
    return m.group(1) if m else session_name


def index_by_animal(registry: Dataset, *, variant: str, group: str,
                    animal_id_fn: Callable[[str], str] = animal_id_default) -> Dict[str, List[str]]:
    """
    Return {animal_id: [pair_keys...]} for a given (variant, group).
    Uses the *src* name for ID by default.
    """
    out: Dict[str, List[str]] = {}
    for pair_key in registry[variant][group]:
        src, _dst = pair_key.split("â†’", 1)
        aid = animal_id_fn(src)
        out.setdefault(aid, []).append(pair_key)
    return out

def gather_across_variants(registry: Dataset, *, group: str,
                           animal_id_fn: Callable[[str], str] = animal_id_default) -> Dict[str, Dict[str, List[str]]]:
    """
    Return {animal_id: {variant: [pair_keys...]}} for a given group across all variants.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for vname in registry:
        idx = index_by_animal(registry, variant=vname, group=group, animal_id_fn=animal_id_fn)
        for aid, keys in idx.items():
            out.setdefault(aid, {})[vname] = keys
    return out

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Introspection helpers: show per-variable dimensions for a loaded file
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from typing import Iterable

def _shape_str(x) -> str:
    """Human-readable size/shape for common containers."""
    try:
        import numpy as _np  # optional
    except Exception:  # pragma: no cover
        _np = None
    try:
        import pandas as _pd  # optional
    except Exception:  # pragma: no cover
        _pd = None

    # pandas
    if _pd is not None and isinstance(x, _pd.DataFrame):
        return f"DataFrame[{x.shape[0]}Ã—{x.shape[1]}]"
    if _pd is not None and isinstance(x, _pd.Series):
        return f"Series[len={x.shape[0]}]"

    # numpy
    if _np is not None and isinstance(x, _np.ndarray):
        return f"ndarray{tuple(x.shape)} dtype={x.dtype}"

    # scipy sparse (best-effort, don't require it)
    try:
        import scipy.sparse as _sp  # type: ignore
        if _sp.issparse(x):
            return f"{type(x).__name__}{tuple(x.shape)} nnz={x.nnz}"
    except Exception:
        pass

    # dict / list / tuple
    if isinstance(x, dict):
        return f"dict[{len(x)} keys]"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[len={len(x)}]"

    # generic objects that expose .shape
    shp = getattr(x, "shape", None)
    if shp is not None:
        try:
            t = tuple(shp)
            return f"{type(x).__name__}{t}"
        except Exception:
            pass

    # fallback
    return type(x).__name__


def describe_variables(obj: Any, *, max_keys: int = 200) -> list[dict]:
    """
    Return a list of {name, kind, detail} for top-level variables in `obj`.
    If `obj` is a dict-like bundle (typical), each entry describes one key.
    """
    rows: list[dict] = []

    # If it's a dict-like container, enumerate keys
    if isinstance(obj, dict):
        keys: Iterable[str] = list(obj.keys())
        for k in list(keys)[:max_keys]:
            v = obj[k]
            rows.append({
                "name": str(k),
                "kind": type(v).__name__,
                "detail": _shape_str(v),
            })
        if len(keys) > max_keys:
            rows.append({"name": "â€¦", "kind": "", "detail": f"(+{len(list(keys))-max_keys} more keys)"})
        return rows

    # Not a dict: just summarize the object itself
    rows.append({"name": "<object>", "kind": type(obj).__name__, "detail": _shape_str(obj)})
    return rows


def print_pair_variables(
    registry: dict,
    variant: str,
    group: str,
    src: str | None = None,
    dst: str | None = None,
    *,
    pair: Any | None = None,         # alternatively pass ("SRC","DST") or slug/dict
    which: str | None = None,        # None | "data" | "shuffle" when registry stores {'data','shuffle'}
    max_keys: int = 200,
) -> None:
    """
    Pretty-print variable dimensions for a specific entry (cross- or single-session).

    Examples
    --------
    print_pair_variables(reg, "d1v1", "once", "418RRday1", "418-3day1")
    print_pair_variables(reg, "d1v1", "once", pair=("418RRday1","418-3day1"))
    print_pair_variables(reg, "d1v1", "once", "418RRday1", "418-3day1", which="shuffle")
    """
    # Resolve canonical key
    if pair is None:
        if src is None or dst is None:
            raise ValueError("Provide (src, dst) or `pair=`.")
        key = canonical_pair_key((src, dst))
    else:
        key = canonical_pair_key(pair)

    if variant not in registry or group not in registry[variant]:
        raise KeyError(f"Unknown variant/group: {variant}/{group}")

    entry = registry[variant][group].get(key)
    if entry is None:
        print(f"[not found] {variant}/{group}/{key}")
        return

    # If we stored {'data','shuffle'}, pick which branch
    obj = entry
    if isinstance(entry, dict) and ("data" in entry or "shuffle" in entry):
        if which is None:
            # default to 'data' if present
            obj = entry.get("data", entry.get("shuffle"))
            branch = "data" if "data" in entry else "shuffle"
        else:
            if which not in ("data", "shuffle"):
                raise ValueError("`which` must be one of None|'data'|'shuffle'")
            obj = entry.get(which)
            branch = which
        print(f"{variant} / {group} / {key} [{branch}]")
    else:
        print(f"{variant} / {group} / {key}")

    if obj is None:
        print("  (entry present but object is None)")
        return

    rows = describe_variables(obj, max_keys=max_keys)
    for r in rows:
        print(f"  - {r['name']}: {r['detail']}")  # e.g., sim_mean: ndarray(600,) dtype=float64

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Similarity summaries: data mean and per-shuffle means
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as _np
from typing import Any, Dict

def _as_float_1d(x) -> _np.ndarray:
    a = _np.asarray(x, dtype=float)
    return a.ravel()

def _nanmean1d(x: _np.ndarray) -> float:
    return float(_np.nanmean(x)) if x.size else _np.nan

# ---------- Flexible getters for nested keys or callables ----------

from typing import Iterable, Sequence, Callable, Any, Optional
import numpy as _np

def _get_nested(obj: Any, key: str) -> Any:
    """
    Get obj[key] supporting dotted paths like 'metrics.similarity'.
    Works with dict keys first, then attribute access as fallback.
    """
    cur = obj
    for part in key.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur

def _resolve_key(obj: Any, spec: Any) -> Any:
    """
    spec can be:
      - callable(obj) -> value
      - string key (supports dotted 'a.b.c')
      - list/tuple of the above (first non-None wins)
    """
    if callable(spec):
        try:
            return spec(obj)
        except Exception:
            return None
    if isinstance(spec, (list, tuple)):
        for s in spec:
            val = _resolve_key(obj, s)
            if val is not None:
                return val
        return None
    if isinstance(spec, str):
        return _get_nested(obj, spec)
    return None

def _summarize_similarity_obj(
    obj: Any,
    *,
    data_key: Any = "similarity",                     # str | [str,...] | callable
    shuffle_repeat_key: Any = "sim_per_repeat",       # str | [str,...] | callable
    shuffle_matrix_key: Any = "sim_matrix",           # str | [str,...] | callable
) -> dict:
    """
    Pull similarity summaries from a single object (either data OR shuffle).

    - DATA: value from `data_key` should be a 1D vector of per-pair similarities.
    - SHUFFLE: prefer `shuffle_repeat_key` (1D, per-repeat). If absent, use
      `shuffle_matrix_key` (2D, repeats Ã— pairs) and average across pairs per repeat.
    """
    out: dict = {}

    # Gracefully handle non-dict objects that are directly arrays
    if not isinstance(obj, dict):
        vec = getattr(obj, "__array__", None)
        if callable(vec):
            vec = _as_float_1d(obj)
            out["data_mean"] = _nanmean1d(vec)
            out["n_pairs"] = int(vec.size)
        return out

    # DATA
    data_vec = _resolve_key(obj, data_key)
    if data_vec is not None:
        v = _as_float_1d(data_vec)
        out["data_mean"] = _nanmean1d(v)
        out["n_pairs"] = int(v.size)

    # SHUFFLE (prefer per-repeat vector)
    rep = _resolve_key(obj, shuffle_repeat_key)
    if rep is not None:
        srep = _as_float_1d(rep)
        out["shuffle_means"] = srep
        out["shuffle_mean"] = _nanmean1d(srep)
        out["shuffle_std"]  = float(_np.nanstd(srep, ddof=1)) if srep.size > 1 else 0.0
        out["n_repeats"]    = int(srep.size)
        return out

    # Otherwise, try a 2D matrix: (repeats Ã— pairs)
    M = _resolve_key(obj, shuffle_matrix_key)
    if M is not None:
        M = _np.asarray(M, dtype=float)
        if M.ndim == 2 and M.size:
            srep = _np.nanmean(M, axis=1)
            out["shuffle_means"] = srep
            out["shuffle_mean"] = _nanmean1d(srep)
            out["shuffle_std"]  = float(_np.nanstd(srep, ddof=1)) if srep.size > 1 else 0.0
            out["n_repeats"]    = int(srep.size)
        else:
            out["shuffle_means"] = _np.asarray([], dtype=float)
            out["shuffle_mean"]  = _np.nan
            out["shuffle_std"]   = _np.nan
            out["n_repeats"]     = 0
    return out


def compute_similarity_summary(
    registry: dict,
    *,
    # Override keys or pass callables if your field names differ:
    data_key: Any = "similarity",
    shuffle_repeat_key: Any = "sim_per_repeat",
    shuffle_matrix_key: Any = "sim_matrix",
) -> dict:
    """
    Build a compact summary aligned to your registry structure.

    Input (registry): variant -> group -> pair_key -> obj
                      OR -> {'data': obj, 'shuffle': obj}

    Output (summary):  variant -> group -> pair_key -> {
        'data_mean': float or nan,
        'n_pairs': int,
        'shuffle_means': 1D np.ndarray (per-repeat means),
        'shuffle_mean': float or nan,
        'shuffle_std': float,
        'n_repeats': int
    }

    You may pass:
      - data_key            = 'pv'                        # single key
      - data_key            = ['pv','similarity']         # first that exists
      - data_key            = lambda o: o['metrics']['pv'] # callable
      - shuffle_repeat_key  = 'repeat_means'
      - shuffle_matrix_key  = 'shuffles'
    """
    out: dict = {}
    for variant, groups in registry.items():
        out[variant] = {}
        for group, pairs in groups.items():
            out[variant][group] = {}
            for pair_key, entry in pairs.items():
                record = {
                    "data_mean": _np.nan,
                    "n_pairs": 0,
                    "shuffle_means": _np.asarray([], dtype=float),
                    "shuffle_mean": _np.nan,
                    "shuffle_std": _np.nan,
                    "n_repeats": 0,
                }

                if isinstance(entry, dict) and ("data" in entry or "shuffle" in entry):
                    if entry.get("data") is not None:
                        d = _summarize_similarity_obj(
                            entry["data"],
                            data_key=data_key,
                            shuffle_repeat_key=shuffle_repeat_key,
                            shuffle_matrix_key=shuffle_matrix_key,
                        )
                        for k in ("data_mean", "n_pairs"):
                            if k in d:
                                record[k] = d[k]
                    if entry.get("shuffle") is not None:
                        s = _summarize_similarity_obj(
                            entry["shuffle"],
                            data_key=data_key,
                            shuffle_repeat_key=shuffle_repeat_key,
                            shuffle_matrix_key=shuffle_matrix_key,
                        )
                        for k in ("shuffle_means", "shuffle_mean", "shuffle_std", "n_repeats"):
                            if k in s:
                                record[k] = s[k]
                else:
                    # treat as data-only
                    d = _summarize_similarity_obj(
                        entry,
                        data_key=data_key,
                        shuffle_repeat_key=shuffle_repeat_key,
                        shuffle_matrix_key=shuffle_matrix_key,
                    )
                    for k in ("data_mean", "n_pairs"):
                        if k in d:
                            record[k] = d[k]

                out[variant][group][pair_key] = record
    return out



# Convenience accessor: get data vs shuffle for a specific pair
def get_pair_similarity(summary, variant: str, group: str, src: str, dst: str):
    key = canonical_pair_key((src, dst))
    rec = summary[variant][group].get(key)
    if rec is None:
        raise KeyError(f"Not found: {variant}/{group}/{key}")
    return rec["data_mean"], rec["shuffle_means"], rec

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Same-animal alignment helpers (pairwise-ready)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as _np
from typing import Callable, List, Tuple, Dict, Any, Optional


def _reduce_values(vals: List[float], how: str | Callable[[List[float]], float] = "first") -> float:
    """Reduce a list of values for a (animal, category) cell to a single number."""
    arr = _np.asarray(vals, float)
    arr = arr[_np.isfinite(arr)]
    if arr.size == 0:
        return _np.nan
    if callable(how):
        return float(how(list(arr)))
    how = str(how).lower()
    if how == "first":  return float(arr[0])
    if how == "last":   return float(arr[-1])
    if how == "mean":   return float(_np.nanmean(arr))
    if how == "median": return float(_np.nanmedian(arr))
    if how == "max":    return float(_np.nanmax(arr))
    if how == "min":    return float(_np.nanmin(arr))
    # default
    return float(arr[0])

def _animal_from_pair_key(pair_key: str, *, animal_id_fn: Callable[[str], str], which_name: str = "src") -> str:
    """Derive animal ID from 'SRCâ†’DST' pair key using src (default) or dst name."""
    if "â†’" not in pair_key:
        return animal_id_fn(pair_key)
    src, dst = pair_key.split("â†’", 1)
    base = src if which_name == "src" else dst
    return animal_id_fn(base)

def _collect_map(summary: Dict[str, Dict[str, Dict[str, Any]]],
                 *, category: str, categories_is_group: bool,
                 fixed_variant: Optional[str], fixed_group: Optional[str],
                 field: str, animal_id_fn: Callable[[str], str], which_name: str) -> Dict[str, List[float]]:
    """
    Build {animal_id: [values...]} for one category (either a group or a variant).
    """
    out: Dict[str, List[float]] = {}
    if categories_is_group:
        # comparing groups within a fixed variant
        assert fixed_variant is not None
        if fixed_variant not in summary or category not in summary[fixed_variant]:
            return out
        bucket = summary[fixed_variant][category]
    else:
        # comparing variants within a fixed group
        assert fixed_group is not None
        if category not in summary or fixed_group not in summary[category]:
            return out
        bucket = summary[category][fixed_group]

    for pair_key, rec in bucket.items():
        aid = _animal_from_pair_key(pair_key, animal_id_fn=animal_id_fn, which_name=which_name)
        val = rec.get(field, _np.nan)
        if _np.isfinite(val):
            out.setdefault(aid, []).append(float(val))
    return out

def align_by_animal_groups(sim_summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                           *, variant: str,
                           groups: List[str],
                           field: str = "data_mean",
                           reducer: str | Callable[[List[float]], float] = "first",
                           animal_id_fn: Callable[[str], str] = animal_id_default,
                           which_name: str = "src",
                           require_all: bool = True) -> Tuple[List[str], _np.ndarray, List[str]]:
    """
    Align same-animal values across multiple GROUPS within one VARIANT.

    Returns:
      animals (list), matrix (n_animals x n_groups), columns == groups
    """
    # Per-group maps {animal: [vals...]}
    maps = {g: _collect_map(sim_summary, category=g, categories_is_group=True,
                            fixed_variant=variant, fixed_group=None, field=field,
                            animal_id_fn=animal_id_fn, which_name=which_name)
            for g in groups}

    # Animal set
    animal_sets = [set(m) for m in maps.values()]
    animals = sorted(set.intersection(*animal_sets)) if (require_all and animal_sets) \
              else sorted(set().union(*animal_sets))

    M = _np.full((len(animals), len(groups)), _np.nan, float)
    idx = {a: i for i, a in enumerate(animals)}
    for j, g in enumerate(groups):
        for a, vals in maps[g].items():
            i = idx.get(a)
            if i is not None:
                M[i, j] = _reduce_values(vals, reducer)
    return animals, M, groups

def align_by_animal_variants(sim_summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                             *, group: str,
                             variants: List[str],
                             field: str = "data_mean",
                             reducer: str | Callable[[List[float]], float] = "first",
                             animal_id_fn: Callable[[str], str] = animal_id_default,
                             which_name: str = "src",
                             require_all: bool = True) -> Tuple[List[str], _np.ndarray, List[str]]:
    """
    Align same-animal values across multiple VARIANTS within one GROUP.

    Returns:
      animals (list), matrix (n_animals x n_variants), columns == variants
    """
    maps = {v: _collect_map(sim_summary, category=v, categories_is_group=False,
                            fixed_variant=None, fixed_group=group, field=field,
                            animal_id_fn=animal_id_fn, which_name=which_name)
            for v in variants}

    animal_sets = [set(m) for m in maps.values()]
    animals = sorted(set.intersection(*animal_sets)) if (require_all and animal_sets) \
              else sorted(set().union(*animal_sets))

    M = _np.full((len(animals), len(variants)), _np.nan, float)
    idx = {a: i for i, a in enumerate(animals)}
    for j, v in enumerate(variants):
        for a, vals in maps[v].items():
            i = idx.get(a)
            if i is not None:
                M[i, j] = _reduce_values(vals, reducer)
    return animals, M, variants

def paired_from_alignment(animals: List[str], M: _np.ndarray, cols: List[str],
                          a: str, b: str) -> Tuple[_np.ndarray, _np.ndarray, List[str]]:
    """
    Extract paired vectors for columns named `a` and `b`, keeping only rows with both finite.
    """
    try:
        ia, ib = cols.index(a), cols.index(b)
    except ValueError as e:
        raise KeyError(f"Columns must contain {a!r} and {b!r}, got {cols}") from e
    mask = _np.isfinite(M[:, ia]) & _np.isfinite(M[:, ib])
    return M[mask, ia], M[mask, ib], [animals[i] for i in _np.where(mask)[0]]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Run align_by_animal_variants for *all* groups (plus optional tidy DataFrame)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as _np
from typing import Dict, List, Tuple, Any, Optional

def align_by_animal_variants_all_groups(
    sim_summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    *,
    variants: List[str],
    groups: Optional[List[str]] = None,     # if None -> auto from sim_summary
    field: str = "data_mean",
    reducer: str | callable = "first",
    animal_id_fn=animal_id_default,
    which_name: str = "src",
    require_all: bool = False,              # for animals: require presence in all columns?
    group_mode: str = "union",              # 'union' or 'intersection' across selected variants
    as_dataframe: bool = False,             # return (result, df)
):
    """
    For each group, align same-animal values across the given variants.

    Returns
    -------
    result : dict
        { group: (animals, M, cols) } where M has shape (n_animals, n_variants).
    df : pandas.DataFrame (optional when as_dataframe=True)
        Long/tidy table with columns: ['group','animal','variant', field]
        (only finite values are included).
    """
    # Determine which groups to process
    if groups is None:
        sets = [set(sim_summary.get(v, {}).keys()) for v in variants if v in sim_summary]
        if not sets:
            groups_list = []
        elif group_mode == "intersection":
            groups_list = sorted(set.intersection(*sets))
        else:
            groups_list = sorted(set.union(*sets))
    else:
        groups_list = list(groups)

    result: Dict[str, Tuple[List[str], _np.ndarray, List[str]]] = {}

    for g in groups_list:
        # Skip groups missing from all selected variants
        if not any(g in sim_summary.get(v, {}) for v in variants):
            continue
        animals, M, cols = align_by_animal_variants(
            sim_summary, group=g, variants=variants, field=field,
            reducer=reducer, animal_id_fn=animal_id_fn,
            which_name=which_name, require_all=require_all
        )
        result[g] = (animals, M, cols)

    if not as_dataframe:
        return result

    # Build tidy DataFrame if requested
    try:
        import pandas as _pd  # optional dependency
    except Exception:
        raise RuntimeError("pandas is required for as_dataframe=True")

    rows = []
    for g, (animals, M, cols) in result.items():
        for j, v in enumerate(cols):
            colvals = M[:, j]
            for i, a in enumerate(animals):
                val = float(colvals[i])
                if _np.isfinite(val):
                    rows.append({"group": g, "animal": a, "variant": v, field: val})
    df = _pd.DataFrame(rows, columns=["group", "animal", "variant", field])
    return result, df

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stack across groups: one big M_v (rows = groupÃ—animal, cols = variants)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as _np
from typing import Dict, List, Tuple, Any, Optional

def align_by_animal_variants_stacked(
    sim_summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    *,
    variants: List[str],                     # e.g. ["d1v1","d1v2","d2v1","d2v2"]
    groups: Optional[List[str]] = None,      # None -> union across variants
    field: str = "data_mean",                # or "shuffle_mean"
    reducer: str | callable = "first",
    animal_id_fn = None,                     # default = animal_id_default
    which_name: str = "src",                 # derive ID from src or dst session
    require_all: bool = False,               # require presence in all selected variants?
    group_mode: str = "union",               # 'union' or 'intersection' (when groups=None)
    as_dataframe: bool = False,              # return (M_v, cols, rows_meta, df)
):
    """
    Build ONE stacked matrix across all requested groups.

    Returns
    -------
    If as_dataframe=False (default):
        M_v        : np.ndarray, shape (sum_g n_animals_g, n_variants)
        cols       : list[str] = variants
        rows_meta  : list[dict] with keys {'group','animal'}
                     (same order as rows in M_v; same animal can appear multiple times)
    If as_dataframe=True:
        (M_v, cols, rows_meta, df) where df is a tidy pandas DataFrame with
        columns: ['group','animal','variant', field]
    """
    if animal_id_fn is None:
        try:
            animal_id_fn = animal_id_default
        except NameError:
            # fail-safe: digits-only
            import re as _aid_re
            _re_digits = _aid_re.compile(r"^(\d+)")
            def animal_id_fn(session_name: str) -> str:
                m = _re_digits.match(str(session_name))
                return m.group(1) if m else str(session_name)

    # Determine which groups to include
    if groups is None:
        sets = [set(sim_summary.get(v, {}).keys()) for v in variants if v in sim_summary]
        if not sets:
            groups_list = []
        elif group_mode == "intersection":
            groups_list = sorted(set.intersection(*sets))
        else:
            groups_list = sorted(set.union(*sets))
    else:
        groups_list = list(groups)

    # Stack matrices from each group
    row_groups: List[str] = []
    row_animals: List[str] = []
    blocks: List[_np.ndarray] = []

    for g in groups_list:
        # Skip groups absent from all requested variants
        if not any(g in sim_summary.get(v, {}) for v in variants):
            continue

        # Reuse the per-group aligner you already have
        animals, M, cols = align_by_animal_variants(
            sim_summary,
            group=g,
            variants=variants,
            field=field,
            reducer=reducer,
            animal_id_fn=animal_id_fn,
            which_name=which_name,
            require_all=require_all,
        )

        if M.size == 0:
            continue

        blocks.append(M)
        row_groups.extend([g] * len(animals))
        row_animals.extend(animals)

    if blocks:
        M_v = _np.vstack(blocks)
    else:
        M_v = _np.empty((0, len(variants)), dtype=float)

    rows_meta = [{"group": g, "animal": a} for g, a in zip(row_groups, row_animals)]
    cols = list(variants)

    if not as_dataframe:
        return M_v, cols, rows_meta

    # Optional tidy DataFrame
    try:
        import pandas as _pd
    except Exception as e:
        raise RuntimeError("pandas is required for as_dataframe=True") from e

    rows = []
    for (g, a), row in zip(zip(row_groups, row_animals), M_v):
        for v, val in zip(cols, row):
            if _np.isfinite(val):
                rows.append({"group": g, "animal": a, "variant": v, field: float(val)})
    df = _pd.DataFrame(rows, columns=["group","animal","variant",field])

    return M_v, cols, rows_meta, df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example usage (reference only; not executed on import)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# if __name__ == "__main__":
#     # 1) Singleâ€‘session
#     singles = {
#         "G1": ["163RLday", "325-6day1"],
#         "G2": ["1639RRday10"],
#     }
#     loaded_single = load_single_session_pickles(
#         singles,
#         data_roots=["/data/m1", "/data/m2"],
#         pattern_template="wtn_{name}_spatial.pkl",
#         workers=8,
#     )
#     print_load_summary(loaded_single, singles)

#     # Peek at one object
#     if loaded_single.get("G1"):
#         any_obj = next(iter(loaded_single["G1"].values()))
#         print(inspect_loaded_object(any_obj))

#     # 2) Crossâ€‘session
#     pairs = {
#         "X1": [
#             ("1639RRday10", "1639-6day1"),                   # tuple
#             {"src": "A", "dst": "B"},                       # mapping
#             "C_to_D",                                         # slug
#         ]
#     }
#     loaded_cross = load_cross_session_pickles(
#         pairs,
#         data_roots=["/data/m1", "/data/m2"],
#         pattern_template="tuning_{src}_to_{dst}.pkl",
#         workers=8,
#     )
#     print_load_summary(loaded_cross, pairs)

#     # Iterate
#     for g, key, obj in flatten_loaded(loaded_cross):
#         prev = inspect_loaded_object(obj)
#         print(g, key, prev)


# Public export surface (top-level callables/classes).
__all__ = [
    n for n, v in globals().items()
    if not n.startswith("_") and callable(v)
]
