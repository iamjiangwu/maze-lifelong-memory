from __future__ import annotations

import csv
import difflib
import os
import pickle
import re
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import FPS, REWARD_NODE
from .maze import Maze


@dataclass
class Traj:
    fr: np.ndarray | list | None
    ce: list[np.ndarray] | None
    ke: list[np.ndarray] | None
    no: list[np.ndarray] | None
    re: list[np.ndarray] | None


def _install_pickle_traj_compat() -> None:
    """
    Provide MM_Traj_Utils.Traj symbol for legacy trajectory pickle compatibility.
    """
    if "MM_Traj_Utils" in sys.modules:
        return
    mod = types.ModuleType("MM_Traj_Utils")
    CompatTraj = dataclass(type("Traj", (), {}))
    CompatTraj.__module__ = "MM_Traj_Utils"
    CompatTraj.__annotations__ = {"fr": object, "ce": object, "ke": object, "no": object, "re": object}
    setattr(mod, "Traj", CompatTraj)
    sys.modules["MM_Traj_Utils"] = mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_traj_dirs() -> list[Path]:
    repo = _repo_root()
    raw_dirs = [
        Path.cwd(),
        Path.cwd() / "mm-trajectories",
        repo / "mm-trajectories",
        repo / "external" / "mouse-maze-analysis-master" / "mm-trajectories",
    ]
    # Optional extra search root for custom/local layouts.
    extra_root = os.environ.get("CA_TRAJ_EXTRA_ROOT", "").strip()
    if extra_root:
        er = Path(extra_root)
        raw_dirs.extend(
            [
                er / "mm-trajectories",
                er / "mouse-maze-analysis-master" / "mm-trajectories",
            ]
        )
    out = []
    seen = set()
    for d in raw_dirs:
        try:
            rp = d.resolve()
        except Exception:
            rp = d
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


@lru_cache(maxsize=1)
def _load_traj_alias_map() -> dict[str, str]:
    """
    Load optional trajectory alias mapping from CSV.

    File precedence:
    1) env var ``CA_TRAJ_ALIAS_FILE``
    2) ``<repo>/config/trajectory_aliases.csv``
    """
    repo = _repo_root()
    candidates = []
    env_file = os.environ.get("CA_TRAJ_ALIAS_FILE")
    if env_file:
        candidates.append(Path(env_file))
    candidates.append(repo / "config" / "trajectory_aliases.csv")

    for p in candidates:
        if not p.exists():
            continue
        mapping: dict[str, str] = {}
        try:
            with p.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and {"alias", "target"}.issubset(set(reader.fieldnames)):
                    for row in reader:
                        alias = (row.get("alias") or "").strip()
                        target = (row.get("target") or "").strip()
                        if alias and target:
                            mapping[alias] = target
                else:
                    f.seek(0)
                    for line in f:
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) >= 2 and parts[0] and parts[1]:
                            mapping[parts[0]] = parts[1]
        except Exception:
            continue
        if mapping:
            return mapping
    return {}


def _candidate_name_variants(name: str) -> list[str]:
    out = [name]
    if name.endswith("-tf"):
        out.append(name[:-3])
    else:
        out.append(name + "-tf")
    return out


def _maybe_alias_name(name: str) -> str:
    aliases = _load_traj_alias_map()
    for k in _candidate_name_variants(name):
        if k in aliases:
            return aliases[k]
    return name


@lru_cache(maxsize=1)
def list_available_trajectory_names() -> tuple[str, ...]:
    names = []
    for d in _candidate_traj_dirs():
        if not d.exists() or not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file():
                names.append(p.name)
    return tuple(sorted(set(names)))


def _token_match_candidates(token: str, names: tuple[str, ...]) -> list[str]:
    token = token.strip()
    if not token:
        return []
    matches = []
    for n in names:
        suffix = n.rsplit("_", 1)[-1]
        if n == token or suffix == token:
            matches.append(n)
            continue
        if suffix.startswith(token + "-") or n.endswith("_" + token):
            matches.append(n)
    return sorted(set(matches))


def _placeholder_like_token(token: str) -> bool:
    return bool(re.match(r"^[A-Za-z]\d+[a-z]?$", token))


def _resolve_traj_path(filename: str | Path):
    f = Path(filename)
    raw_name = f.name
    aliased_name = _maybe_alias_name(raw_name)

    search_names = [raw_name]
    if aliased_name != raw_name:
        search_names.append(aliased_name)

    candidates = [f, Path.cwd() / f]
    for d in _candidate_traj_dirs():
        for nm in search_names:
            candidates.append(d / nm)

    for c in candidates:
        if c.exists():
            return c

    names = list_available_trajectory_names()
    token = raw_name[:-3] if raw_name.endswith("-tf") else raw_name
    token = _maybe_alias_name(token)
    token_matches = _token_match_candidates(token, names)
    if len(token_matches) == 1:
        match = token_matches[0]
        for d in _candidate_traj_dirs():
            p = d / match
            if p.exists():
                return p

    # Optional placeholder policy: warn and return None.
    # Use only when porting legacy notebooks that still contain example IDs.
    policy = os.environ.get("CA_TRAJ_PLACEHOLDER_POLICY", "error").strip().lower()
    if token and _placeholder_like_token(token) and policy == "warn":
        print(
            f"[ca_analysis] WARN: placeholder-like trajectory id '{raw_name}' was not resolved; "
            "returning None due to CA_TRAJ_PLACEHOLDER_POLICY=warn."
        )
        return None

    hint_source = list(token_matches) if token_matches else list(
        difflib.get_close_matches(raw_name, names, n=6, cutoff=0.45)
    )
    hint = ", ".join(hint_source[:6]) if hint_source else "none"
    raise FileNotFoundError(
        "Trajectory file not found: "
        f"{filename}. "
        f"Closest available names: {hint}. "
        "If this is a placeholder nickname (e.g., B1, D7a), replace it with real session ids "
        "or provide an alias map via CA_TRAJ_ALIAS_FILE."
    )


def LoadTraj(filename):
    """
    Load a trajectory pickle from canonical or legacy mm-trajectories locations.
    """
    p = _resolve_traj_path(filename)
    if p is None:
        policy = os.environ.get("CA_TRAJ_PLACEHOLDER_POLICY", "error").strip().lower()
        if policy == "warn":
            return None
        raise FileNotFoundError(f"Trajectory file not found: {filename}")
    _install_pickle_traj_compat()
    with p.open("rb") as f:
        return pickle.load(f)


def FindPathsToNode(n, tr, ma: Maze):
    """
    Find monotonic paths to node n during trajectory tr.

    Returns ndarray rows: [bout_idx, frame_idx_in_bout, node_distance, absolute_frame].
    """
    n = int(n)
    wd = np.array([ma.di[r[-1], ma.ru[n][-1]] for r in ma.ru], dtype=float)
    ptn = []
    for i, b in enumerate(getattr(tr, "no", []) or []):
        b = np.asarray(b)
        if b.ndim != 2 or b.shape[0] <= 1:
            continue
        d = wd[b[:-1, 0].astype(int)]
        js = np.where(d[1:] == 0)[0] + 1
        for j in js:
            k = int(j - 1)
            while k >= 0 and d[k] > d[k + 1]:
                k -= 1
            k += 1
            ptn.append([i, k, int(j - k), int(b[k, 1]) + int(tr.fr[i, 0])])
    if not ptn:
        return np.zeros((0, 4), dtype=int)
    return np.asarray(ptn, dtype=int)


def TimeInMaze(f, tf):
    """
    Convert absolute frame index to cumulative in-maze seconds.
    """
    f = float(f)
    fr = np.asarray(tf.fr, dtype=float)
    if fr.ndim != 2 or fr.shape[1] < 2 or fr.shape[0] == 0:
        return np.nan
    ti = 0.0
    i = 0
    n = fr.shape[0]
    while i < n and fr[i, 1] < f:
        ti += float(fr[i, 1] - fr[i, 0])
        i += 1
    if i >= n:
        return float(ti / FPS)
    ti += float(f - fr[i, 0])
    return float(ti / FPS)


def SplitModeClips(tf, ma: Maze, re=True):
    """
    Split trajectory into leave/drink/explore mode clips.
    """
    water = REWARD_NODE
    leave, drink, explore = (0, 1, 2)
    if water >= len(ma.ru):
        water = max(0, len(ma.ru) - 1)
    ed = np.array([ma.di[r[-1], 0] for r in ma.ru], dtype=float) + 1.0
    wd = np.array([ma.di[r[-1], ma.ru[water][0]] for r in ma.ru], dtype=float) + 1.0

    clips = []
    for i, b in enumerate(getattr(tf, "no", []) or []):
        b = np.asarray(b)
        if b.ndim != 2 or b.shape[0] <= 1:
            continue
        nodes = b[:-1, 0].astype(int)
        if nodes.size == 0:
            continue
        edb = ed[nodes]
        wdb = wd[nodes]

        j = len(edb) - 1
        k = j - 1
        while k >= 0 and edb[k] > edb[k + 1]:
            k -= 1
        k += 1
        clips.append([i, k, int(j - k), leave])

        while k > 0:
            prev_k = int(k)
            if int(b[k, 0]) == water and bool(re):
                j = k
                k = j - 1
                while k >= 0 and wdb[k] > wdb[k + 1]:
                    k -= 1
                k += 1
                clip_len = int(j - k)
                if clip_len > 0:
                    clips.append([i, k, clip_len, drink])
            else:
                j = k
                k = j - 1
                while k >= 0 and int(b[k, 0]) != water:
                    k -= 1
                if k == -1:
                    k += 1
                clip_len = int(j - k)
                if clip_len > 0:
                    clips.append([i, k, clip_len, explore])

            # Guard against non-progress loops on degenerate bouts (e.g. repeated water node).
            if k >= prev_k:
                k = prev_k - 1
    if not clips:
        return np.zeros((0, 4), dtype=int)
    cl = np.asarray(clips, dtype=int)
    return cl[np.lexsort((cl[:, 1], cl[:, 0]))]


__all__ = [
    "Traj",
    "LoadTraj",
    "list_available_trajectory_names",
    "FindPathsToNode",
    "TimeInMaze",
    "SplitModeClips",
]
