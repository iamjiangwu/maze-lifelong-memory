from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Maze:
    le: int
    ru: list[list[int]]
    pa: np.ndarray
    ch: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    ce: dict[tuple[int, int], int]
    rc: np.ndarray | None
    di: np.ndarray | None
    cl: np.ndarray | None
    wa: np.ndarray | None
    st: np.ndarray | None


def RunIndex(c: int, m: Maze):
    for i, r in enumerate(m.ru):
        if int(c) in r:
            return i
    return None


def HomePath(c: int, m: Maze):
    ret = []
    i = int(m.rc[int(c)])
    ret += m.ru[i][m.ru[i].index(int(c)) :: -1]
    i = int(m.pa[i])
    while i >= 0:
        ret += m.ru[i][::-1]
        i = int(m.pa[i])
    return ret


def ConnectPath(c1: int, c2: int, m: Maze):
    r1 = HomePath(int(c1), m)
    r2 = HomePath(int(c2), m)[::-1]
    for i in r1:
        if i in r2:
            return r1[: r1.index(i)] + r2[r2.index(i) :]
    return [int(c1), int(c2)]


def ConnectDistance(m: Maze):
    nc = len(m.xc)
    return np.array(
        [[len(ConnectPath(c1, c2, m)) - 1 for c2 in range(nc)] for c1 in range(nc)],
        dtype=int,
    )


def MakeStepType(m: Maze):
    exitstate = len(m.ru)
    st = np.full((len(m.ru) + 1, len(m.ru) + 1), -1, dtype=int)
    for i in range(m.le + 1):
        for j in range(2**i - 1, 2 ** (i + 1) - 1):
            if j > 0:
                if (i + j + int(m.pa[j])) % 2 == 0:
                    st[j, int(m.pa[j])] = 2
                else:
                    st[j, int(m.pa[j])] = 3
            if i < m.le:
                for c in m.ch[j]:
                    if (i + j + int(c)) % 2 == 0:
                        st[j, int(c)] = 1
                    else:
                        st[j, int(c)] = 0
    st[0, exitstate] = 3
    return st


def MazeCenter(m: Maze):
    def _acc(i: int):
        r = m.ru[int(i)][:]
        if int(m.ch[int(i), 0]) != -1:
            r += _acc(int(m.ch[int(i), 0]))
            r += [m.ru[int(i)][-1]]
            r += _acc(int(m.ch[int(i), 1]))
            r += m.ru[int(i)][-1::-1]
        return r

    c = _acc(0)
    return np.array([m.xc[c], m.yc[c]]).T


def MazeWall(m: Maze):
    (xc, yc, ru, pa) = (m.xc, m.yc, m.ru, m.pa)
    ch = [np.where(np.asarray(pa) == i)[0].astype(int) for i in range(len(ru))]

    def _acw(i: int):
        r = ru[int(i)]
        c0 = np.array([xc[r[0]], yc[r[0]]], dtype=float)
        c1 = np.array([xc[r[-1]], yc[r[-1]]], dtype=float)
        if int(i) == 0:
            d = np.array([1, 0], dtype=float)
        else:
            p1 = np.array([xc[ru[int(pa[int(i)])][-1]], yc[ru[int(pa[int(i)])][-1]]], dtype=float)
            d = c0 - p1
        sw = 0.5 * np.array([-d[0] - d[1], d[0] - d[1]], dtype=float)
        se = 0.5 * np.array([-d[0] + d[1], -d[0] - d[1]], dtype=float)
        nw = 0.5 * np.array([d[0] - d[1], d[0] + d[1]], dtype=float)
        ne = 0.5 * np.array([d[0] + d[1], -d[0] + d[1]], dtype=float)
        p = [c0 + sw] if int(i) == 0 else []
        p += [c1 + sw]
        if len(ch[int(i)]):
            e = np.array([xc[ru[int(ch[int(i)][0])][0]], yc[ru[int(ch[int(i)][0])][0]]], dtype=float) - c1
            if np.array_equal(e, np.array([-d[1], d[0]], dtype=float)):
                il = int(ch[int(i)][0])
                ir = int(ch[int(i)][1])
            else:
                il = int(ch[int(i)][1])
                ir = int(ch[int(i)][0])
            p += _acw(il)
            p += [c1 + ne]
            p += _acw(ir)
            p += [c0 + se]
        else:
            p += [c1 + nw, c1 + ne, c1 + se]
        return p

    return np.asarray(_acw(0), dtype=float)


def NewMaze(n: int = 6):
    """
    Construct the standard binary-tree labyrinth maze.

    This is a native migration of the core MM_Maze_Utils.NewMaze behavior.
    """
    n = int(n)
    ru_xy = []
    pa = []
    for i in range(n + 1):
        xd = (i + 1) % 2
        yd = i % 2
        di = 2 ** ((n - i) // 2)
        for j in range(2**i):
            if i == 0:
                pa.append(-1)
                (x, y) = (int(2 ** (n / 2) - 1), int(2 ** (n / 2) - 1))
                ru_xy.append([(x1, y) for x1 in range(0, x + 1)])
            else:
                k = 2 ** (i - 1) - 1 + j // 2
                pa.append(k)
                (x0, y0) = ru_xy[k][-1]
                (xs, ys) = (xd * (2 * (j % 2) - 1), yd * (2 * (j % 2) - 1))
                (x, y) = (x0 + xs * di, y0 + ys * di)
                if xs == 0:
                    ru_xy.append([(x, y1) for y1 in range(y0 + ys, y + ys, ys)])
                else:
                    ru_xy.append([(x1, y) for x1 in range(x0 + xs, x + xs, xs)])

    ce = {}
    lo = {}
    c = 0
    for r in ru_xy:
        for p in r:
            ce[p] = c
            lo[c] = p
            c += 1
    nc = c
    ru = [[ce[p] for p in r] for r in ru_xy]
    pa = np.asarray(pa, dtype=int)

    ch = np.full((len(ru), 2), -1, dtype=int)
    for i, p in enumerate(pa):
        if int(p) >= 0:
            if ch[int(p), 0] == -1:
                ch[int(p), 0] = i
            else:
                ch[int(p), 1] = i

    xc = np.array([lo[cid][0] for cid in range(nc)], dtype=int)
    yc = np.array([lo[cid][1] for cid in range(nc)], dtype=int)
    ma = Maze(
        le=n,
        ru=ru,
        pa=pa,
        ch=ch,
        xc=xc,
        yc=yc,
        ce=ce,
        rc=None,
        di=None,
        cl=None,
        wa=None,
        st=None,
    )
    ma.rc = np.array([RunIndex(cid, ma) for cid in range(nc)], dtype=int)
    ma.di = ConnectDistance(ma)
    ma.cl = MazeCenter(ma)
    ma.wa = MazeWall(ma)
    ma.st = MakeStepType(ma)
    return ma


__all__ = [
    "Maze",
    "NewMaze",
    "RunIndex",
    "HomePath",
    "ConnectPath",
    "ConnectDistance",
    "MakeStepType",
    "MazeCenter",
    "MazeWall",
]
