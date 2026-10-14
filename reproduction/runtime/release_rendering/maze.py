"""Maze geometry used by the published schematics."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from ._validation import nonnegative_integer, positive_integer
CANONICAL_MAZE_DEPTH = 6

IntArray = NDArray[np.int_]


FloatArray = NDArray[np.float64]


MAX_MAZE_DEPTH = 8


def _validated_maze_depth(depth: object, *, name: str = "depth") -> int:
    """Bound maze depth before allocating dense geometry and distance arrays."""

    value = positive_integer(depth, name=name)
    if value > MAX_MAZE_DEPTH:
        raise ValueError(
            f"{name} must be at most {MAX_MAZE_DEPTH}; the experiment uses depth 6"
        )
    return value


@dataclass(slots=True)
class Maze:
    """Complete geometry and topology of one binary-tree maze.

    Attributes
    ----------
    depth:
        Number of branch levels.  The experiment uses depth 6.
    runs:
        Cell IDs along each run node, ordered from parent to child.
    parent, children:
        Binary-tree relationships among run nodes.
    cell_x, cell_y:
        Grid coordinates indexed by cell ID.
    run_for_cell:
        Run-node ID indexed by cell ID.
    cell_distance:
        Shortest maze distance, in grid steps, between every pair of cells.
    step_type:
        Directed run-to-run orientation codes used by the branch-policy model.
        Codes 0/1 enter the left/right child and codes 2/3 leave toward the
        parent with the two maze orientations kept distinct.
    """

    depth: int
    runs: tuple[tuple[int, ...], ...]
    parent: IntArray
    children: IntArray
    cell_x: IntArray
    cell_y: IntArray
    coordinate_to_cell: dict[tuple[int, int], int]
    run_for_cell: IntArray
    cell_distance: IntArray
    centerline_xy: FloatArray
    wall_xy: FloatArray
    step_type: IntArray

    @property
    def n_runs(self) -> int:
        """Number of run nodes, including internal and terminal runs."""

        return len(self.runs)

    @property
    def n_cells(self) -> int:
        """Number of occupied cells in the maze grid."""

        return int(self.cell_x.size)

    @property
    def exit_run(self) -> int:
        """Virtual run-node ID immediately outside the maze root."""

        return self.n_runs


def run_index_for_cell(cell: int, maze: Maze) -> int:
    """Return the run node containing ``cell``.

    Cells shared by adjacent drawing segments are assigned according to the
    canonical construction order, matching the stored trajectory parser.
    """

    cell_id = nonnegative_integer(cell, name="cell")
    if cell_id < 0 or cell_id >= maze.n_cells:
        raise IndexError(f"cell ID {cell_id} is outside 0..{maze.n_cells - 1}")
    return int(maze.run_for_cell[cell_id])


def cell_path_to_home(cell: int, maze: Maze) -> IntArray:
    """Return the shortest cell path from ``cell`` to the maze entrance."""

    cell_id = nonnegative_integer(cell, name="cell")
    run = run_index_for_cell(cell_id, maze)
    path = list(maze.runs[run][: maze.runs[run].index(cell_id) + 1][::-1])
    run = int(maze.parent[run])
    while run >= 0:
        path.extend(maze.runs[run][::-1])
        run = int(maze.parent[run])
    return np.asarray(path, dtype=int)


def cell_path_between(start_cell: int, end_cell: int, maze: Maze) -> IntArray:
    """Return the shortest connected cell path from start to end."""

    start_home = cell_path_to_home(start_cell, maze).tolist()
    home_end = cell_path_to_home(end_cell, maze)[::-1].tolist()
    for cell in start_home:
        if cell in home_end:
            left = start_home[: start_home.index(cell)]
            right = home_end[home_end.index(cell) :]
            return np.asarray(left + right, dtype=int)
    raise RuntimeError("maze tree paths did not share a common ancestor")


def run_path_to_root(run: int, maze: Maze) -> IntArray:
    """Return run nodes from the root through ``run``, both inclusive."""

    run_id = nonnegative_integer(run, name="run")
    if run_id < 0 or run_id >= maze.n_runs:
        raise IndexError(f"run ID {run_id} is outside 0..{maze.n_runs - 1}")
    reverse_path = [run_id]
    while run_id > 0:
        run_id = int(maze.parent[run_id])
        reverse_path.append(run_id)
    return np.asarray(reverse_path[::-1], dtype=int)


def run_path_between(start_run: int, end_run: int, maze: Maze) -> IntArray:
    """Return the unique tree path between two run nodes."""

    start = run_path_to_root(start_run, maze).tolist()
    end = run_path_to_root(end_run, maze).tolist()
    shared = 0
    for index, (left, right) in enumerate(zip(start, end)):
        if left != right:
            break
        shared = index
    return np.asarray(start[:shared:-1] + end[shared:], dtype=int)


def build_maze(depth: int = 6) -> Maze:
    """Construct the canonical binary-tree maze.

    ``depth=6`` produces the experimental 15 x 15 grid, 127 run nodes and 176
    occupied cells.  The construction is a typed, dependency-free port of the
    acquisition maze utility used to parse the stored trajectories.
    """

    level_count = _validated_maze_depth(depth)

    runs_xy: list[list[tuple[int, int]]] = []
    parents: list[int] = []
    for level in range(level_count + 1):
        x_direction = (level + 1) % 2
        y_direction = level % 2
        run_length = 2 ** ((level_count - level) // 2)
        for branch in range(2**level):
            if level == 0:
                parents.append(-1)
                center = int(2 ** (level_count / 2) - 1)
                runs_xy.append([(x, center) for x in range(center + 1)])
                continue

            parent = 2 ** (level - 1) - 1 + branch // 2
            parents.append(parent)
            x0, y0 = runs_xy[parent][-1]
            sign = 2 * (branch % 2) - 1
            x_step, y_step = x_direction * sign, y_direction * sign
            x1, y1 = x0 + x_step * run_length, y0 + y_step * run_length
            if x_step == 0:
                runs_xy.append(
                    [(x1, y) for y in range(y0 + y_step, y1 + y_step, y_step)]
                )
            else:
                runs_xy.append(
                    [(x, y1) for x in range(x0 + x_step, x1 + x_step, x_step)]
                )

    coordinate_to_cell: dict[tuple[int, int], int] = {}
    cell_to_coordinate: dict[int, tuple[int, int]] = {}
    next_cell = 0
    for run in runs_xy:
        for coordinate in run:
            coordinate_to_cell[coordinate] = next_cell
            cell_to_coordinate[next_cell] = coordinate
            next_cell += 1

    runs = tuple(
        tuple(coordinate_to_cell[coordinate] for coordinate in run) for run in runs_xy
    )
    parent = np.asarray(parents, dtype=int)
    children = np.full((len(runs), 2), -1, dtype=int)
    for run, parent_run in enumerate(parent):
        if parent_run < 0:
            continue
        empty = int(np.flatnonzero(children[parent_run] < 0)[0])
        children[parent_run, empty] = run

    cell_x = np.asarray(
        [cell_to_coordinate[cell][0] for cell in range(next_cell)], dtype=int
    )
    cell_y = np.asarray(
        [cell_to_coordinate[cell][1] for cell in range(next_cell)], dtype=int
    )

    # Every generated cell belongs to one run. Bind each cell to that run for
    # trajectory parsing and tree-depth assignment.
    run_for_cell = np.empty(next_cell, dtype=int)
    for run, cells in enumerate(runs):
        run_for_cell[np.asarray(cells, dtype=int)] = run

    partial = Maze(
        depth=level_count,
        runs=runs,
        parent=parent,
        children=children,
        cell_x=cell_x,
        cell_y=cell_y,
        coordinate_to_cell=coordinate_to_cell,
        run_for_cell=run_for_cell,
        cell_distance=np.empty((0, 0), dtype=int),
        centerline_xy=np.empty((0, 2), dtype=float),
        wall_xy=np.empty((0, 2), dtype=float),
        step_type=np.empty((0, 0), dtype=int),
    )
    partial.cell_distance = _cell_distance_matrix(partial)
    partial.step_type = _step_type_matrix(partial)
    partial.centerline_xy = _centerline(partial)
    partial.wall_xy = _wall_outline(partial)
    return partial


def _cell_distance_matrix(maze: Maze) -> IntArray:
    """Compute shortest-path distances between all maze cells."""

    return np.asarray(
        [
            [cell_path_between(a, b, maze).size - 1 for b in range(maze.n_cells)]
            for a in range(maze.n_cells)
        ],
        dtype=int,
    )


def _step_type_matrix(maze: Maze) -> IntArray:
    """Encode parent-child traversal types between tree runs."""

    matrix = np.full((maze.n_runs + 1, maze.n_runs + 1), -1, dtype=int)
    for level in range(maze.depth + 1):
        for run in range(2**level - 1, 2 ** (level + 1) - 1):
            if run > 0:
                parent = int(maze.parent[run])
                matrix[run, parent] = 2 if (level + run + parent) % 2 == 0 else 3
            if level < maze.depth:
                for child in maze.children[run]:
                    child_id = int(child)
                    matrix[run, child_id] = (
                        1 if (level + run + child_id) % 2 == 0 else 0
                    )
    matrix[0, maze.exit_run] = 3
    return matrix


def _centerline(maze: Maze) -> FloatArray:
    """Trace a continuous centerline through the complete binary tree."""

    def walk(run: int) -> list[int]:
        """Traverse one subtree and return its ordered cell indices."""

        cells = list(maze.runs[run])
        left = int(maze.children[run, 0])
        if left >= 0:
            right = int(maze.children[run, 1])
            cells.extend(walk(left))
            cells.append(maze.runs[run][-1])
            cells.extend(walk(right))
            cells.extend(maze.runs[run][::-1])
        return cells

    cells = np.asarray(walk(0), dtype=int)
    return np.column_stack((maze.cell_x[cells], maze.cell_y[cells])).astype(float)


def _wall_outline(maze: Maze) -> FloatArray:
    """Construct a counterclockwise wall outline around the maze tree."""

    children = [
        np.flatnonzero(maze.parent == run).astype(int) for run in range(maze.n_runs)
    ]

    def counterclockwise(run: int) -> list[FloatArray]:
        """Return outline segments around one run and its descendants."""

        cells = maze.runs[run]
        start = np.array([maze.cell_x[cells[0]], maze.cell_y[cells[0]]], dtype=float)
        end = np.array([maze.cell_x[cells[-1]], maze.cell_y[cells[-1]]], dtype=float)
        if run == 0:
            direction = np.array([1.0, 0.0])
        else:
            parent_end_cell = maze.runs[int(maze.parent[run])][-1]
            parent_end = np.array(
                [maze.cell_x[parent_end_cell], maze.cell_y[parent_end_cell]],
                dtype=float,
            )
            direction = start - parent_end

        south_west = 0.5 * np.array(
            [-direction[0] - direction[1], direction[0] - direction[1]]
        )
        south_east = 0.5 * np.array(
            [-direction[0] + direction[1], -direction[0] - direction[1]]
        )
        north_west = 0.5 * np.array(
            [direction[0] - direction[1], direction[0] + direction[1]]
        )
        north_east = 0.5 * np.array(
            [direction[0] + direction[1], -direction[0] + direction[1]]
        )
        points: list[FloatArray] = [start + south_west] if run == 0 else []
        points.append(end + south_west)
        if children[run].size:
            first, second = (int(value) for value in children[run])
            first_cell = maze.runs[first][0]
            first_direction = (
                np.array(
                    [maze.cell_x[first_cell], maze.cell_y[first_cell]], dtype=float
                )
                - end
            )
            if np.array_equal(first_direction, np.array([-direction[1], direction[0]])):
                left, right = first, second
            else:
                left, right = second, first
            points.extend(counterclockwise(left))
            points.append(end + north_east)
            points.extend(counterclockwise(right))
            points.append(start + south_east)
        else:
            points.extend((end + north_west, end + north_east, end + south_east))
        return points

    return np.asarray(counterclockwise(0), dtype=float)

