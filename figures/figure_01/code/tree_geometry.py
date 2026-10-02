"""Binary-tree coordinates shared by the Figure 1/5 display convention."""
import numpy as np

GEOMETRY = {'tree_depth': 7, 'level_count': 7, 'tree_width': 12.55, 'tree_height': 3.6, 'tree_vertical_pad': 0.1, 'root_line_length': 1.15, 'tree_left_pad': 0.8, 'tree_right_pad': 0.1, 'maze_margin': 0.15}


def _compute_complete_binary_tree_positions(depth: int, tree_width: float, tree_height: float):
    pos = {}
    n_leaves = 2 ** (depth - 1)
    xs_leaves = np.linspace(0.0, tree_width, n_leaves)
    y_step = tree_height / (depth - 1) if depth > 1 else 0.0
    for i in range(n_leaves):
        pos[(depth - 1, i)] = (xs_leaves[i], 0.0)
    for level in range(depth - 2, -1, -1):
        n_nodes = 2 ** level
        for i in range(n_nodes):
            left_child = (level + 1, 2 * i)
            right_child = (level + 1, 2 * i + 1)
            x = 0.5 * (pos[left_child][0] + pos[right_child][0])
            y = pos[left_child][1] + y_step
            pos[(level, i)] = (x, y)
    return pos


def _horizontal_binary_tree_positions(depth: int, tree_width: float, tree_height: float):
    vertical = _compute_complete_binary_tree_positions(depth, tree_width, tree_height)
    y_center = 0.5 * tree_width
    return {
        key: (tree_height - xy[1], y_center - xy[0])
        for key, xy in vertical.items()
    }
