from __future__ import annotations

# Auto-migrated local copy from legacy source.
# This module is self-contained within the repository.

import os

import numpy as np
from scipy.io import loadmat
from scipy.special import erfinv, gammaln

try:
    import cv2
except Exception:
    cv2 = None

import csv
import glob
import inspect
import itertools
import math
import pickle
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import make_dataclass
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
)

import h5py
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import pandas as pd
import scipy.io
import scipy.sparse as sp
import scipy.sparse as sps
from matplotlib import cm, cycler, patches
from matplotlib import pyplot as plt
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    is_color_like,
    to_hex,
    to_rgb,
)
from matplotlib.widgets import Slider
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.stats import rankdata, spearmanr, zscore
from .stats import holm_correction as _holm_correction_canonical
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from sklearn.feature_selection import mutual_info_regression as mi_skl
from tqdm import tqdm

from .data_loader_utils import DataLoader
from .mm_maze_utils import *
from .mm_maze_utils import NewMaze
from .mm_plot_utils import hist, plot
from .mm_traj import *
from .mm_traj import LoadTraj, SplitModeClips
from .query_data_utils import QueryData

try:
    from track_linearization import (
        get_linearized_position,
        make_track_graph,
        plot_track_graph,
    )
    from track_linearization.utils import plot_graph_as_1D
except Exception:
    make_track_graph = None
    plot_track_graph = None
    plot_graph_as_1D = None

    def get_linearized_position(*args, **kwargs):
        raise ImportError(
            "track_linearization is required for linearized-position utilities. "
            "Install it to use build_1d_session and related functions."
        )


# Function discovery helpers
_CATEGORY_KEYWORDS = {
    "io_loading": {"load", "read", "save", "pickle", "path", "file", "session"},
    "plotting": {"plot", "draw", "scatter", "hist", "figure", "color", "style"},
    "similarity_stats": {
        "similarity",
        "corr",
        "correlation",
        "zscore",
        "shuffle",
        "sem",
        "paired",
        "gini",
        "sparsity",
    },
    "mapping_spatial": {
        "map",
        "spatial",
        "maze",
        "bin",
        "position",
        "level",
        "cluster",
        "linearized",
    },
    "ordering_classification": {
        "order",
        "classify",
        "refine",
        "peak",
        "seriation",
    },
    "pipeline_analysis": {
        "process",
        "run",
        "compute",
        "collect",
        "stack",
        "filter",
        "apply",
        "build",
    },
    "utility": set(),
}


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _normalize_tokens(text):
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _function_catalog(include_private=False):
    """
    Build lightweight metadata for top-level functions in this module.
    Returns:
        dict: {function_name: {"line": int, "category": str, "tokens": set[str], "doc": str}}
    """
    catalog = {}
    for name, obj in globals().items():
        if not callable(obj):
            continue
        if getattr(obj, "__module__", None) != __name__:
            continue
        if not include_private and name.startswith("_"):
            continue
        if not hasattr(obj, "__code__"):
            continue
        doc = inspect.getdoc(obj) or ""
        name_tokens = set(_normalize_tokens(name))
        doc_tokens = set(_normalize_tokens(doc))
        tokens = name_tokens | doc_tokens
        # Strong prefix rules first for stable categorization.
        if name.startswith(("plot_", "draw_", "scatter_", "hist_")):
            best_cat = "plotting"
        elif name.startswith(("load_", "read_", "save_")):
            best_cat = "io_loading"
        elif name.startswith(("order_", "classify_", "refine_")):
            best_cat = "ordering_classification"
        elif name.startswith(("compute_", "process_", "run_", "collect_", "stack_")):
            best_cat = "pipeline_analysis"
        else:
            best_cat = "utility"
            best_score = 0
            for cat, kws in _CATEGORY_KEYWORDS.items():
                if not kws:
                    continue
                # Prioritize function name semantics over doc wording.
                score = 3 * len(name_tokens & kws) + len(doc_tokens & kws)
                if score > best_score:
                    best_score = score
                    best_cat = cat
        catalog[name] = {
            "line": obj.__code__.co_firstlineno,
            "category": best_cat,
            "tokens": tokens,
            "doc": doc.splitlines()[0] if doc else "",
        }
    return catalog


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def find_related_functions(query, top_k=12, include_private=False):
    """
    Find likely candidate functions for a task/query.
    Parameters:
        query (str): Natural-language intent or function-type hint.
        top_k (int): Number of candidates to return.
        include_private (bool): If True, include functions prefixed with "_".
    Returns:
        list[dict]: Ranked candidates with keys:
            - name
            - score
            - category
            - line
            - summary
    """
    q_tokens = set(_normalize_tokens(query))
    if not q_tokens:
        return []
    catalog = _function_catalog(include_private=include_private)
    scored = []
    for name, meta in catalog.items():
        overlap = len(q_tokens & meta["tokens"])
        if overlap == 0:
            continue
        # Light bias for prefix/substring matches in function name.
        name_l = name.lower()
        prefix_bonus = 2 if any(name_l.startswith(t) for t in q_tokens) else 0
        contains_bonus = 1 if any(t in name_l for t in q_tokens) else 0
        score = overlap + prefix_bonus + contains_bonus
        scored.append(
            {
                "name": name,
                "score": score,
                "category": meta["category"],
                "line": meta["line"],
                "summary": meta["doc"],
            }
        )
    scored.sort(key=lambda x: (-x["score"], x["name"]))
    return scored[: max(1, int(top_k))]


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def has_related_function(query, include_private=False, min_score=2):
    """
    Quick boolean check for whether related functions likely exist.
    """
    candidates = find_related_functions(
        query=query, top_k=1, include_private=include_private
    )
    if not candidates:
        return False
    return candidates[0]["score"] >= min_score


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin]
def arena_binned(x, y, xdim, ydim, pix_cm, pf_cm):
    """
    Functionality:
    - Arena binned.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `y`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `xdim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `ydim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pix_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pf_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 6 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    d_x = xdim[0] - xdim[1]  # xdim[0] = max, xdim[1] = min
    d_y = ydim[0] - ydim[1]  # ydim[0] = max, ydim[1] = min
    nBnx = int((d_x / (pix_cm * pf_cm)))
    nBny = int((d_y / (pix_cm * pf_cm)))
    bn_x = [i for i in np.linspace(xdim[0], xdim[1], nBnx + 1)]
    bn_y = [i for i in np.linspace(ydim[0], ydim[1], nBny + 1)]
    bx = ((x - bn_x[0]) / (bn_x[1] - bn_x[0])).astype(int)
    by = ((y - bn_y[0]) / (bn_y[1] - bn_y[0])).astype(int)
    S = np.vstack((bx, by))
    linS = np.ravel_multi_index(S, (nBnx, nBny))
    occMap = sps.csr_matrix(
        (np.ones(len(bx)), (bx, by)), shape=(nBnx, nBny), dtype=float
    ).todense()
    return linS.astype(float), occMap, nBnx, nBny, bx, by


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_occMap_with_start_and_end_time_single_cell(
    selected_df, start_frame, end_frame
):
    """
    Functionality:
    - Generate occMap with start and end time single cell.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `occMap` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    temp_df = selected_df[
        (selected_df["frame"] > start_frame) & (selected_df["frame"] < end_frame)
    ].copy()
    # temp_df = selected_df.copy()
    y = 1 - np.array(temp_df.x)
    x = 1 - np.array(temp_df.y)
    xdim = [1, 0]  # Set arena x dimensions (northwest - (-20,-20), 320x320)
    ydim = [1, 0]  # Set Arena y dimensions
    pix_cm = 1 / 15
    pf_cm = 1 / 3
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    return occMap


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_placeMap_with_start_and_end_time_single_cell(
    selected_df, S_trans, start_frame, end_frame
):
    """
    Functionality:
    - Generate placeMap with start and end time single cell.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `placeMap` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    temp_df = selected_df[
        (selected_df["frame"] > start_frame) & (selected_df["frame"] < end_frame)
    ].copy()
    # temp_df = selected_df.copy()
    y = 1 - np.array(temp_df.x)
    x = 1 - np.array(temp_df.y)
    xdim = [1, 0]  # Set arena x dimensions (northwest - (-20,-20), 320x320)
    ydim = [1, 0]  # Set Arena y dimensions
    pix_cm = 1 / 15
    pf_cm = 1 / 3
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    s_idx = np.array(temp_df["frame"]).astype(int)
    S_selected_4x = S_trans[s_idx].T
    nonnan_idx = np.where(~np.isnan(S_selected_4x))[0]
    S_selected_4x = S_selected_4x[nonnan_idx]
    bx = bx[nonnan_idx]
    by = by[nonnan_idx]
    rateMap = sps.csr_matrix(
        (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
    ).todense()
    placeMap = np.divide(rateMap, occMap)
    # placeMap[np.isnan(placeMap)] = 0
    return placeMap


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def auto_threshold(img, sigma=0):
    # Apply Gaussian filter with specified sigma
    """
    Functionality:
    - Auto threshold.
    Inputs:
    - `img`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `sigma`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `binary` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    img_g = gaussian(img, sigma=sigma)
    # Calculate threshold value using Otsu's method on the Gaussian filtered image
    thresh = threshold_otsu(img_g)
    # Create binary array by setting pixel values above threshold to 1 and below threshold to 0
    binary = np.zeros_like(img_g)
    binary[img_g > thresh] = 1
    return binary


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_connected_regions(binary, min_size=4, max_size=np.inf):
    # Label connected regions in binary image
    """
    Functionality:
    - Get connected regions.
    Inputs:
    - `binary`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_size`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `max_size`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `regions` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    labeled = label(binary)
    # Get properties of labeled regions
    props = regionprops(labeled)
    # Create list of connected regions
    regions = []
    for prop in props:
        # Get indices of pixels in current region
        indices = prop.coords
        # Check if current region size is within specified range
        size = prop.area
        if size >= min_size and size <= max_size:
            # Add indices to list of connected regions
            regions.append(indices)
    return regions


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_regions(binary, regions):
    # Create plot and set limits
    """
    Functionality:
    - Plot regions.
    Inputs:
    - `binary`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `regions`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, binary.shape[1])
    ax.set_ylim(0, binary.shape[0])
    # Plot binary image
    ax.imshow(binary, cmap="gray")
    # Plot each region as a colored patch
    for i, region in enumerate(regions):
        patch = plt.Polygon(
            region[:, ::-1], closed=True, edgecolor="r", facecolor="none", linewidth=2
        )
        ax.add_patch(patch)
    plt.show()


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def centroid(arr):
    """
    Functionality:
    - Centroid results.
    Inputs:
    - `arr`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x_coords = arr[:, 0]
    y_coords = arr[:, 1]
    x_centroid = np.sum(x_coords) / len(x_coords)
    y_centroid = np.sum(y_coords) / len(y_coords)
    return (x_centroid, y_centroid)


# get the peak of the place field


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: peak]
def get_peak_place_field(placeMap_g, regions):
    """
    Functionality:
    - Get peak place field.
    Inputs:
    - `placeMap_g`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `regions`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `peak` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    peak = []
    for region in regions:
        peak_ind = np.argmax(placeMap_g[region[:, 0], region[:, 1]])
        peak.append(list(region[peak_ind]))
    return peak


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def regions_centroids(regions):
    """
    Functionality:
    - Regions centroids.
    Inputs:
    - `regions`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `centroids` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    centroids = []
    for region in regions:
        centroids.append(centroid(region))
    return centroids


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def regions_size(regions):
    """
    Functionality:
    - Regions size.
    Inputs:
    - `regions`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `sizes` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    sizes = []
    for region in regions:
        sizes.append(len(region))
    return sizes


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_region_frames(region, placeMap_g, selected_df):
    """
    Functionality:
    - Get region frames.
    Inputs:
    - `region`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `placeMap_g`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    region1D = np.ravel_multi_index(region.T, placeMap_g.shape)
    region_df = selected_df[selected_df["linS"].isin(region1D)]
    return region_df["frame"].values.astype(int)


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_region_frames_head(region, head_cat, placeMap_g, selected_df):
    """
    Functionality:
    - Get region frames head.
    Inputs:
    - `region`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `head_cat`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `placeMap_g`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    region = np.vstack((region.T, np.zeros(region.shape[0])))
    region[-1] = head_cat
    region = region.astype(int)
    region1D = np.ravel_multi_index(
        region, (placeMap_g.shape[0], placeMap_g.shape[1], 8)
    )
    region_df = selected_df[selected_df["linS"].isin(region1D)]
    return region_df["frame"].values.astype(int)


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_regional_spiketrain(region_frames, S_4x_nn_single):
    """
    Functionality:
    - Get regional spiketrain.
    Inputs:
    - `region_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `S_4x_nn_single`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    return S_4x_nn_single[region_frames]


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_start_end_time_spike_train(spiketrain, region_frames):
    """
    Functionality:
    - Get start end time spike train.
    Inputs:
    - `spiketrain`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `region_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    start_ind = np.where(spiketrain == 1)[0][0]
    end_ind = np.where(spiketrain == 1)[0][-1]
    start_time = region_frames[start_ind]
    end_time = region_frames[end_ind]
    return start_time, end_time


# time in recording to time in maze


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_placeMap_with_start_and_end_time_single_cell_m(
    selected_df, S_trans, start_frame, end_frame
):
    """
    Functionality:
    - Generate placeMap with start and end time single cell m.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `placeMap` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    temp_df = selected_df[
        (selected_df["maze_time"] > start_frame / 25)
        & (selected_df["maze_time"] < end_frame / 25)
    ].copy()
    # temp_df = selected_df.copy()
    y = 1 - np.array(temp_df.x)
    x = 1 - np.array(temp_df.y)
    xdim = [1, 0]  # Set arena x dimensions (northwest - (-20,-20), 320x320)
    ydim = [1, 0]  # Set Arena y dimensions
    pix_cm = 1 / 15
    pf_cm = 1 / 3
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    s_idx = np.array(temp_df["frame"]).astype(int)
    S_selected_4x = S_trans[s_idx].T
    rateMap = sps.csr_matrix(
        (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
    ).todense()
    placeMap = np.divide(rateMap, occMap)
    placeMap[np.isnan(placeMap)] = 0
    # placeMap = placeMap/np.sum(placeMap)
    return placeMap


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_placeMap_with_start_and_end_time_single_cell_m_sim(
    selected_df, S_trans, start_frame, end_frame
):
    """
    Functionality:
    - Generate placeMap with start and end time single cell m sim.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `placeMap` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    temp_df = selected_df[
        (selected_df["maze_time"] > start_frame / 25)
        & (selected_df["maze_time"] < end_frame / 25)
    ].copy()
    # temp_df = selected_df.copy()
    y = 1 - np.array(temp_df.x)
    x = 1 - np.array(temp_df.y)
    xdim = [1, 0]  # Set arena x dimensions (northwest - (-20,-20), 320x320)
    ydim = [1, 0]  # Set Arena y dimensions
    pix_cm = 1 / 15
    pf_cm = 1 / 3
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    # s_idx = np.array(temp_df['frame']).astype(int)
    s_idx = temp_df.index
    S_selected_4x = S_trans[s_idx].T
    rateMap = sps.csr_matrix(
        (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
    ).todense()
    placeMap = np.divide(rateMap, occMap)
    placeMap[np.isnan(placeMap)] = 0
    # placeMap = placeMap/np.sum(placeMap)
    return placeMap


# calculation correlation coefficient when a location was visited prior and post


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr, correlation]
def calculate_correlation_coefficient_2D_matrix_p(A, B):
    """
    Calculates the correlation coefficient between two 2D matrices.
    Args:
        A: A 2D matrix.
        B: A 2D matrix.
    Returns:
        The correlation coefficient between the two matrices.
    """
    A = np.reshape(A, (A.shape[0] * A.shape[1]))
    B = np.reshape(B, (B.shape[0] * B.shape[1]))
    ind = np.where((A != -1) & (B != -1))
    A = A[ind]
    B = B[ind]
    return np.corrcoef(A, B)[0, 1]


# calculate head direction based on the nose and mass position


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_head_dir(x, y, mass_x, mass_y):
    """
    Functionality:
    - Get head dir.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `y`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `mass_x`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `mass_y`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    return np.arctan2(mass_y - y, mass_x - x)


# group the head direction to 8 categories


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_head_cat(head_dir):
    """input: a column of head direction in radians"""
    head_cat = np.zeros(len(head_dir))
    for i in range(len(head_dir)):
        if (head_dir[i] < 7 / 8 * np.pi) & (head_dir[i] >= 5 / 8 * np.pi):
            head_cat[i] = 1
        elif (head_dir[i] < 5 / 8 * np.pi) & (head_dir[i] >= 3 / 8 * np.pi):
            head_cat[i] = 2
        elif (head_dir[i] < 3 / 8 * np.pi) & (head_dir[i] >= 1 / 8 * np.pi):
            head_cat[i] = 3
        elif (head_dir[i] < 1 / 8 * np.pi) & (head_dir[i] >= -1 / 8 * np.pi):
            head_cat[i] = 4
        elif (head_dir[i] < -1 / 8 * np.pi) & (head_dir[i] >= -3 / 8 * np.pi):
            head_cat[i] = 5
        elif (head_dir[i] < -3 / 8 * np.pi) & (head_dir[i] >= -5 / 8 * np.pi):
            head_cat[i] = 6
        elif (head_dir[i] < -5 / 8 * np.pi) & (head_dir[i] >= -7 / 8 * np.pi):
            head_cat[i] = 7
        else:
            head_cat[i] = 0
    return head_cat.astype(int)


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin]
def generate_placeMap_with_start_and_end_time_single_cell_cat(
    selected_df, S_trans, start_frame=0, end_frame=np.inf, nBnx=15, nBny=15
):
    """
    Functionality:
    - Generate placeMap with start and end time single cell cat.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `nBnx`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `nBny`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    slice_df = selected_df[
        (selected_df["frame"] > start_frame) & (selected_df["frame"] < end_frame)
    ].copy()
    placeMap = np.zeros((8, nBnx, nBny))
    for k in range(8):
        temp_df = slice_df[slice_df["head_cat"] == k].copy()
        # temp_df = selected_df.copy()
        y = 1 - np.array(temp_df.x)
        x = 1 - np.array(temp_df.y)
        xdim = [1, 0]  # Set arena x dimensions (northwest - (-20,-20), 320x320)
        ydim = [1, 0]  # Set Arena y dimensions
        pix_cm = 1 / 15
        pf_cm = 1
        linS, occMap, nBnx, nBny, bx, by = arena_binned(
            x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
        )
        s_idx = np.array(temp_df["frame"]).astype(int)
        S_selected_4x = S_trans[s_idx].T
        rateMap = sps.csr_matrix(
            (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
        ).todense()
        placeMap[k] = np.divide(rateMap, occMap)
        # placeMap[np.isnan(placeMap)] = 0
        nn_placeMap = placeMap.copy()
        nn_placeMap[np.isnan(nn_placeMap)] = 0
    return placeMap, nn_placeMap


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: filter]
def gaussian_filter_xy(center_x, center_y, width_x, width_y, shape):
    """
    Functionality:
    - Gaussian filter xy.
    Inputs:
    - `center_x`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `center_y`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `width_x`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `width_y`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `shape`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(
        -(
            ((xx - center_x) ** 2) / (2 * width_x**2)
            + ((yy - center_y) ** 2) / (2 * width_y**2)
        )
    )
    return gaussian_filter(gaussian, 1)


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_multi_gaussian_map(centers, widths, rate_modulator, shape):
    """
    Functionality:
    - Generate multi gaussian map.
    Inputs:
    - `centers`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `widths`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `rate_modulator`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `shape`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `gaussian_map` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.arange(0, shape[1], 1, float)
    y = np.arange(0, shape[0], 1, float)
    xx, yy = np.meshgrid(x, y)
    gaussian_map = np.zeros(shape)
    for center, width, r in zip(centers, widths, rate_modulator):
        # add a small value to avoid division by zero
        width = width + 0.00001
        gaussian_map += r * np.exp(
            -(
                ((xx - center[0]) ** 2) / (2 * width[0] ** 2)
                + ((yy - center[1]) ** 2) / (2 * width[1] ** 2)
            )
        )
    return gaussian_map


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def calculate_width_xy(coords):
    # Calculate the center of mass
    """
    Functionality:
    - Calculate width xy.
    Inputs:
    - `coords`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    center = np.mean(coords, axis=0)
    # Subtract the center of mass from the coordinates
    coords_centered = coords - center
    # Calculate the covariance matrix
    cov = np.cov(coords_centered.T)
    # Extract the variances from the covariance matrix
    var_x = cov[0, 0]
    var_y = cov[1, 1]
    # Calculate the standard deviations
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    return std_x, std_y


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def var_poiss(T, R, n, s):
    """
    Samples a variable rate Poisson process
    T = duration of the process
    R = instantaneous rate
    n = number of trials to simulate
    s = random seed
    Returns: n lists of spike times, one for each trial
    """
    from scipy import interpolate
    dt = T / (len(R))  # time resolution
    t = np.linspace(0, T, len(R))  # time axis
    u = (
        np.insert(np.cumsum((R[:-1] + R[1:]) / 2), 0, 0) * dt
    )  # integrate rate to get warp time
    U = u[-1]  # duration in warp time
    f = interpolate.interp1d(u, t)  # inverse function u^-1
    np.random.seed(s)  # set random seed
    m = np.random.poisson(U, n)  # number of events in each trial
    return [
        np.sort(f(np.random.uniform(0, U, k))) for k in m
    ]  # list of arrays of event times


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def var_poiss_spiketrain(T, R, n, s):
    """
    Samples a variable rate Poisson process
    T = duration of the process
    R = instantaneous rate
    n = number of trials to simulate
    s = random seed
    Returns: n lists of spike times, one for each trial
    """
    from scipy import interpolate
    dt = T / (len(R) - 1)  # time resolution
    t = np.linspace(0, T, len(R))  # time axis
    u = (
        np.insert(np.cumsum((R[:-1] + R[1:]) / 2), 0, 0) * dt
    )  # integrate rate to get warp time
    U = u[-1]  # duration in warp time
    f = interpolate.interp1d(u, t)  # inverse function u^-1
    np.random.seed(s)  # set random seed
    m = np.random.poisson(U, n)  # number of events in each trial
    spiketrains = np.zeros((n, len(R)), dtype=int)
    for i, k in enumerate(m):
        spike_indices = np.round(f(np.random.uniform(0, U, k)) / dt).astype(int)
        spiketrains[i, spike_indices] = 1
    return spiketrains


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def var_poiss_firingrate(T, R, n, s):
    """
    Samples a variable rate Poisson process and returns firing rate trains.
    T = duration of the process
    R = instantaneous rate
    n = number of trials to simulate
    s = random seed
    Returns: n firing rate trains, one for each trial
    """
    from scipy import interpolate
    dt = T / (len(R) - 1)  # time resolution
    t = np.linspace(0, T, len(R))  # time axis
    u = (
        np.insert(np.cumsum((R[:-1] + R[1:]) / 2), 0, 0) * dt
    )  # integrate rate to get warp time
    U = u[-1]  # duration in warp time
    f = interpolate.interp1d(u, t)  # inverse function u^-1
    np.random.seed(s)  # set random seed
    m = np.random.poisson(U, n)  # number of events in each trial
    firingrate_trains = np.zeros((n, len(R)), dtype=float)
    for i, k in enumerate(m):
        spike_times = f(np.random.uniform(0, U, k))
        # Convert spike times to firing rates by dividing by time bin width
        spike_indices = np.round(spike_times / dt).astype(int)
        # Compute firing rates as number of spikes in each time bin divided by time bin width
        firingrate, _ = np.histogram(spike_indices, bins=len(R), range=(0, len(R)))
        firingrate = firingrate / dt
        firingrate_trains[i] = firingrate
    return firingrate_trains


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: process]
def process_mouse_positions(data, video, nickname, tr):
    """
    Functionality:
    - Process mouse positions.
    Inputs:
    - `data`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `video`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `nickname`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `tr`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `nose_interp_df` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    nose_pos = np.array(data["warped numpy"][video[nickname]])[:, 0, :]
    mass_pos = np.array(data["warped numpy"][video[nickname]])[:, -1, :]
    nose_df = pd.DataFrame(nose_pos, columns=["x", "y", "confidence"])
    nose_df = nose_df.reset_index()
    nose_df = nose_df.rename(columns={"index": "frame"})
    nose_df["mass_x"] = mass_pos[:, 0]
    nose_df["mass_y"] = mass_pos[:, 1]
    nose_xy_df = nose_df[~nose_df.isnull().any(axis=1)].copy()
    nose_xy_df.loc[:, "x_interp"] = nose_xy_df["x"]
    nose_xy_df.loc[:, "y_interp"] = nose_xy_df["y"]
    nose_xy_df.loc[nose_xy_df["confidence"] < 0.95, "x_interp"] = np.nan
    nose_xy_df.loc[nose_xy_df["confidence"] < 0.95, "y_interp"] = np.nan
    nose_xy_df["x_interp"] = nose_xy_df["x_interp"].interpolate(method="linear")
    nose_xy_df["y_interp"] = nose_xy_df["y_interp"].interpolate(method="linear")
    nose_interp_df = pd.DataFrame(
        columns=["frame", "x_interp", "y_interp", "mass_x", "mass_y"]
    )
    num_bout = 0
    for s_fr, e_fr in tr.fr:
        nose_interp_df = pd.concat(
            [
                nose_interp_df,
                nose_xy_df.loc[
                    s_fr:e_fr, ["frame", "x_interp", "y_interp", "mass_x", "mass_y"]
                ],
            ],
            ignore_index=True,
        )
        nose_interp_df.loc[
            nose_interp_df["frame"].isin(range(s_fr, e_fr + 1)), "bout"
        ] = num_bout
        num_bout += 1
    nose_interp_df["speed"] = np.nan
    for bout in range(int(num_bout)):
        nose_bout_df = nose_interp_df.loc[nose_interp_df["bout"] == bout]
        nose_interp_df.loc[nose_interp_df["bout"] == bout, "dist"] = np.sqrt(
            (nose_bout_df["x_interp"].diff() * 57) ** 2
            + (nose_bout_df["y_interp"].diff() * 57) ** 2
        )
        nose_interp_df.loc[nose_interp_df["bout"] == bout, "speed"] = (
            nose_interp_df["dist"] * 25 / (nose_bout_df["frame"].diff())
        )
    nose_interp_df.loc[0, "speed"] = 0
    nose_interp_df.loc[nose_interp_df["bout"].diff() == 1, "speed"] = 0
    nose_interp_df = nose_interp_df.rename(columns={"x_interp": "x", "y_interp": "y"})
    # if x or y is smaller than 0, set it to a small value
    nose_interp_df.loc[nose_interp_df["x"] < 0, "x"] = 0.001
    nose_interp_df.loc[nose_interp_df["y"] < 0, "y"] = 0.001
    nose_interp_df.loc[nose_interp_df["mass_x"] < 0, "mass_x"] = 0.001
    nose_interp_df.loc[nose_interp_df["mass_y"] < 0, "mass_y"] = 0.001
    nose_interp_df.loc[nose_interp_df["x"] > 1, "x"] = 0.999
    nose_interp_df.loc[nose_interp_df["y"] > 1, "y"] = 0.999
    nose_interp_df.loc[nose_interp_df["mass_x"] > 1, "mass_x"] = 0.999
    nose_interp_df.loc[nose_interp_df["mass_y"] > 1, "mass_y"] = 0.999
    head_dir = get_head_dir(
        nose_interp_df["x"],
        nose_interp_df["y"],
        nose_interp_df["mass_x"],
        nose_interp_df["mass_y"],
    )
    head_cat = get_head_cat(head_dir)
    nose_interp_df["head_dir"] = head_dir
    nose_interp_df["head_cat"] = head_cat
    return nose_interp_df


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr]
def correct_nose_position(selected_df, ma, xdim, ydim, pix_cm, pf_cm):
    """
    Functionality:
    - Correct nose position.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `ma`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `xdim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `ydim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pix_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pf_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 3 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    error_count = 0
    unfixed_count = 0
    for idx, row in tqdm(selected_df.iterrows()):
        mass_idx = (row["mass_by"], row["mass_bx"])
        nose_idx = (row["by"], row["bx"])
        if (nose_idx in ma.ce.keys()) and (mass_idx in ma.ce.keys()):
            mass_nc = ma.ce[mass_idx]
            nose_nc = ma.ce[nose_idx]
            dist = ma.di[mass_nc, nose_nc]
            if dist > 2:
                x = row["x"]
                y = row["y"]
                mass_x = row["mass_x"]
                mass_y = row["mass_y"]
                selected_df.at[idx, "x"] = mass_x
                selected_df.at[idx, "y"] = mass_y
                error_count += 1
        elif (nose_idx not in ma.ce.keys()) and (mass_idx in ma.ce.keys()):
            x = row["x"]
            y = row["y"]
            mass_x = row["mass_x"]
            mass_y = row["mass_y"]
            selected_df.at[idx, "x"] = mass_x
            selected_df.at[idx, "y"] = mass_y
            error_count += 1
        elif (nose_idx in ma.ce.keys()) and (mass_idx not in ma.ce.keys()):
            continue
        else:
            unfixed_count += 1
            selected_df.drop(idx, inplace=True)
    y = 1 - np.array(selected_df.x)
    x = 1 - np.array(selected_df.y)
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    selected_df["bx"] = bx
    selected_df["by"] = by
    S = np.vstack((selected_df["bx"], selected_df["by"], selected_df["head_cat"]))
    linS = np.ravel_multi_index(S, (nBnx, nBny, 8))
    selected_df["linS"] = linS
    return selected_df, error_count, unfixed_count


# time in recording to time in maze


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_placeMap_with_head_dir_single_cell_m(
    selected_df, S_trans, start_frame, end_frame
):
    """
    Functionality:
    - Generate placeMap with head dir single cell m.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    cat_df = selected_df[
        (selected_df["maze_time"] > start_frame / 25)
        & (selected_df["maze_time"] < end_frame / 25)
    ].copy()
    placeMap = np.zeros((15, 15, 8))
    occMap = np.zeros((15, 15, 8))
    for i in range(8):
        temp_df = cat_df[cat_df["head_cat"] == i]
        y = 1 - np.array(temp_df.x)
        x = 1 - np.array(temp_df.y)
        xdim = [1, 0]  # Set arena x dimensions
        ydim = [1, 0]  # Set arena y dimensions
        pix_cm = 1 / 15
        pf_cm = 1
        linS, occMap_single, nBnx, nBny, bx, by = arena_binned(
            x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
        )
        s_idx = np.array(temp_df["frame"]).astype(int)
        S_selected_4x = S_trans[s_idx].T
        rateMap = sps.csr_matrix(
            (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
        ).todense()
        occMap[:, :, i] = occMap_single
        placeMap[:, :, i] = np.divide(rateMap, occMap_single)
        # Replace NaN values with zeros
    placeMap = np.nan_to_num(placeMap)
    # placeMap = placeMap/np.sum(placeMap)
    return occMap, placeMap


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def generate_placeMap_with_head_dir_single_cell_m_sim(
    selected_df, S_trans, start_frame, end_frame
):
    """
    Functionality:
    - Generate placeMap with head dir single cell m sim.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `S_trans`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_frame`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    cat_df = selected_df[
        (selected_df["maze_time"] > start_frame / 25)
        & (selected_df["maze_time"] < end_frame / 25)
    ].copy()
    placeMap = np.zeros((15, 15, 8))
    occMap = np.zeros((15, 15, 8))
    for i in range(8):
        temp_df = cat_df[cat_df["head_cat"] == i]
        y = 1 - np.array(temp_df.x)
        x = 1 - np.array(temp_df.y)
        xdim = [1, 0]  # Set arena x dimensions
        ydim = [1, 0]  # Set arena y dimensions
        pix_cm = 1 / 15
        pf_cm = 1
        linS, occMap_single, nBnx, nBny, bx, by = arena_binned(
            x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
        )
        # s_idx = np.array(temp_df['frame']).astype(int)
        s_idx = temp_df.index
        S_selected_4x = S_trans[s_idx].T
        rateMap = sps.csr_matrix(
            (S_selected_4x, (bx, by)), shape=(nBnx, nBny), dtype=float
        ).todense()
        occMap[:, :, i] = occMap_single
        placeMap[:, :, i] = np.divide(rateMap, occMap_single)
    placeMap = np.nan_to_num(placeMap)
    # placeMap = placeMap/np.sum(placeMap)
    return occMap, placeMap


# calculate correlation coefficient with 3D matrix


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr, correlation]
def calculate_correlation_coefficient_3D_matrix(
    place1, place2, occ1, occ2, threshold=1
):
    """
    Calculates the correlation coefficient between two 3D matrices.
    Args:
        A: A 3D matrix.
        B: A 3D matrix.
    Returns:
        The correlation coefficient between the two matrices.
    """
    if place1.shape != place2.shape:
        raise ValueError("The two matrices must have the same shape.")
    # num_slices, num_rows, num_cols = place1.shape
    # A = np.reshape(place1, (num_slices, -1))
    # B = np.reshape(place2, (num_slices, -1))
    # occ1 = np.reshape(occ1, (num_slices, -1))
    # occ2 = np.reshape(occ2, (num_slices, -1))
    A = np.ravel(place1)
    B = np.ravel(place2)
    occ1 = np.ravel(occ1)
    occ2 = np.ravel(occ2)
    ind = np.where((occ1 >= threshold) & (occ2 >= threshold))
    A = A[ind]
    B = B[ind]
    return np.corrcoef(A, B)[0, 1]


# calculation correlation coefficient when a location was visited prior and post


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr, correlation]
def process_selected_df(
    selected_df, tr, speed_max, speed_min, xdim, ydim, pix_cm, pf_cm
):
    """
    Functionality:
    - Process selected df.
    Inputs:
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `tr`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `speed_max`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `speed_min`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `xdim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `ydim`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pix_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pf_cm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `selected_df` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    selected_df = selected_df[
        (selected_df["speed"] < speed_max) & (selected_df["speed"] > speed_min)
    ].copy()
    selected_df = selected_df[selected_df["frame"] <= tr.fr[-1][1]]
    y = 1 - np.array(selected_df.mass_x)
    x = 1 - np.array(selected_df.mass_y)
    linS, occMap, nBnx, nBny, mass_bx, mass_by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    selected_df["mass_bx"] = mass_bx
    selected_df["mass_by"] = mass_by
    y = 1 - np.array(selected_df.x)
    x = 1 - np.array(selected_df.y)
    linS, occMap, nBnx, nBny, bx, by = arena_binned(
        x=x, y=y, xdim=xdim, ydim=ydim, pix_cm=pix_cm, pf_cm=pf_cm
    )
    selected_df["bx"] = bx
    selected_df["by"] = by
    S = np.vstack((selected_df["bx"], selected_df["by"], selected_df["head_cat"]))
    linS = np.ravel_multi_index(S, (nBnx, nBny, 8))
    selected_df["linS"] = linS
    maze_times = []
    for abs_frame in selected_df.frame:
        maze_time = TimeInMaze(abs_frame, tr)
        maze_times.append(maze_time)
    selected_df["maze_time"] = maze_times
    return selected_df


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: read]
def read_csv(file_name):
    """
    Functionality:
    - Read csv.
    Inputs:
    - `file_name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `data` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def extract_info(file_name):
    # ID: digits + optional lowercase letter (e.g., '1616b')
    # Optional: -<digits> as experiment_type
    # Optional: one or more R/L flags
    # Optional: 'day' + <digits> as experiment_day
    """
    Functionality:
    - Extract info.
    Inputs:
    - `file_name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 3 element(s); shapes depend on branch/data.
    - None in some branches.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    pattern = r"^([0-9]+[a-z]?)(?:-?(\d*))?(?:[RL]+)?(?:day)?(\d+)?$"
    match = re.match(pattern, file_name)
    if match:
        animal_id = match.group(1)
        experiment_type = int(match.group(2)) if match.group(2) else 0
        experiment_day = int(match.group(3)) if match.group(3) else 1
        return animal_id, experiment_type, experiment_day
    else:
        return None


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: read]
def read_csv_to_dict(csv_file_path):
    """
    Read a CSV file and use the first column as the key and the second column as the value to save the result in a dictionary called 'video'.
    Parameters:
    csv_file_path (str): Path to the CSV file.
    Returns:
    dict: A dictionary with the first column as keys and the second column as values.
    """
    video = {}
    with open(csv_file_path, mode="r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if len(row) >= 2:  # Ensure there are at least two columns
                key = row[1]
                value = row[0]
                video[key] = value
    return video


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def load_sparse_matrix(group):
    """Load a sparse matrix from a group and convert to a dense numpy array."""
    data = group["data"][:]
    indices = group["ir"][:]
    indptr = group["jc"][:]
    # Reconstruct the shape
    # num_rows = int(indices.max() + 1)  # Assuming 0-based indexing
    # num_cols = int(len(indptr) - 1)
    # shape = (num_rows, num_cols)
    sparse_matrix = sp.csc_matrix((data, indices, indptr))
    return sparse_matrix.toarray()  # Convert to a dense numpy array


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: read, file]
def read_mat_file(neural_dir, exp_date, datasets=["S", "Cn", "PNR"], verbose=True):
    """Read data from a MATLAB .mat file, handling sparse matrices and datasets."""
    file_path = os.path.join(neural_dir, "final_day" + str(exp_date) + ".mat")
    # Step 1: Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    data_dict = {}
    # Step 2: Inspect the file structure and read the data
    with h5py.File(file_path, "r") as neural_data:
        # print("Keys: %s" % list(neural_data.keys()))  # List all keys in the file
        # Define the datasets you are interested in
        # datasets = ['A', 'C', 'S', 'C_raw', 'Cn', 'PNR']
        # Access the datasets or groups
        for dataset in datasets:
            if dataset in neural_data:
                obj = neural_data[dataset]
                if isinstance(obj, h5py.Group):
                    # Assume it's a sparse matrix stored as a group
                    data_dict[dataset] = load_sparse_matrix(obj)
                    if verbose:
                        print(
                            f"{dataset} sparse matrix read and converted to dense array successfully."
                        )
                elif isinstance(obj, h5py.Dataset):
                    data_dict[dataset] = obj[:]
                    if verbose:
                        print(f"{dataset} dataset read successfully.")
                else:
                    print(f"Unknown type for {dataset}.")
            else:
                print(f"Dataset '{dataset}' not found.")
        # Ensure Cn and A are in the data_dict for reshaping A
        # if 'Cn' in data_dict and 'A' in data_dict:
        #     Cn = data_dict['Cn']
        #     A = data_dict['A']
        #     # Reshape A to match the dimensions of Cn
        #     height, width = Cn.shape
        #     depth = A.shape[1]
        #     data_dict['A_reshaped'] = A.reshape((depth, height, width))
        #     print(f"A reshaped to match the dimensions of Cn with shape {data_dict['A_reshaped'].shape}.")
        # Example of accessing a group and its datasets
        group_name = "your_group_name"  # Replace with actual group name if needed
        if group_name in neural_data and isinstance(
            neural_data[group_name], h5py.Group
        ):
            group = neural_data[group_name]
            print(f"Group '{group_name}' found.")
            for name, obj in group.items():
                if isinstance(obj, h5py.Group):
                    data_dict[name] = load_sparse_matrix(obj)
                    print(
                        f"Sparse matrix '{name}' in group '{group_name}' read and converted to dense array successfully."
                    )
                elif isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[:]
                    print(
                        f"Dataset '{name}' in group '{group_name}' read successfully."
                    )
                else:
                    print(f"Unknown type for {name} in group {group_name}.")
    return data_dict


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr]
def display_corr(Cn, resized=True):
    """
    Display the image of Cn using matplotlib.
    Parameters:
    Cn (numpy.ndarray): 2D array representing the image.
    """
    if resized:
        Cn = resize_image(Cn, (576, 720))
    plt.figure()
    plt.imshow(Cn, vmin=np.min(Cn), vmax=np.max(Cn), cmap="gray")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
    plt.axis("off")
    plt.show()


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def resize_image(image, size):
    """
    Resize an image to the given size.
    Parameters:
    image (numpy.ndarray): The image to resize.
    size (tuple): The target size (height, width).
    Returns:
    numpy.ndarray: The resized image.
    """
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: process]
def process_behav_data(working_dir, video, nickname, tr, ma):
    """
    Process mouse data and save the results to CSV files.
    Parameters:
    working_dir (str): The working directory.
    video (dict): Dictionary containing video information.
    nickname (str): Nickname of the mouse.
    tr: Parameter for processing (provide appropriate type).
    ma: Parameter for correcting nose position (provide appropriate type).
    """
    # Query and get data
    Data = QueryData(working_dir=working_dir)
    data = Data.get_data(video[nickname], save_dir=None)
    # Interpolate uncertain positions
    nose_interp_df = process_mouse_positions(data, video, nickname, tr)
    # Speed thresholds
    speed_max = 60  # cm/s
    speed_min = 2  # cm/s
    # Parameters for discretization
    xdim = [1, 0]  # Set arena x dimensions
    ydim = [1, 0]  # Set arena y dimensions
    pix_cm = 1 / 15
    pf_cm = 1
    # Select frames with speed threshold and discretize positions and head directions
    selected_df = process_selected_df(
        nose_interp_df, tr, speed_max, speed_min, xdim, ydim, pix_cm, pf_cm
    )
    selected_df, _, _ = correct_nose_position(
        selected_df, ma, xdim, ydim, pix_cm, pf_cm
    )
    # Save the selected data
    save_dir = os.path.join(working_dir, "selected_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    selected_df.to_csv(
        os.path.join(save_dir, nickname + "_selected_data.csv"), index=False
    )
    nose_interp_df.to_csv(
        os.path.join(save_dir, nickname + "_nose_interp_data.csv"), index=False
    )
    return selected_df, nose_interp_df


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def find_subsequences(states_frames, start_state, end_state, threshold):
    """
    Functionality:
    - Find subsequences.
    Inputs:
    - `states_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `threshold`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `subsequences` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    subsequences = []
    n = states_frames.shape[0]
    i = 0
    while i < n:
        if states_frames[i, 0] == start_state:
            subseq_start = i
            found_end_state = False
            for j in range(i + 1, min(i + threshold, n)):
                if states_frames[j, 0] == end_state and not (
                    start_state in states_frames[i + 1 : j, 0]
                    or end_state in states_frames[i + 1 : j, 0]
                ):
                    subsequences.append(states_frames[subseq_start : j + 1])
                    i = j
                    found_end_state = True
                    break
            if not found_end_state:
                i += 1  # Move to next state if no end state found in the threshold
        else:
            i += 1
    return subsequences


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def prepend_home_exit_state(bout, start_frame):
    """
    Prepend the 'exit from home' state to the beginning of each bout.
    Parameters:
    - bout: np.array representing a bout with shape (n, 2) for states and start frames.
    - start_frame: int, the start frame of the bout.
    Returns:
    - adjusted_bout: np.array with the 'exit from home' state prepended.
    """
    exit_home_state = np.array(
        [[127, start_frame]]
    )  # Assuming the exit occurs immediately before the bout starts
    adjusted_bout = np.vstack((exit_home_state, bout))
    return adjusted_bout


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def find_subsequences_in_bouts(bouts, tf_fr, start_state, end_state, threshold):
    """
    Functionality:
    - Find subsequences in bouts.
    Inputs:
    - `bouts`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `tf_fr`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `start_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `end_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `threshold`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `all_subsequences` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    all_subsequences = []
    for bout_index, bout in enumerate(bouts):
        if (
            bout.shape[0] < 2
        ):  # Skip bouts that are too short to form a valid subsequence
            continue
        bout_start_frame = tf_fr[bout_index, 0]
        adjusted_bout = prepend_home_exit_state(bout, 0)
        subsequences = find_subsequences(
            adjusted_bout, start_state, end_state, threshold
        )
        if not subsequences:  # Continue to the next bout if no subsequences are found
            continue
        for subsequence in subsequences:
            adjusted_subsequence = subsequence.copy()
            adjusted_subsequence[
                :, 1
            ] += bout_start_frame  # Adjust time frames, excluding the first frame which is the exit from home
            all_subsequences.append(adjusted_subsequence)
    return all_subsequences


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: path]
def find_path_node_based(start_index, end_index):
    """Finds the path between two nodes in a zero-based indexed binary tree, with a special node 127 connected to the root."""
    def path_to_root(index):
        """Helper function to find the path from a node to the root, adjusting for special node 127."""
        path = []
        if index == 127:
            return [127, 0]
        while index >= 0:
            path.append(index)
            if index == 0:
                break
            index = (index - 1) // 2
        return path
    # Get paths to root for both start and end indices
    start_path = path_to_root(start_index)
    end_path = path_to_root(end_index)
    # Find lowest common ancestor (LCA) by finding the common path elements
    lca = -1
    min_length = min(len(start_path), len(end_path))
    for i in range(min_length):
        if start_path[-1 - i] == end_path[-1 - i]:
            lca = start_path[-1 - i]
        else:
            break
    # Construct the full path from start to end via LCA
    # Path from start to LCA
    start_to_lca = start_path[: start_path.index(lca) + 1]
    # Path from LCA to end (need to reverse and include LCA correctly)
    lca_to_end = end_path[: end_path.index(lca) + 1]
    lca_to_end.reverse()
    lca_to_end = lca_to_end[1:]  # Exclude the LCA to avoid duplication in the middle
    # Combine paths to form the full path from start to end
    full_path = start_to_lca + lca_to_end
    return full_path


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def load_hash_map(file_path):
    """
    Functionality:
    - Load hash map.
    Inputs:
    - `file_path`: Filesystem path.
      Typical shape: Scalar string/path.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: position]
def get_node_positions(sequence, hash_map):
    """
    Functionality:
    - Get node positions.
    Inputs:
    - `sequence`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `hash_map`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `node_positions` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    node_positions = [hash_map[node] for node in sequence]
    return node_positions


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def get_edges(sequence):
    """
    Functionality:
    - Get edges.
    Inputs:
    - `sequence`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `edges` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    edges = [(i, i + 1) for i in range(len(sequence) - 1)]
    return edges


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: process]
def process_firing_rate_map(
    position_df, S_tr, nickname, bin_size=0.3, sigma=1, normalize=True
):
    """
    Process firing rate map and visualize the result.
    Parameters:
    position_df (pandas.DataFrame): DataFrame containing the position data.
    S_tr (numpy.ndarray): 2D array containing the firing data.
    bin_size (float): Size of the spatial bins. Default is 0.3.
    sigma (float): Standard deviation for Gaussian smoothing. Default is 1.
    Returns:
    numpy.ndarray: The sorted firing rate maps.
    numpy.ndarray: The sorted firing data.
    """
    # Create spatial bins
    position_min, position_max = (
        position_df["linear_position"].min(),
        position_df["linear_position"].max(),
    )
    bins = np.arange(position_min, position_max + 1 * bin_size, bin_size)
    # Digitize the positions to find bin indices
    position_indices = np.digitize(position_df["linear_position"], bins) - 1
    # Initialize firing rate maps
    firing_rate_maps = np.zeros((len(bins), S_tr.shape[0]))
    # Compute occupancy and firing rates
    for idx, bin_edge in enumerate(bins[:-1]):
        in_bin = position_indices == idx
        if np.any(in_bin) and S_tr[:, in_bin].size > 0:
            firing_rate_maps[idx, :] = np.nanmean(S_tr[:, in_bin], axis=1)
    # Smooth the firing rate maps
    firing_rate_maps = gaussian_filter1d(firing_rate_maps, sigma=sigma, axis=0)
    # Order firing rate maps based on the position of peak firing rate
    peak_firing_idx = np.argmax(firing_rate_maps, axis=0)
    sorted_idx = np.argsort(peak_firing_idx)
    firing_rate_maps_sorted = firing_rate_maps[:, sorted_idx]
    S_tr_sorted = S_tr[sorted_idx]
    # Get rid of the neurons that do not fire
    valid_neurons = np.sum(firing_rate_maps_sorted, axis=0) > 0
    firing_rate_maps_plot = firing_rate_maps_sorted[:, valid_neurons]
    # Normalize each row to have a max of 1
    if normalize:
        firing_rate_maps_plot = firing_rate_maps_plot / np.max(
            firing_rate_maps_plot, axis=0
        )
    # Plot the sorted firing rate maps
    plt.figure(figsize=(5, 10))
    plt.xticks(np.arange(0, len(bins), 10), bins[::10].round(2))
    plt.imshow(
        firing_rate_maps_plot.T, aspect="auto", cmap="hot", interpolation="nearest"
    )
    plt.xlabel("Position Bins")
    plt.ylabel("Neurons")
    plt.title(f"Firing Rate Maps for {nickname}")
    plt.show()
    return position_indices, bins, firing_rate_maps_sorted, S_tr_sorted, sorted_idx


class PoissonNaiveBayes:
    def __init__(self):
        self.lambdas = None
        self.labels = None

    def fit(self, X_tr, y_tr):
        """
        Train the Poisson Naive Bayes classifier.
        Args:
        X_tr (np.array): Training data, shape (n_features, n_samples)
        y_tr (np.array): Training labels, shape (n_samples,)
        """
        X_tr = X_tr.T
        if X_tr.shape[1] != y_tr.shape[0]:
            raise ValueError(
                "Number of columns in X_tr must match the number of elements in y_tr"
            )
        self.labels = np.unique(y_tr)
        self.lambdas = np.zeros((X_tr.shape[0], len(self.labels)))
        for i, label in enumerate(self.labels):
            mask = y_tr == label
            self.lambdas[:, i] = np.mean(X_tr[:, mask], axis=1)
            # Replace zero lambdas with a small value
            zero_lambda_replacement = 1 / (np.sum(mask) + 1)
            self.lambdas[self.lambdas[:, i] == 0, i] = zero_lambda_replacement

    def predict(self, X_te):
        """
        Test the Poisson Naive Bayes classifier.
        Args:
        X_te (np.array): Test data, shape (n_features, n_samples)
        Returns:
        predicted_labels (np.array): Predicted labels for test data
        decision_values (np.array): Log-likelihoods for each class
        """
        X_te = X_te.T
        n_features, n_samples = X_te.shape
        n_classes = len(self.labels)
        # Compute log-likelihoods
        curr_lambdas = np.repeat(self.lambdas[:, :, np.newaxis], n_samples, axis=2)
        X_te_expanded = np.repeat(X_te[:, np.newaxis, :], n_classes, axis=1)
        log_likelihoods = np.sum(
            -curr_lambdas
            + X_te_expanded * np.log(curr_lambdas)
            - gammaln(X_te_expanded + 1),
            axis=0,
        )
        # Find the class with maximum log-likelihood
        predicted_indices = np.argmax(log_likelihoods, axis=0)
        predicted_labels = self.labels[predicted_indices]
        decision_values = log_likelihoods.T
        return predicted_labels, decision_values


class GaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr]
def correct_tracking_errors(node_sequence):
    """
    Functionality:
    - Correct tracking errors.
    Inputs:
    - `node_sequence`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    changes_made = True  # Flag to track whether any changes were made in the loop
    while changes_made:
        corrected_sequence = []
        i = 0
        changes_made = False  # Reset flag for this pass
        while i < len(node_sequence):
            node = node_sequence[i, 0]
            if (
                i + 2 < len(node_sequence)
                and node_sequence[i, 0] == node_sequence[i + 2, 0]
            ):
                # Check if the intermediate node is at a significantly different level
                intermediate_node = node_sequence[i + 1, 0]
                if (node >= 63 and node <= 126) and (
                    intermediate_node >= 31 and intermediate_node <= 62
                ):
                    corrected_sequence.append(node_sequence[i])
                    i += 2  # Skip the next node, which is an erroneous detection
                    changes_made = True  # Indicate that a change was made
                else:
                    corrected_sequence.append(node_sequence[i])
            else:
                corrected_sequence.append(node_sequence[i])
            i += 1
        node_sequence = np.array(
            corrected_sequence
        )  # Update node_sequence with the corrected sequence
    return np.array(corrected_sequence)


# turn frames into time in h,m,s


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def frames_to_time(frames, fps=25):
    """
    Functionality:
    - Frames to time.
    Inputs:
    - `frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `fps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 3 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    s = frames / fps
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def time_to_frames(h, m, s, fps=25):
    """
    Functionality:
    - Time to frames.
    Inputs:
    - `h`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `m`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `s`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `fps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    return (h * 3600 + m * 60 + s) * fps


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: position]
def discretize_positions(positions, n_bins_x=15, n_bins_y=15):
    """
    Clip positions to [0,1] range and discretize into bin indices.
    Parameters
    ----------
    positions : array-like
        - 2D: shape (T, 2), interpreted as (x, y) in [0,1] (may contain <0 or >1)
        - 1D: shape (T,) or (T, 1), interpreted as x in [0,1] with a single
          degenerate y-bin (all samples in y-bin 0).
    n_bins_x : int
        Number of bins along x.
    n_bins_y : int
        Number of bins along y.
    Returns
    -------
    labels : (T,) int array
        Integer bin indices in [0 .. n_bins_x * n_bins_y - 1].
        In the 1D case (positions 1D or (T,1)), only the first "row"
        (y-bin 0) is used, so labels are effectively 0..n_bins_x-1.
    x_edges : (n_bins_x+1,) float array
        Bin edges from 0 to 1 along x.
    y_edges : (n_bins_y+1,) float array
        Bin edges from 0 to 1 along y. In the 1D case this is still returned
        (so the API is unchanged), but all points fall into y-bin 0.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        # Pure 1D: positions is (T,)
        x = np.clip(positions, 0.0, 1.0)
        # Degenerate y dimension: all zeros â†’ all samples in y-bin 0
        y = np.zeros_like(x)
    elif positions.ndim == 2:
        # Allow (T,1) or (T,>=2)
        clipped = np.clip(positions, 0.0, 1.0)
        x = clipped[:, 0]
        if clipped.shape[1] > 1:
            y = clipped[:, 1]
        else:
            # (T,1) â†’ treat as 1D: all y=0
            y = np.zeros_like(x)
    else:
        raise ValueError("positions must be a 1D array (T,) or 2D array (T,1)/(T,2).")
    # Define uniform edges between 0 and 1
    x_edges = np.linspace(0.0, 1.0, n_bins_x + 1)
    y_edges = np.linspace(0.0, 1.0, n_bins_y + 1)
    # Digitize into 0-based bin indices
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins_x - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins_y - 1)
    # Linear index
    labels = ix + iy * n_bins_x
    return labels, x_edges, y_edges


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_tuning_and_prior(
    spike_counts,  # (T, N) event counts per time bin & neuron
    labels,  # (T,) spatial bin index per time bin [0..n_bins-1]
    head_labels=None,  # (T,) head-direction index [0..n_head-1]
    n_bins=15 * 15,  # total number of spatial bins
    dt=0.04,  # duration of each time bin (s)
    filter_ratemap=False,
    sigma=1.0,  # spatial smoothing Ïƒ (in bins)
    min_time=5.0,  # minimum occupancy per bin (s)
    eps=1e-12,  # small constant to avoid zeros
    decode_head="none",  # 'none','independent','joint','joint_pos','joint_lvl'
    labels_lvl=None,  # (T,) level labels [0..n_lvls-1] if using 'joint_lvl'
    n_lvls=None,  # int, number of levels if using 'joint_lvl'
    spatial_shape=None,  # NEW: None | (n_bins,) for 1D | (nx, ny) for 2D
):
    """
    Compute Poisson tuning curves and priors for space and optional head-direction (and levels).
    spatial_shape:
      - None  : auto-detect (old behavior: square if n_bins is a perfect square)
      - (K,)  : 1D track with K bins
      - (nx, ny): 2D grid with nx * ny == n_bins
    Returns
    -------
    - if decode_head in {'none'}:
        (lambdas_spatial, prior_spatial)
    - if decode_head in {'independent'}:
        (lambdas_spatial, prior_spatial, lambdas_hd, prior_hd)
    - if decode_head in {'joint','joint_pos'}:
        (lambdas_joint_pos, prior_joint_pos)  # shapes: (n_bins, n_head, N), (n_bins, n_head)
    - if decode_head == 'joint_lvl':
        (lambdas_spatial, prior_spatial, lambdas_joint_lvl, prior_joint_lvl)
        # lambdas_joint_lvl/prior_joint_lvl shapes: (n_lvls, n_head, N), (n_lvls, n_head)
    """
    # Normalize mode names
    if decode_head == "joint":
        decode_head = "joint_pos"
    T, N = spike_counts.shape
    # --- Helper to interpret spatial_shape ---
    def _get_spatial_layout(n_bins, spatial_shape):
        """
        Returns (mode, shape) where:
          mode='1d', shape=(K,)
          mode='2d', shape=(nx, ny)
        """
        if spatial_shape is not None:
            spatial_shape = tuple(int(s) for s in spatial_shape)
            prod = 1
            for s in spatial_shape:
                prod *= s
            if prod != n_bins:
                raise ValueError(
                    f"spatial_shape {spatial_shape} has product {prod}, "
                    f"but n_bins={n_bins}."
                )
            if len(spatial_shape) == 1:
                return "1d", spatial_shape
            elif len(spatial_shape) == 2:
                return "2d", spatial_shape
            else:
                raise ValueError(
                    "spatial_shape must be 1D (K,) or 2D (nx, ny), "
                    f"got {spatial_shape}."
                )
        # Auto mode: preserve old behavior
        side = int(np.sqrt(n_bins))
        if side * side == n_bins:
            return "2d", (side, side)
        else:
            return "1d", (n_bins,)
    spatial_mode, spatial_shape = _get_spatial_layout(n_bins, spatial_shape)
    # spatial_shape is now a tuple; for 1d, e.g. (64,); for 2d, e.g. (8,8)
    # ---- Spatial occupancy & counts ----
    occupancy = np.zeros(n_bins, dtype=float)
    count_map = np.zeros((n_bins, N), dtype=float)
    for b in range(n_bins):
        mb = labels == b
        if mb.any():
            occupancy[b] = mb.sum() * dt
            count_map[b] = spike_counts[mb].sum(axis=0)
    # Spatial rates
    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas_spatial = count_map / np.maximum(occupancy[:, None], eps)
    # Exclude low-occupancy bins
    low_occ = occupancy < min_time
    if np.any(low_occ):
        lambdas_spatial[low_occ] = 0.0
        occupancy[low_occ] = 0.0
    # Optional spatial smoothing
    if filter_ratemap:
        sm = np.zeros_like(lambdas_spatial)
        if spatial_mode == "2d":
            nx, ny = spatial_shape
            for i in range(N):
                grid = lambdas_spatial[:, i].reshape(nx, ny)
                sm[:, i] = gaussian_filter(grid, sigma=sigma, mode="constant").ravel()
        else:  # '1d'
            for i in range(N):
                sm[:, i] = gaussian_filter(
                    lambdas_spatial[:, i], sigma=sigma, mode="constant"
                )
        lambdas_spatial = sm
    lambdas_spatial = lambdas_spatial + eps
    # Spatial prior (proportional to time)
    prior_spatial = np.zeros(n_bins, dtype=float)
    if occupancy.sum() > 0:
        prior_spatial = occupancy / occupancy.sum()
    # No head decoding requested
    if decode_head == "none":
        return lambdas_spatial, prior_spatial
    # From here on we need head labels
    if head_labels is None:
        raise ValueError("head_labels is required when decode_head!='none'.")
    n_head = int(np.max(head_labels)) + 1
    # ---- Independent head tuning (no spatial conditioning) ----
    if decode_head == "independent":
        occ_hd = np.zeros(n_head, dtype=float)
        cnt_hd = np.zeros((n_head, N), dtype=float)
        for h in range(n_head):
            mh = head_labels == h
            if mh.any():
                occ_hd[h] = mh.sum() * dt
                cnt_hd[h] = spike_counts[mh].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdas_hd = cnt_hd / np.maximum(occ_hd[:, None], eps)
        low_hd = occ_hd < min_time
        if np.any(low_hd):
            lambdas_hd[low_hd] = 0.0
            occ_hd[low_hd] = 0.0
        lambdas_hd = lambdas_hd + eps
        prior_hd = np.zeros(n_head, dtype=float)
        if occ_hd.sum() > 0:
            prior_hd = occ_hd / occ_hd.sum()
        return lambdas_spatial, prior_spatial, lambdas_hd, prior_hd
    # ---- Joint head Ã— position tuning ----
    if decode_head == "joint_pos":
        occ_j = np.zeros((n_bins, n_head), dtype=float)
        cnt_j = np.zeros((n_bins, n_head, N), dtype=float)
        for b in range(n_bins):
            mb = labels == b
            if not mb.any():
                continue
            hb = head_labels[mb]
            Xb = spike_counts[mb]
            for h in range(n_head):
                mh = hb == h
                if mh.any():
                    occ_j[b, h] = mh.sum() * dt
                    cnt_j[b, h] = Xb[mh].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdas_j = cnt_j / np.maximum(occ_j[..., None], eps)
        # Be lenient for joint occupancy (finer bins): require smaller min_time
        low_j = occ_j < max(min_time / 8.0, dt)
        if np.any(low_j):
            lambdas_j[low_j] = 0.0
            occ_j[low_j] = 0.0
        # Optional smoothing over space dimension per head & neuron
        if filter_ratemap:
            for h in range(n_head):
                for i in range(N):
                    vec = lambdas_j[:, h, i]
                    if spatial_mode == "2d":
                        nx, ny = spatial_shape
                        grid = vec.reshape(nx, ny)
                        lambdas_j[:, h, i] = gaussian_filter(
                            grid, sigma=sigma, mode="constant"
                        ).ravel()
                    else:  # '1d'
                        lambdas_j[:, h, i] = gaussian_filter(
                            vec, sigma=sigma, mode="constant"
                        )
        lambdas_j = lambdas_j + eps
        prior_j = np.zeros_like(occ_j)
        tot = occ_j.sum()
        if tot > 0:
            prior_j = occ_j / tot
        return lambdas_j, prior_j
    # ---- Joint head Ã— level tuning ----
    if decode_head == "joint_lvl":
        if labels_lvl is None:
            raise ValueError("labels_lvl is required for decode_head='joint_lvl'.")
        if n_lvls is None:
            n_lvls = int(np.max(labels_lvl)) + 1
        occ_lh = np.zeros((n_lvls, n_head), dtype=float)
        cnt_lh = np.zeros((n_lvls, n_head, N), dtype=float)
        for l in range(n_lvls):
            ml = labels_lvl == l
            if not ml.any():
                continue
            hl = head_labels[ml]
            Xl = spike_counts[ml]
            for h in range(n_head):
                mh = hl == h
                if mh.any():
                    occ_lh[l, h] = mh.sum() * dt
                    cnt_lh[l, h] = Xl[mh].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lambdas_lh = cnt_lh / np.maximum(occ_lh[..., None], eps)
        # Similar leniency as joint_pos
        low_lh = occ_lh < max(min_time / 8.0, dt)
        if np.any(low_lh):
            lambdas_lh[low_lh] = 0.0
            occ_lh[low_lh] = 0.0
        lambdas_lh = lambdas_lh + eps
        prior_lh = np.zeros_like(occ_lh)
        tot = occ_lh.sum()
        if tot > 0:
            prior_lh = occ_lh / tot
        # We still return spatial maps for position decoding alongside joint lvlÃ—hd
        return lambdas_spatial, prior_spatial, lambdas_lh, prior_lh
    raise ValueError(f"Unknown decode_head mode: {decode_head}")


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_tuning_and_prior_peak(
    spike_counts,  # (T, N) array: event counts per frame
    labels,  # (T,)   array: spatial-bin indices [0..n_bins-1]
    head_labels=None,  # (T,)   array: head-direction indices [0..n_head-1]
    n_bins=15 * 15,  # total number of spatial bins
    dt=0.04,  # duration of each time bin (s)
    filter_ratemap=False,
    sigma=1.0,  # spatial smoothing Ïƒ (in bins)
    min_visits=10,  # minimum number of visits for inclusion
    eps=1e-12,  # small constant to avoid exact zeros
    decode_head="none",  # 'none','independent','joint','joint_pos','joint_lvl'
    labels_lvl=None,  # (T,) level labels if using 'joint_lvl'
    n_lvls=None,  # int, number of levels if using 'joint_lvl'
    spatial_shape=None,  # NEW: None | (n_bins,) | (nx, ny)
):
    """
    Peak-per-visit tuning curves & priors (fast with targeted run-length encoding).
    spatial_shape:
      - None        : auto-detect (square if n_bins is perfect square)
      - (K,)        : 1D track, K bins
      - (nx, ny)    : 2D grid, nx * ny == n_bins
    """
    # normalize mode name
    if decode_head == "joint":
        decode_head = "joint_pos"
    T, N = spike_counts.shape
    # Helper to interpret spatial_shape
    def _get_spatial_layout(n_bins, spatial_shape):
        if spatial_shape is not None:
            spatial_shape = tuple(int(s) for s in spatial_shape)
            prod = 1
            for s in spatial_shape:
                prod *= s
            if prod != n_bins:
                raise ValueError(
                    f"spatial_shape {spatial_shape} has product {prod}, "
                    f"but n_bins={n_bins}."
                )
            if len(spatial_shape) == 1:
                return "1d", spatial_shape
            elif len(spatial_shape) == 2:
                return "2d", spatial_shape
            else:
                raise ValueError(
                    "spatial_shape must be 1D (K,) or 2D (nx, ny), "
                    f"got {spatial_shape}."
                )
        side = int(np.sqrt(n_bins))
        if side * side == n_bins:
            return "2d", (side, side)
        else:
            return "1d", (n_bins,)
    spatial_mode, spatial_shape = _get_spatial_layout(n_bins, spatial_shape)
    # ---------------------------
    # Helper: build runs for a key
    # key_fn maps t -> tuple of state labels (e.g., (bin,), (hd,), (bin,hd), (lvl,hd))
    # ---------------------------
    def build_runs(key_fn):
        runs = []
        s = 0
        prev = key_fn(0)
        for t in range(1, T):
            cur = key_fn(t)
            if cur != prev:
                runs.append((prev, s, t))
                prev, s = cur, t
        runs.append((prev, s, T))
        return runs
    # ---------------------------
    # 1) Spatial peak-per-visit (contiguous runs of same spatial bin)
    # ---------------------------
    runs_pos = build_runs(lambda t: (int(labels[t]),))
    lam_sp = np.zeros((n_bins, N), dtype=float)
    visits_sp = np.zeros(n_bins, dtype=int)
    for (key, s, e) in runs_pos:
        b = key[0]
        seg = spike_counts[s:e]
        if seg.size == 0:
            continue
        lam_sp[b] += np.nanmax(seg, axis=0)
        visits_sp[b] += 1
    ok_sp = visits_sp >= min_visits
    # average peak per visit and convert to rate
    lam_sp[ok_sp] = lam_sp[ok_sp] / np.maximum(visits_sp[ok_sp, None], 1)
    lam_sp[~ok_sp] = 0.0
    lam_sp = lam_sp / dt
    # optional smoothing over space (per neuron)
    if filter_ratemap:
        sm = np.zeros_like(lam_sp)
        if spatial_mode == "2d":
            nx, ny = spatial_shape
            for i in range(N):
                sm[:, i] = gaussian_filter(
                    lam_sp[:, i].reshape(nx, ny), sigma, mode="reflect"
                ).ravel()
        else:  # '1d'
            for i in range(N):
                sm[:, i] = gaussian_filter(lam_sp[:, i], sigma, mode="reflect")
        lam_sp = sm
    lam_sp += eps
    prior_sp = np.zeros(n_bins, dtype=float)
    if np.any(ok_sp):
        # prior proportional to visit counts (excluding low-visit bins)
        v = visits_sp.astype(float).copy()
        v[~ok_sp] = 0.0
        s = v.sum()
        if s > 0:
            prior_sp = v / s
    if decode_head == "none":
        return lam_sp, prior_sp
    # Need head labels beyond this point
    if head_labels is None:
        raise ValueError("head_labels is required when decode_head!='none'.")
    n_head = int(np.max(head_labels)) + 1
    # ---------------------------
    # 2) Independent head-direction peak-per-visit
    #    (contiguous runs of same head state)
    # ---------------------------
    if decode_head in ("independent", "joint_pos", "joint_lvl"):
        runs_hd = build_runs(lambda t: (int(head_labels[t]),))
        lam_hd = np.zeros((n_head, N), dtype=float)
        visits_hd = np.zeros(n_head, dtype=int)
        for (key, s, e) in runs_hd:
            h = key[0]
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_hd[h] += np.nanmax(seg, axis=0)
            visits_hd[h] += 1
        # lenient threshold for head-only map
        th_hd = max(3, min_visits // 4)
        ok_hd = visits_hd >= th_hd
        lam_hd[ok_hd] = lam_hd[ok_hd] / np.maximum(visits_hd[ok_hd, None], 1)
        lam_hd[~ok_hd] = 0.0
        lam_hd = lam_hd / dt + eps
        prior_hd = np.zeros(n_head, dtype=float)
        v = visits_hd.astype(float).copy()
        v[~ok_hd] = 0.0
        s = v.sum()
        if s > 0:
            prior_hd = v / s
        if decode_head == "independent":
            return lam_sp, prior_sp, lam_hd, prior_hd
    # ---------------------------
    # 3) Joint position Ã— head (contiguous runs of same (bin, head))
    # ---------------------------
    if decode_head == "joint_pos":
        runs_pxh = build_runs(lambda t: (int(labels[t]), int(head_labels[t])))
        lam_pxh = np.zeros((n_bins, n_head, N), dtype=float)
        visits_pxh = np.zeros((n_bins, n_head), dtype=int)
        for (key, s, e) in runs_pxh:
            b, h = key
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_pxh[b, h] += np.nanmax(seg, axis=0)
            visits_pxh[b, h] += 1
        # more lenient for joint states
        th_j = max(2, min_visits // 5)
        ok_j = visits_pxh >= th_j
        lam_pxh[ok_j] = lam_pxh[ok_j] / np.maximum(visits_pxh[ok_j, None], 1)
        lam_pxh[~ok_j] = 0.0
        lam_pxh = lam_pxh / dt
        # optional smoothing over space for each (hd, neuron)
        if filter_ratemap:
            for h in range(n_head):
                for i in range(N):
                    vec = lam_pxh[:, h, i]
                    if spatial_mode == "2d":
                        nx, ny = spatial_shape
                        lam_pxh[:, h, i] = gaussian_filter(
                            vec.reshape(nx, ny), sigma, mode="reflect"
                        ).ravel()
                    else:  # '1d'
                        lam_pxh[:, h, i] = gaussian_filter(vec, sigma, mode="reflect")
        lam_pxh += eps
        prior_pxh = np.zeros_like(visits_pxh, dtype=float)
        vv = visits_pxh.astype(float).copy()
        vv[~ok_j] = 0.0
        s = vv.sum()
        if s > 0:
            prior_pxh = vv / s
        return lam_pxh, prior_pxh
    # ---------------------------
    # 4) Joint level Ã— head (contiguous runs of same (level, head))
    # ---------------------------
    if decode_head == "joint_lvl":
        if labels_lvl is None:
            raise ValueError("labels_lvl is required for decode_head='joint_lvl'.")
        if n_lvls is None:
            n_lvls = int(np.max(labels_lvl)) + 1
        runs_lxh = build_runs(lambda t: (int(labels_lvl[t]), int(head_labels[t])))
        lam_lxh = np.zeros((n_lvls, n_head, N), dtype=float)
        visits_lxh = np.zeros((n_lvls, n_head), dtype=int)
        for (key, s, e) in runs_lxh:
            l, h = key
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_lxh[l, h] += np.nanmax(seg, axis=0)
            visits_lxh[l, h] += 1
        th_lh = max(2, min_visits // 5)
        ok_lh = visits_lxh >= th_lh
        lam_lxh[ok_lh] = lam_lxh[ok_lh] / np.maximum(visits_lxh[ok_lh, None], 1)
        lam_lxh[~ok_lh] = 0.0
        lam_lxh = lam_lxh / dt + eps
        prior_lxh = np.zeros_like(visits_lxh, dtype=float)
        vv = visits_lxh.astype(float).copy()
        vv[~ok_lh] = 0.0
        s = vv.sum()
        if s > 0:
            prior_lxh = vv / s
        # We still return spatial maps (lam_sp, prior_sp) for position decoding
        return lam_sp, prior_sp, lam_lxh, prior_lxh
    # unknown mode
    raise ValueError(f"Unknown decode_head mode: {decode_head}")


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: build]
def build_count_matrix(
    spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids=None
):
    """
    Build N x T count matrix Q by summing counts in Â±time_window around each frame.
    spike_counts shape: (total_frames, total_neurons)
    frames: array of frame indices to decode
    """
    if neuron_ids is None:
        neuron_ids = np.arange(spike_counts.shape[1])
    N = len(neuron_ids)
    T = len(frames)
    Q = np.zeros((N, T))
    # replace all NaN values in raw_counts with 0
    raw_counts = np.nan_to_num(raw_counts, nan=0.0)
    for j, f in enumerate(frames):
        # print(f)
        rf = selected_df.iloc[f].frame
        rf = int(rf)  # Ensure rf is an integer frame index
        wind = slice(max(0, rf - time_window), min(rf + 1, raw_counts.shape[0]))
        Q[:, j] = raw_counts[wind, neuron_ids].sum(axis=0)
    return Q


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def generate_level_list(ma_ce, grid_size=15):
    """
    Generate 8 level masks over a grid of size grid_size x grid_size,
    using the mapping `ma_ce[(i, j)] -> linear index` to populate bitmask arrays.
    Parameters:
    -----------
    ma_ce : dict or mapping from (i, j) tuples to linear indices [0..len(ma_ce)-1]
    grid_size : int, size of the square grid (default 15)
    Returns:
    --------
    level_list : list of 8 numpy arrays, each of shape (len(ma_ce),),
                 containing 0/1 masks for each level pattern.
    """
    # Pre-allocate masks
    n_cells = len(ma_ce)
    levels = [np.zeros(n_cells, dtype=int) for _ in range(8)]
    # Level 1: column j=7, rows i=0..6
    for i in range(0, 7):
        idx = ma_ce[(i, 7)]
        levels[0][idx] = 1
    # Level 2: row i=7, columns j=4..10
    for j in range(4, 11):
        idx = ma_ce[(7, j)]
        levels[1][idx] = 1
    # Level 3: rows i=4..10, columns j in {3, 11}
    for i in range(4, 11):
        for j in [3, 11]:
            idx = ma_ce[(i, j)]
            levels[2][idx] = 1
    # Level 4: corners (3,3), (3,11), (11,3), (11,11) plus adjacent horizontally
    for i in [3, 11]:
        for j in [3, 11]:
            for dj in [0, -1, 1]:
                ii, jj = i, j + dj
                idx = ma_ce.get((ii, jj))
                if idx is not None:
                    levels[3][idx] = 1
    # Level 5: corners (3,1), (3,5), (3,9), (3,13) plus adjacent vertically
    for i in [3, 11]:
        for j in range(1, 14, 4):
            for di in [0, -1, 1]:
                ii, jj = i + di, j
                idx = ma_ce.get((ii, jj))
                if idx is not None:
                    levels[4][idx] = 1
    # Level 6: grid with step 4 starting at (1,1)
    for i in range(1, grid_size, 4):
        for j in range(1, grid_size, 4):
            idx = ma_ce[(i, j)]
            levels[5][idx] = 1
    # Level 7: rows step 4 from i=1, columns step 2 from j=0
    for i in range(1, grid_size, 4):
        for j in range(0, grid_size, 2):
            idx = ma_ce[(i, j)]
            levels[6][idx] = 1
    # Level 8: full checkerboard at stride 2
    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            idx = ma_ce[(i, j)]
            levels[7][idx] = 1
    return levels


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def assign_levels(
    selected_df,
    ma_ce,
    level_list,
    frame_col="frame",
    bx_col="mass_bx",
    by_col="mass_by",
):
    """
    Assigns a level index to each row in selected_df based on cell occupancy and predefined level masks.
    If a tracked (by, bx) pair is missing in ma_ce, searches backward in time for the closest valid cell.
    Parameters:
    -----------
    selected_df : pandas.DataFrame
        DataFrame containing at least the frame, mass_bx, and mass_by columns.
    ma_ce : dict
        Mapping from (by, bx) tuples to linear cell indices.
    level_list : list of 1D numpy arrays
        Each array is a binary mask (length == n_cells) indicating membership in that level.
    frame_col : str
        Name of the frame index column in selected_df.
    bx_col : str
        Name of the mass_bx column in selected_df.
    by_col : str
        Name of the mass_by column in selected_df.
    Returns:
    --------
    levels : list of int or None
        For each row in selected_df, the index of the level that contains its cell.
        If no valid cell is found or no level matches, returns None for that frame.
    """
    # Build a quick lookup for past positions
    frame_positions = {
        int(r[frame_col]): (int(r[bx_col]), int(r[by_col]))
        for _, r in selected_df.iterrows()
    }
    levels = []
    for _, row in selected_df.iterrows():
        frame = int(row[frame_col])
        bx = int(row[bx_col])
        by = int(row[by_col])
        # Search backward until a valid cell is found or run out of frames
        while (by, bx) not in ma_ce and frame >= 0:
            frame -= 1
            if frame in frame_positions:
                bx, by = frame_positions[frame]
            else:
                continue
        # Determine cell index
        cid = ma_ce.get((by, bx))
        if cid is None:
            levels.append(None)
            continue
        # Find which level mask contains this cell
        lvl_idx = None
        for idx, mask in enumerate(level_list):
            if mask[cid] == 1:
                lvl_idx = idx
                break
        levels.append(lvl_idx)
    return levels


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot, hist]
def plot_similarity_hist(similarity, bins=30, title="Tuning similarity", savepath=None):
    """
    Functionality:
    - Plot similarity hist.
    Inputs:
    - `similarity`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `bins`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `savepath`: Filesystem path.
      Typical shape: Scalar string/path.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    s = similarity[np.isfinite(similarity)]
    ax.hist(s, bins=bins, alpha=0.8)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig, ax


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot, scatter]
def plot_pair_scatter(mapA, mapB, title="Map A vs Map B", savepath=None):
    """
    mapA, mapB: 1D vectors of same length (after masking & z-scoring if you want).
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(mapA, mapB, s=6, alpha=0.5)
    ax.set_xlabel("Session A")
    ax.set_ylabel("Session B")
    ax.set_title(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    return fig, ax


# ===== Correlation & PV utilities (moved from 260209_correlation.ipynb) =====


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _pick_matched_neurons(
    cimap, n_neurons=None, random_state=0, n_train=None, n_test=None
):
    """
    Returns train_ids, test_ids (0-based, same length).
    - If cimap is None: same-index matching up to min(n_train, n_test).
    - If cimap provided: uses pairs; accepts 1- or 0-based.
    """
    rng = np.random.RandomState(random_state)
    if cimap is None:
        if n_train is None or n_test is None:
            raise ValueError("When cimap=None, provide n_train and n_test.")
        K = min(int(n_train), int(n_test))
        train_all = np.arange(K, dtype=int)
        test_all = np.arange(K, dtype=int)
    else:
        pairs = np.asarray(cimap, dtype=int)
        if pairs.min() >= 0:
            pairs = pairs - 1
        valid = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
        test_all, train_all = pairs[valid, 0], pairs[valid, 1]
    # Optional subsample
    if n_neurons is not None and len(train_all) > n_neurons:
        sel = rng.choice(len(train_all), int(n_neurons), replace=False)
        train_all, test_all = train_all[sel], test_all[sel]
    return train_all, test_all


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def _fit_maps_for_session(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    neuron_ids,
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    eps=1e-12,
):
    """
    Returns (maps, valid_mask, meta):
      maps: (K, N_sel) flattened tuning matrix per neuron
      valid_mask: (K,) mask of states kept (non-zero prior)
      meta: dict with shapes/edges (e.g., n_bins/n_lvls/n_hd, x_edges/y_edges)
    Supports both 2D and 1D positions:
      - positions shape (T, 2): 2D arena, use (n_bins_x, n_bins_y).
      - positions shape (T,) or (T,1): 1D track, REQUIRE n_bins_y == 1 so that
        n_bins = n_bins_x and spatial_shape = (n_bins,).
    """
    positions_arr = np.asarray(positions)
    is_1d_pos = (positions_arr.ndim == 1) or (
        positions_arr.ndim == 2 and positions_arr.shape[1] == 1
    )
    if is_1d_pos and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 so that "
            "n_bins = n_bins_x. (Current n_bins_y != 1.)"
        )
    # Position labels for spatial/joint_pos (still cheap to compute)
    # discretize_positions already handles 1D vs 2D input.
    labels_pos, x_edges, y_edges = discretize_positions(
        positions_arr, n_bins_x, n_bins_y
    )
    n_bins = n_bins_x * n_bins_y  # for 1D, n_bins_y == 1 â†’ n_bins = n_bins_x
    # Spatial shape for smoothing:
    #  - 1D: (n_bins,)
    #  - 2D: (n_bins_x, n_bins_y)
    if is_1d_pos:
        spatial_shape_pos = (n_bins,)
    else:
        spatial_shape_pos = (n_bins_x, n_bins_y)
    # Level labels (needed for mode 'level' and 'joint_lvl')
    labels_lvl = None
    uniq_lvls = None
    n_lvls = 0
    if mode in ("level", "joint_lvl"):
        if level_col not in selected_df.columns:
            raise ValueError(f"{mode} requires level_col in selected_df.")
        lv = selected_df[level_col].values
        uniq_lvls = np.unique(lv)
        lvl_map = {v: i for i, v in enumerate(uniq_lvls)}
        labels_lvl = np.array([lvl_map[v] for v in lv], dtype=int)
        n_lvls = len(uniq_lvls)
    # Head labels (needed for 'hd', 'joint_pos', 'joint_lvl')
    labels_hd = None
    n_hd = 0
    if mode in ("hd", "joint_pos", "joint_lvl"):
        if head_col not in selected_df.columns:
            raise ValueError(f"{mode} requires head_col in selected_df.")
        hv = selected_df[head_col].values
        uniq_hd = np.unique(hv)
        hd_map = {v: i for i, v in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[v] for v in hv], dtype=int)
        n_hd = len(uniq_hd)
    # Counts (T x N_sel)
    frames = np.arange(len(positions_arr))
    Q = build_count_matrix(
        spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids
    ).T
    dt_eff = dt * (time_window + 1)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    if mode == "spatial":
        lam_sp, pr_sp = tuner(
            Q,
            labels_pos,
            None,
            n_bins,
            dt_eff,
            filter_ratemap,
            sigma,
            eps=eps,
            decode_head="none",
            spatial_shape=spatial_shape_pos,
        )
        maps = lam_sp  # (n_bins, N)
        valid_mask = pr_sp > 0
        meta = dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges)
    elif mode == "level":
        # independent level tuning (no head)
        lam_lvl, pr_lvl = compute_tuning_and_prior(
            Q,
            labels_lvl,
            None,
            n_lvls,
            dt_eff,
            False,
            0.0,  # no smoothing for levels
            eps=eps,
            decode_head="none",
        )
        maps = lam_lvl  # (n_lvls, N)
        valid_mask = pr_lvl > 0
        meta = dict(n_lvls=n_lvls, uniq_lvls=uniq_lvls)
    elif mode == "hd":
        # independent head tuning (no space used in the output)
        # but the tuner still builds spatial maps internally; we pass spatial_shape_pos
        _, _, lam_hd, pr_hd = tuner(
            Q,
            labels_pos,
            labels_hd,
            n_bins,
            dt_eff,
            filter_ratemap,
            sigma,
            eps=eps,
            decode_head="independent",
            spatial_shape=spatial_shape_pos,
        )
        maps = lam_hd  # (n_hd, N)
        valid_mask = pr_hd > 0
        meta = dict(n_hd=n_hd)
    elif mode == "joint_pos":
        lam_jp, pr_jp = tuner(
            Q,
            labels_pos,
            labels_hd,
            n_bins,
            dt_eff,
            filter_ratemap,
            sigma,
            eps=eps,
            decode_head="joint_pos",
            spatial_shape=spatial_shape_pos,
        )
        maps = lam_jp.reshape(n_bins * n_hd, -1)  # (n_bins*n_hd, N)
        valid_mask = pr_jp.ravel() > 0
        meta = dict(n_bins=n_bins, n_hd=n_hd, x_edges=x_edges, y_edges=y_edges)
    elif mode == "joint_lvl":
        # returns (lam_sp, pr_sp, lam_lh, pr_lh)
        _, _, lam_lh, pr_lh = tuner(
            Q,
            labels_pos,
            labels_hd,
            n_bins,
            dt_eff,
            filter_ratemap,
            sigma,
            eps=eps,
            decode_head="joint_lvl",
            labels_lvl=labels_lvl,
            n_lvls=n_lvls,
            spatial_shape=spatial_shape_pos,
        )
        maps = lam_lh.reshape(n_lvls * n_hd, -1)  # (n_lvls*n_hd, N)
        valid_mask = pr_lh.ravel() > 0
        meta = dict(n_lvls=n_lvls, n_hd=n_hd, uniq_lvls=uniq_lvls)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    maps = np.asarray(maps, float)
    maps[maps < eps] = eps
    return maps, valid_mask, meta


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def _vector_similarity(a, b, metric="pearson"):
    """
    a, b: 1D arrays (same length), already masked/aligned.
    metric: 'pearson' | 'spearman' | 'cosine'
    """
    if metric == "pearson":
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    elif metric == "spearman":
        r, _ = spearmanr(a, b)
        return float(r)
    elif metric == "cosine":
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _normalize_columns(X, mode="l2", eps=1e-12):
    """
    Column-wise normalization.
    mode:
      - 'none'    : return X unchanged
      - 'l2'      : divide by sqrt(sum(x^2))
      - 'l1'      : divide by sum(|x|)
      - 'length'  : alias for 'l2'
      - 'zscore'  : (x - mean)/std
      - 'max'     : divide by column-wise max value (abs optional)
    Returns a copy.
    """
    X = np.asarray(X, float).copy()
    if mode in (None, "none"):
        return X
    if mode in ("l2", "length"):
        norms = np.linalg.norm(X, axis=0)
        return X / (norms + eps)
    if mode == "l1":
        norms = np.sum(np.abs(X), axis=0)
        return X / (norms + eps)
    if mode == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        return (X - mu) / (sd + eps)
    if mode == "max":
        max_vals = np.nanmax(
            np.abs(X), axis=0, keepdims=True
        )  # abs ensures stable scaling
        return X / (max_vals + eps)
    raise ValueError(f"Unknown normalization mode: {mode}")


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def _pick_matched_neurons_for_level(cimap, n_train, n_test, n_neurons=None, rng=None):
    """
    Resolve train/test neuron indices to compare.
    If cimap is None, fall back to same-index matching (capped by n_neurons and both session sizes).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if cimap is not None:
        pairs = np.array(cimap, int) - 1
        ok = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 0] < n_test)
            & (pairs[:, 1] < n_train)
        )
        pairs = pairs[ok]
        if n_neurons is not None and len(pairs) > n_neurons:
            sel = rng.choice(len(pairs), n_neurons, replace=False)
            pairs = pairs[sel]
        train_ids = pairs[:, 1].astype(int)
        test_ids = pairs[:, 0].astype(int)
    else:
        # same-index fallback
        m = n_train if n_neurons is None else min(n_neurons, n_train, n_test)
        ids = rng.choice(min(n_train, n_test), m, replace=False)
        train_ids = ids.astype(int)
        test_ids = ids.astype(int)
    return train_ids, test_ids


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: build]
def _build_level_labels(selected_df_train, selected_df_test, level_col):
    """Unify level indexing across sessions; return integer labels and the shared level list."""
    lv_tr_raw = selected_df_train[level_col].values
    lv_te_raw = selected_df_test[level_col].values
    uniq_lvls = np.unique(np.concatenate([lv_tr_raw, lv_te_raw]))
    lvl_to_idx = {v: i for i, v in enumerate(uniq_lvls)}
    labels_tr = np.array([lvl_to_idx[v] for v in lv_tr_raw], dtype=int)
    labels_te = np.array([lvl_to_idx[v] for v in lv_te_raw], dtype=int)
    return labels_tr, labels_te, uniq_lvls


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _windowed_counts(
    spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids
):
    """
    Wrapper around your build_count_matrix to get (T, N_sel) counts aligned to 'frames'.
    """
    Q = build_count_matrix(
        spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids
    ).T
    return Q  # (T, Nsel)


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map, level]
def _fit_level_map(Q, level_labels, K, dt_eff, tuner, eps=1e-12):
    """
    Fit level tuning maps: returns lam (K x Nsel), prior (K,), and a boolean valid mask (K,)
    marking nonzero-occupancy states used in the prior.
    """
    lam, pr = tuner(
        Q, level_labels, None, K, dt_eff, False, 0.0, eps=eps, decode_head="none"
    )
    valid = pr > 0
    return lam, pr, valid


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def _normalize_cols_level(X, mode="zscore", eps=1e-12):
    """
    Normalize across levels independently per neuron (column-wise).
    """
    if mode is None:
        return X
    if mode == "zscore":
        return zscore(X, axis=0, ddof=1, nan_policy="omit")
    if mode == "l2":
        n = np.linalg.norm(X, axis=0, keepdims=True)
        return X / (n + eps)
    if mode == "auto":
        # For circular lag search, zscore is typically the most stable default
        return zscore(X, axis=0, ddof=1, nan_policy="omit")
    raise ValueError("normalization must be 'zscore' | 'l2' | 'auto' | None")


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr]
def _best_circ_corr_and_lag(A, B, min_levels=3, eps=1e-12):
    """
    For each neuron (column), compute the best circularly-shifted Pearson correlation
    between A[:,i] and B[:,i], returning per-neuron (best_corr, best_lag).
    Lag is in bins and signed in [-floor(L/2), ceil(L/2)-1].
    """
    Kp, M = A.shape
    best_corr = np.full(M, np.nan, float)
    best_lag = np.full(M, np.nan, float)
    for i in range(M):
        a = A[:, i]
        b = B[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        a = a[good]
        b = b[good]
        L = len(a)
        if L < min_levels:
            continue
        az = (a - a.mean()) / (a.std(ddof=1) + eps)
        bz = (b - b.mean()) / (b.std(ddof=1) + eps)
        dots = np.empty(L)
        for s in range(L):
            r = np.roll(bz, s)
            dots[s] = np.dot(az, r) / (L - 1 + eps)
        s_best = int(np.argmax(dots))
        val = float(np.clip(dots[s_best], -1.0, 1.0))
        signed = s_best if s_best <= L // 2 else s_best - L
        best_corr[i] = val
        best_lag[i] = float(signed)
    return best_corr, best_lag


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _norm_row(v, mode, eps=1e-12):
    """
    Functionality:
    - Norm row.
    Inputs:
    - `v`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `mode`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `eps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `v` (type/shape from runtime context).
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if mode == "none":
        return v
    elif mode == "zscore":
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        return (v - mu) / (sd + eps)
    elif mode == "l2":
        return v / (np.linalg.norm(v) + eps)
    else:
        return v


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def _pv_from_maps(
    maps_tr,
    maps_te,
    mask_tr,
    mask_te,
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    min_neurons=3,
    eps=1e-12,
):
    """
    Compute PV correlation per bin given precomputed maps.
    maps_*: (K, M)  K=bins/states, M=neurons
    mask_*: (K,)
    Returns pv:(K,), mask:(K,)
    """
    assert maps_tr.shape == maps_te.shape
    K, M = maps_tr.shape
    if intersect_mask:
        mask = mask_tr.astype(bool) & mask_te.astype(bool)
    else:
        mask = mask_tr.astype(bool) | mask_te.astype(bool)
    if normalization == "auto":
        row_norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        row_norm_mode = normalization
    pv = np.full(K, np.nan, float)
    for b in range(K):
        if not mask[b]:
            continue
        a = maps_tr[b, :].astype(float, copy=False)
        c = maps_te[b, :].astype(float, copy=False)
        good = np.isfinite(a) & np.isfinite(c)
        if good.sum() < min_neurons:
            continue
        a = _norm_row(a[good], row_norm_mode, eps=eps)
        c = _norm_row(c[good], row_norm_mode, eps=eps)
        if metric == "pearson":
            pv[b] = np.corrcoef(a, c)[0, 1]
        elif metric == "spearman":
            pv[b] = spearmanr(a, c, nan_policy="omit")[0]
        else:  # cosine
            pv[b] = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + eps))
    return pv, mask


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def _prefit_train_and_test_maps(
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    train_ids,
    test_ids,
    mode="spatial",
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    eps=1e-12,
):
    """
    Fit maps once for observed PV (keeps neuron order aligned by train_ids vs test_ids).
    Returns: maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te
    """
    maps_tr, mask_tr, meta_tr = _fit_maps_for_session(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        positions_train,
        train_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    maps_te, mask_te, meta_te = _fit_maps_for_session(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        positions_test,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    return maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _impute_neuronwise(X, good_mask):
    # per-neuron (column) mean over kept rows
    """
    Functionality:
    - Impute neuronwise.
    Inputs:
    - `X`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `good_mask`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `Xf` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    has_any = np.any(good_mask, axis=0)
    col_means = np.zeros(X.shape[1], dtype=float)
    if np.any(has_any):
        col_means[has_any] = np.nanmean(
            np.where(good_mask[:, has_any], X[:, has_any], np.nan), axis=0
        )
    Xf = X.copy()
    bad = ~np.isfinite(Xf)
    if np.any(bad):
        Xf[bad] = col_means[np.newaxis, :][bad]
    # final safety
    Xf[~np.isfinite(Xf)] = 0.0
    return Xf


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _to_int_array(x):
    """
    Functionality:
    - To int array.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    return x.astype(int, copy=False)


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def cross_session_tuning_similarity(
    # data
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    # what to compare
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    # how to compute tuning
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    # which neurons to use
    n_neurons=None,
    random_state=0,
    train_ids=None,  # ZERO-BASED explicit train neuron indices
    test_ids=None,  # ZERO-BASED explicit test neuron indices
    # similarity
    metric="pearson",  # 'pearson' | 'spearman' | 'cosine'
    normalization="auto",  # 'auto' | 'none' | 'l2' | 'length' | 'l1' | 'zscore'
    intersect_mask=True,
    eps=1e-12,
    # NEW: firing-rate based masking (spikes per minute)
    min_spikes_per_min=None,  # float or None -> apply to BOTH sessions if set
    min_spikes_per_min_train=None,  # optional override for TRAIN
    min_spikes_per_min_test=None,  # optional override for TEST
    rate_source="spike",  # 'spike' (spike_counts_*) | 'raw' (raw_counts_*)
    drop_if_below="either",  # 'either' (drop if train<thr OR test<thr) | 'both' (keep if at least one passes)
):
    """
    Similarity of tuning between TRAIN and TEST for matched neurons.
    1D vs 2D positions
    ------------------
    - For 2D arenas:
        positions_* shape: (T, 2)
        n_bins_x, n_bins_y as usual (e.g. 15, 15 -> 225 bins).
    - For 1D tracks:
        positions_* shape: (T,) or (T, 1)
        REQUIRE: n_bins_y == 1, n_bins_x = number of 1D bins.
        Then n_bins = n_bins_x and spatial smoothing is done in 1D inside
        the tuning functions.
    NEW (rate gating):
      - min_spikes_per_min[,_train,_test]: thresholds to exclude low-firing neurons
        on a per-day basis (events per minute over entire session).
      - rate_source: which counts to use to compute rates ('spike' or 'raw').
      - drop_if_below:
           'either' -> drop pair if EITHER day is below its threshold (stringent).
           'both'   -> drop pair only if BOTH days are below threshold (lenient).
    Returns:
      dict(
        train_map, test_map,
        train_mask, test_mask,
        train_ids, test_ids,
        similarity, metric, mode,
        valid_counts, summary,
        meta_train, meta_test,
        dropped
      )
    """
    rng = np.random.RandomState(random_state)
    # --- Normalize positions & detect 1D vs 2D (for sanity checking) ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    # Basic consistency check: don't mix 1D vs 2D in the same call
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    # For 1D positions, enforce n_bins_y == 1 so that n_bins = n_bins_x
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # --- Helper: choose counts for rate computation ---
    if rate_source == "spike":
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == "raw":
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    # --- Compute spikes-per-minute per neuron for each session ---
    # total minutes = (num_frames * dt) / 60
    def _spikes_per_min(C, dt_sec):
        # C: (T, N)
        C = np.asarray(C, dtype=float)
        total_events = C.sum(axis=0)  # (N,)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        if total_minutes <= 0:
            return np.zeros(C.shape[1], dtype=float)
        return total_events / total_minutes
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    # --- Resolve neuron IDs (assumed ZERO-BASED if provided) ---
    if train_ids is not None and test_ids is not None:
        tr_ids = _to_int_array(train_ids)
        te_ids = _to_int_array(test_ids)
        if tr_ids.shape[0] != te_ids.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr_ids.size == 0:
            raise ValueError("Empty train_ids/test_ids after parsing.")
        # Range checks
        if tr_ids.min() < 0 or tr_ids.max() >= n_tr:
            raise ValueError(
                "train_ids out of range for TRAIN session (expected zero-based)."
            )
        if te_ids.min() < 0 or te_ids.max() >= n_te:
            raise ValueError(
                "test_ids out of range for TEST session (expected zero-based)."
            )
        # Optional subsample to n_neurons
        if n_neurons is not None and tr_ids.size > n_neurons:
            sel = rng.choice(tr_ids.size, int(n_neurons), replace=False)
            tr_ids = tr_ids[sel]
            te_ids = te_ids[sel]
        train_ids = tr_ids
        test_ids = te_ids
    else:
        # Fall back to cimap-based (or same-index) matching
        train_ids, test_ids = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
    # --- Apply firing-rate based masking on matched PAIRS ---
    # Determine thresholds
    thr_tr = (
        min_spikes_per_min_train
        if min_spikes_per_min_train is not None
        else min_spikes_per_min
    )
    thr_te = (
        min_spikes_per_min_test
        if min_spikes_per_min_test is not None
        else min_spikes_per_min
    )
    # Create pass flags (if threshold is None -> treat as passed)
    train_ids = np.asarray(train_ids)
    test_ids = np.asarray(test_ids)
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[train_ids] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[test_ids] >= float(thr_te)
    if drop_if_below == "either":
        keep_mask = pass_tr & pass_te
    elif drop_if_below == "both":
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    # Filter pairs
    train_ids = train_ids[keep_mask]
    test_ids = test_ids[keep_mask]
    # If nothing left after masking
    M = int(len(train_ids))
    if M == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=0)
        return dict(
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            similarity=np.array([], dtype=float),
            metric=metric,
            mode=mode,
            valid_counts=np.array([], dtype=int),
            summary=summary,
            meta_train=dict(
                rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
            ),
            meta_test=dict(
                rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
            ),
            dropped=dict(
                n_pairs_initial=int(len(keep_mask)),
                n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()),
                policy=drop_if_below,
            ),
        )
    # --- Fit maps on each session (only kept IDs) ---
    maps_tr, mask_tr, meta_tr = _fit_maps_for_session(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        train_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    maps_te, mask_te, meta_te = _fit_maps_for_session(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    # --- Choose common/union mask of states ---
    mask = (mask_tr & mask_te) if intersect_mask else (mask_tr | mask_te)
    idx = np.where(mask)[0]
    if idx.size == 0:
        sim = np.full(M, np.nan, float)
        valid_counts = np.zeros(M, int)
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=M)
        # Augment meta with thresholds
        meta_tr["rate_per_min"] = rate_tr
        meta_tr["threshold"] = thr_tr
        meta_tr["rate_source"] = rate_source
        meta_te["rate_per_min"] = rate_te
        meta_te["threshold"] = thr_te
        meta_te["rate_source"] = rate_source
        return dict(
            train_ids=train_ids,
            test_ids=test_ids,
            similarity=sim,
            metric=metric,
            mode=mode,
            valid_counts=valid_counts,
            summary=summary,
            meta_train=meta_tr,
            meta_test=meta_te,
            dropped=dict(
                n_pairs_initial=int(len(keep_mask)),
                n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()),
                policy=drop_if_below,
            ),
        )
    # --- Slice maps to common states ---
    A = maps_tr[idx, :]  # (K', M)
    B = maps_te[idx, :]  # (K', M)
    # --- Normalization policy ---
    if normalization == "auto":
        norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        norm_mode = normalization
    A = _normalize_columns(A, mode=norm_mode, eps=eps)
    B = _normalize_columns(B, mode=norm_mode, eps=eps)
    # --- Per-neuron similarity ---
    sim = np.empty(M, float)
    valid_counts = np.empty(M, int)
    for i in range(M):
        a = A[:, i]
        b = B[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        valid_counts[i] = int(good.sum())
        sim[i] = (
            np.nan
            if valid_counts[i] == 0
            else _vector_similarity(a[good], b[good], metric=metric)
        )
    summary = dict(
        mean=float(np.nanmean(sim)),
        std=float(np.nanstd(sim)),
        median=float(np.nanmedian(sim)),
        n=int(M),
    )
    # Augment meta with thresholds + per-neuron rate vectors
    meta_tr["rate_per_min"] = rate_tr
    meta_tr["threshold"] = thr_tr
    meta_tr["rate_source"] = rate_source
    meta_te["rate_per_min"] = rate_te
    meta_te["threshold"] = thr_te
    meta_te["rate_source"] = rate_source
    return dict(
        train_map=maps_tr,
        test_map=maps_te,
        train_mask=mask_tr,
        test_mask=mask_te,
        train_ids=train_ids,
        test_ids=test_ids,
        similarity=sim,
        metric=metric,
        mode=mode,
        valid_counts=valid_counts,
        summary=summary,
        meta_train=meta_tr,
        meta_test=meta_te,
        dropped=dict(
            n_pairs_initial=int(len(keep_mask)),
            n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()),
            policy=drop_if_below,
        ),
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity, shuffle]
def pf_shuffle_tuning_similarity(
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    mode="spatial",
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    n_neurons=None,
    random_state=0,
    # explicit ZERO-BASED neuron selections (override cimap if both provided)
    train_ids=None,
    test_ids=None,
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    eps=1e-12,
    n_repeats=100,
    shuffle_session="test",  # 'test' or 'train'
    shuffle_type="map",  # 'map' or 'neuron_id'
    # ---- NEW: firing-rate masking (same as cross_session_tuning_similarity) ----
    min_spikes_per_min=None,  # float or None -> apply to BOTH sessions if set
    min_spikes_per_min_train=None,  # optional override for TRAIN
    min_spikes_per_min_test=None,  # optional override for TEST
    rate_source="spike",  # 'spike' uses spike_counts_* ; 'raw' uses raw_counts_*
    drop_if_below="either",  # 'either' (strict) | 'both' (lenient)
):
    """
    Cross-session per-neuron tuning similarity nulls WITH firing-rate-based masking.
    1D vs 2D positions
    ------------------
    - 2D: positions_* shape (T, 2), n_bins_x Ã— n_bins_y as usual.
    - 1D: positions_* shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and
          n_bins_x = number of spatial bins (so n_bins = n_bins_x).
    shuffle_type:
      - 'map'       : permute the BIN/STATE axis within each neuron's map in one session
                      (destroys topography; preserves trainâ†”test neuron identity).
      - 'neuron_id' : permute the neuron columns in one session
                      (breaks trainâ†”test neuron correspondence; preserves each map).
    """
    try:
        from scipy.stats import spearmanr as _scipy_spearmanr
    except Exception:
        _scipy_spearmanr = None
    if shuffle_session not in ("test", "train"):
        raise ValueError("shuffle_session must be 'test' or 'train'")
    if shuffle_type not in ("map", "neuron_id"):
        raise ValueError("shuffle_type must be 'map' or 'neuron_id'")
    rng = np.random.RandomState(random_state)
    # --- Normalize positions & detect 1D vs 2D ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # -------- pick counts used to compute firing rates --------
    if rate_source == "spike":
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == "raw":
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, dtype=float).sum(axis=0)  # (N,)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        if total_minutes <= 0:
            return np.zeros(C.shape[1], dtype=float)
        return total_events / total_minutes
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    # -------- resolve matched neuron IDs --------
    if train_ids is not None and test_ids is not None:
        tr_ids = _to_int_array(train_ids)
        te_ids = _to_int_array(test_ids)
        if tr_ids.shape[0] != te_ids.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr_ids.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr_ids.min() < 0 or tr_ids.max() >= n_tr:
            raise ValueError("train_ids out of range for TRAIN (zero-based).")
        if te_ids.min() < 0 or te_ids.max() >= n_te:
            raise ValueError("test_ids out of range for TEST (zero-based).")
        if n_neurons is not None and tr_ids.size > n_neurons:
            sel = rng.choice(tr_ids.size, int(n_neurons), replace=False)
            tr_ids = tr_ids[sel]
            te_ids = te_ids[sel]
        train_ids = tr_ids
        test_ids = te_ids
    else:
        train_ids, test_ids = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
    # -------- apply firing-rate-based masking on matched PAIRS --------
    thr_tr = (
        min_spikes_per_min_train
        if min_spikes_per_min_train is not None
        else min_spikes_per_min
    )
    thr_te = (
        min_spikes_per_min_test
        if min_spikes_per_min_test is not None
        else min_spikes_per_min
    )
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == "either":
        keep_mask = pass_tr & pass_te
    elif drop_if_below == "both":
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    train_ids = np.asarray(train_ids)[keep_mask]
    test_ids = np.asarray(test_ids)[keep_mask]
    M = int(len(train_ids))
    if M == 0:
        return dict(
            sim_matrix=np.zeros((0, 0), float),
            sim_per_repeat=np.zeros((0,), float),
            mean=np.nan,
            std=np.nan,
            mode=mode,
            metric=metric,
            normalization=normalization,
            n_repeats=int(n_repeats),
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            shuffle_type=shuffle_type,
            shuffle_session=shuffle_session,
            meta_train=dict(
                rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
            ),
            meta_test=dict(
                rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
            ),
            dropped=dict(
                n_pairs_initial=int(keep_mask.size),
                n_pairs_dropped=int(keep_mask.size - keep_mask.sum()),
                policy=drop_if_below,
            ),
        )
    # -------- build maps (rows=states/bins K, cols=neurons M) --------
    maps_tr, mask_tr, meta_tr = _fit_maps_for_session(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        train_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    maps_te, mask_te, meta_te = _fit_maps_for_session(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    if maps_tr.shape != maps_te.shape:
        raise ValueError(
            f"Train/Test maps shape mismatch after selection: {maps_tr.shape} vs {maps_te.shape}"
        )
    # common mask over states/bins
    mask = (mask_tr & mask_te) if intersect_mask else (mask_tr | mask_te)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return dict(
            sim_matrix=np.zeros((0, M), float),
            sim_per_repeat=np.zeros((0,), float),
            mean=np.nan,
            std=np.nan,
            mode=mode,
            metric=metric,
            normalization=normalization,
            n_repeats=int(n_repeats),
            train_ids=train_ids,
            test_ids=test_ids,
            shuffle_type=shuffle_type,
            shuffle_session=shuffle_session,
            meta_train=meta_tr
            | dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source),
            meta_test=meta_te
            | dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source),
            dropped=dict(
                n_pairs_initial=int(keep_mask.size),
                n_pairs_dropped=int(keep_mask.size - keep_mask.sum()),
                policy=drop_if_below,
            ),
        )
    # pre-sliced bases
    A_full = maps_tr
    B_full = maps_te
    A_base = A_full[idx, :]
    B_base = B_full[idx, :]
    K_full = A_full.shape[0]
    # normalization policy
    if normalization == "auto":
        norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        norm_mode = normalization
    def _normalize_columns(X, mode="l2", eps=1e-12):
        X = X.astype(float, copy=True)
        if mode == "zscore":
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True) + eps
            return (X - mu) / sd
        elif mode in ("l2", "length"):
            nrm = np.linalg.norm(X, axis=0, keepdims=True) + eps
            return X / nrm
        elif mode == "l1":
            nrm = np.abs(X).sum(axis=0, keepdims=True) + eps
            return X / nrm
        elif mode == "none":
            return X
        else:
            return X  # default fallback
    def _spearman(u, v):
        if _scipy_spearmanr is not None:
            return float(_scipy_spearmanr(u, v)[0])
        # rank-transform fallback
        ru = np.argsort(np.argsort(u))
        rv = np.argsort(np.argsort(v))
        u0 = (ru - ru.mean()) / (ru.std() + eps)
        v0 = (rv - rv.mean()) / (rv.std() + eps)
        return float(np.dot(u0, v0) / (np.linalg.norm(u0) * np.linalg.norm(v0) + eps))
    sim_rows = []
    sim_means = []
    for _ in range(n_repeats):
        if shuffle_type == "map":
            if shuffle_session == "test":
                B_shuf = B_full.copy()
                for n in range(M):
                    perm = rng.permutation(K_full)
                    B_shuf[:, n] = B_full[perm, n]
                A_use = A_base
                B_use = B_shuf[idx, :]
            else:
                A_shuf = A_full.copy()
                for n in range(M):
                    perm = rng.permutation(K_full)
                    A_shuf[:, n] = A_full[perm, n]
                A_use = A_shuf[idx, :]
                B_use = B_base
        else:  # 'neuron_id'
            if shuffle_session == "test":
                perm_cols = rng.permutation(M)
                A_use = A_base
                B_use = B_base[:, perm_cols]
            else:
                perm_cols = rng.permutation(M)
                A_use = A_base[:, perm_cols]
                B_use = B_base
        A_n = _normalize_columns(A_use, mode=norm_mode, eps=eps)
        B_n = _normalize_columns(B_use, mode=norm_mode, eps=eps)
        vals = np.empty(M, float)
        for i in range(M):
            a = A_n[:, i]
            b = B_n[:, i]
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() == 0:
                vals[i] = np.nan
            else:
                aa, bb = a[good], b[good]
                if metric == "pearson":
                    aa0 = aa - aa.mean()
                    bb0 = bb - bb.mean()
                    denom = np.linalg.norm(aa0) * np.linalg.norm(bb0) + eps
                    vals[i] = float(np.dot(aa0, bb0) / denom)
                elif metric == "spearman":
                    vals[i] = _spearman(aa, bb)
                else:  # 'cosine'
                    denom = np.linalg.norm(aa) * np.linalg.norm(bb) + eps
                    vals[i] = float(np.dot(aa, bb) / denom)
        sim_rows.append(vals)
        sim_means.append(np.nanmean(vals))
    sim_matrix = np.vstack(sim_rows) if sim_rows else np.zeros((0, M), float)
    sim_per_repeat = np.asarray(sim_means, float)
    # augment metas with rate info & thresholds for auditing
    meta_tr |= dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source)
    meta_te |= dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source)
    return dict(
        sim_matrix=sim_matrix,
        sim_per_repeat=sim_per_repeat,
        mean=float(np.nanmean(sim_per_repeat)) if sim_per_repeat.size else np.nan,
        std=float(np.nanstd(sim_per_repeat)) if sim_per_repeat.size else np.nan,
        mode=mode,
        metric=metric,
        normalization=normalization,
        n_repeats=int(n_repeats),
        train_ids=train_ids,
        test_ids=test_ids,
        shuffle_type=shuffle_type,
        shuffle_session=shuffle_session,
        meta_train=meta_tr,
        meta_test=meta_te,
        dropped=dict(
            n_pairs_initial=int(keep_mask.size),
            n_pairs_dropped=int(keep_mask.size - keep_mask.sum()),
            policy=drop_if_below,
        ),
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def _wlc_build_majority_bin_levels(
    positions_train,
    selected_df_train,
    positions_test,
    selected_df_test,
    level_col="level",
    n_bins_x=15,
    n_bins_y=15,
    majority_min_frac=0.5,
):
    """
    Build a single spatial-bin -> level codebook by majority vote over train+test frames.

    Returns
    -------
    bin_level : (K,) int
        Contiguous level index per spatial bin; -1 means unresolved/unknown.
    meta : dict
        Includes vote counts and the original level values mapped to contiguous indices.
    """
    lab_tr, _, _ = discretize_positions(positions_train, n_bins_x, n_bins_y)
    lab_te, _, _ = discretize_positions(positions_test, n_bins_x, n_bins_y)
    lv_tr_raw = np.asarray(selected_df_train[level_col].values)
    lv_te_raw = np.asarray(selected_df_test[level_col].values)
    if lv_tr_raw.shape[0] != lab_tr.shape[0]:
        raise ValueError(
            "selected_df_train level_col length must match positions_train length."
        )
    if lv_te_raw.shape[0] != lab_te.shape[0]:
        raise ValueError(
            "selected_df_test level_col length must match positions_test length."
        )
    valid_tr = ~pd.isna(lv_tr_raw)
    valid_te = ~pd.isna(lv_te_raw)
    uniq_vals = np.unique(np.concatenate([lv_tr_raw[valid_tr], lv_te_raw[valid_te]]))
    K = int(n_bins_x * n_bins_y)
    if uniq_vals.size == 0:
        return np.full(K, -1, dtype=int), dict(
            uniq_level_values=np.array([], dtype=float),
            votes=np.zeros((K, 0), dtype=int),
            majority_frac=np.zeros(K, dtype=float),
            visited_counts=np.zeros(K, dtype=int),
            majority_min_frac=float(majority_min_frac),
        )
    level_to_idx = {v: i for i, v in enumerate(uniq_vals.tolist())}
    lv_tr = np.array([level_to_idx[v] for v in lv_tr_raw[valid_tr]], dtype=int)
    lv_te = np.array([level_to_idx[v] for v in lv_te_raw[valid_te]], dtype=int)
    votes = np.zeros((K, uniq_vals.size), dtype=int)
    np.add.at(votes, (lab_tr[valid_tr], lv_tr), 1)
    np.add.at(votes, (lab_te[valid_te], lv_te), 1)
    visited_counts = votes.sum(axis=1)
    best = np.argmax(votes, axis=1)
    best_count = votes[np.arange(K), best]
    majority_frac = np.zeros(K, dtype=float)
    mask_vis = visited_counts > 0
    majority_frac[mask_vis] = best_count[mask_vis] / visited_counts[mask_vis]
    bin_level = np.full(K, -1, dtype=int)
    keep = mask_vis & (majority_frac >= float(majority_min_frac))
    bin_level[keep] = best[keep]
    return bin_level, dict(
        uniq_level_values=uniq_vals,
        votes=votes,
        majority_frac=majority_frac,
        visited_counts=visited_counts,
        majority_min_frac=float(majority_min_frac),
    )


def _wlc_neighbors_4(bin_idx, n_bins_x, n_bins_y):
    """4-neighborhood in flattened x + y*n_bins_x indexing."""
    x = int(bin_idx % n_bins_x)
    y = int(bin_idx // n_bins_x)
    out = []
    if x > 0:
        out.append(bin_idx - 1)
    if x < n_bins_x - 1:
        out.append(bin_idx + 1)
    if y > 0:
        out.append(bin_idx - n_bins_x)
    if y < n_bins_y - 1:
        out.append(bin_idx + n_bins_x)
    return out


def _wlc_order_component_bins(component_bins_global, n_bins_x, n_bins_y):
    """
    Build a deterministic order over bins in one connected component.

    - path-like components: endpoint-to-endpoint walk
    - cycle-like components: cyclic walk
    - branched components: BFS order from smallest-index seed
    """
    bins = np.asarray(component_bins_global, dtype=int)
    if bins.size <= 1:
        return bins.copy()
    bin_set = set(bins.tolist())
    neigh = {b: [] for b in bins.tolist()}
    for b in bins.tolist():
        for nb in _wlc_neighbors_4(b, n_bins_x, n_bins_y):
            if nb in bin_set:
                neigh[b].append(nb)
        neigh[b].sort()
    deg = {b: len(neigh[b]) for b in bins.tolist()}
    n = bins.size
    deg1 = [b for b in bins.tolist() if deg[b] == 1]
    maxdeg = max(deg.values()) if n > 0 else 0
    # Path
    if maxdeg <= 2 and len(deg1) == 2:
        start = min(deg1)
        order = [start]
        prev = None
        cur = start
        for _ in range(n - 1):
            nxts = [x for x in neigh[cur] if x != prev]
            if not nxts:
                break
            nxt = min(nxts)
            order.append(nxt)
            prev, cur = cur, nxt
        if len(order) == n:
            return np.asarray(order, dtype=int)
    # Cycle
    if maxdeg == 2 and all(d == 2 for d in deg.values()):
        start = int(np.min(bins))
        nbs = neigh[start]
        if len(nbs) >= 1:
            order = [start]
            prev = None
            cur = start
            for _ in range(n - 1):
                nxts = [x for x in neigh[cur] if x != prev]
                if not nxts:
                    break
                # keep deterministic traversal direction
                nxt = min(nxts) if prev is None else nxts[0]
                if nxt in order:
                    break
                order.append(nxt)
                prev, cur = cur, nxt
            if len(order) == n:
                return np.asarray(order, dtype=int)
    # Branched / fallback: BFS
    seed = int(np.min(bins))
    q = deque([seed])
    seen = {seed}
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in neigh[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return np.asarray(order, dtype=int)


def _wlc_build_components(bin_level_global, kept_bins_global, n_bins_x, n_bins_y):
    """
    Connected components over kept spatial bins, constrained to same level.

    Returns list of dict(level, bins_global).
    """
    kept_bins_global = np.asarray(kept_bins_global, dtype=int)
    kept_set = set(kept_bins_global.tolist())
    visited = set()
    comps = []
    for b0 in kept_bins_global.tolist():
        if b0 in visited:
            continue
        lvl0 = int(bin_level_global[b0])
        stack = [b0]
        visited.add(b0)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in _wlc_neighbors_4(u, n_bins_x, n_bins_y):
                if (
                    v in kept_set
                    and v not in visited
                    and int(bin_level_global[v]) == lvl0
                ):
                    visited.add(v)
                    stack.append(v)
        comp = np.asarray(sorted(comp), dtype=int)
        comps.append(dict(level=lvl0, bins_global=comp))
    return comps


def _wlc_sample_component_mapping(order_local, rng, reverse_prob=0.5):
    """
    Sample one topology-preserving row mapping for one component.

    - size 1: identity
    - size 2: identity/swap
    - size >=3: circular shift (+ optional reversal)
    """
    order_local = np.asarray(order_local, dtype=int)
    n = order_local.size
    if n <= 1:
        return order_local.copy()
    if n == 2:
        if rng.rand() < 0.5:
            return order_local[::-1].copy()
        return order_local.copy()
    k = int(rng.randint(0, n))
    mapped = np.roll(order_local, k)
    if rng.rand() < float(reverse_prob):
        mapped = mapped[::-1]
    return mapped.copy()


def _wlc_similarity_per_neuron(A, B, metric="pearson", normalization="auto", eps=1e-12):
    """Column-wise similarity between two (K, M) matrices."""
    if normalization == "auto":
        norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        norm_mode = normalization
    A_n = _normalize_columns(A, mode=norm_mode, eps=eps)
    B_n = _normalize_columns(B, mode=norm_mode, eps=eps)
    M = A_n.shape[1]
    sim = np.full(M, np.nan, dtype=float)
    valid_counts = np.zeros(M, dtype=int)
    for i in range(M):
        a = A_n[:, i]
        b = B_n[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        valid_counts[i] = int(good.sum())
        if valid_counts[i] == 0:
            continue
        sim[i] = _vector_similarity(a[good], b[good], metric=metric)
    return sim, valid_counts


def _wlc_residualize_by_level(X, level_idx_rows):
    """
    Remove per-neuron level means from map rows.

    X: (K, M), level_idx_rows: (K,) contiguous level id, -1 means unknown.
    """
    Xr = np.asarray(X, dtype=float).copy()
    lv = np.asarray(level_idx_rows, dtype=int)
    for l in np.unique(lv):
        if l < 0:
            continue
        ridx = np.where(lv == l)[0]
        if ridx.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Xr[ridx, :], axis=0, keepdims=True)
        Xr[ridx, :] = Xr[ridx, :] - mu
    return Xr


def _wlc_edges_local_from_components(components, global_to_local, n_bins_x, n_bins_y):
    """Edges (u,v) in local row coordinates, within components."""
    edges = []
    for comp in components:
        bins = comp["bins_global"]
        bset = set(bins.tolist())
        for u in bins.tolist():
            for v in _wlc_neighbors_4(u, n_bins_x, n_bins_y):
                if v in bset and u < v:
                    edges.append((global_to_local[u], global_to_local[v]))
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(edges, dtype=int)


def _wlc_laplacian_energy_cols(X, edges):
    """
    Per-neuron roughness (mean squared neighbor difference) over given edges.
    """
    X = np.asarray(X, dtype=float)
    edges = np.asarray(edges, dtype=int)
    M = X.shape[1]
    if edges.size == 0:
        return np.full(M, np.nan, dtype=float)
    u = edges[:, 0]
    v = edges[:, 1]
    D = X[u, :] - X[v, :]
    with np.errstate(invalid="ignore"):
        E = np.nanmean(D * D, axis=0)
    return E


def _wlc_component_graph_smooth_cols(X, edges, alpha=0.35, n_iters=1):
    """
    Diffusion-style smoothing over component graph edges.
    Operates per neuron (columns) and never mixes across disconnected bins.
    """
    X = np.asarray(X, dtype=float)
    edges = np.asarray(edges, dtype=int)
    alpha = float(alpha)
    n_iters = int(n_iters)
    if edges.size == 0 or n_iters <= 0 or alpha <= 0.0:
        return X.copy()
    alpha = max(0.0, min(1.0, alpha))
    K, M = X.shape
    u = edges[:, 0]
    v = edges[:, 1]
    deg = np.zeros(K, dtype=float)
    np.add.at(deg, u, 1.0)
    np.add.at(deg, v, 1.0)
    has_nb = deg > 0
    out = X.copy()
    for _ in range(n_iters):
        nb_sum = np.zeros((K, M), dtype=float)
        np.add.at(nb_sum, u, out[v, :])
        np.add.at(nb_sum, v, out[u, :])
        nb_avg = out.copy()
        nb_avg[has_nb, :] = nb_sum[has_nb, :] / deg[has_nb, None]
        out = (1.0 - alpha) * out + alpha * nb_avg
    return out


def _wlc_smooth_to_target_ratio(
    X,
    edges,
    obs_energy,
    target=1.0,
    max_iters=4,
    alpha=0.35,
):
    """
    Try 0..max_iters smoothing steps and keep the map whose median
    smoothness ratio (shuffle/observed Laplacian energy) is closest to target.
    """
    X = np.asarray(X, dtype=float)
    obs_energy = np.asarray(obs_energy, dtype=float)
    target = float(target)
    max_iters = int(max_iters)
    alpha = float(alpha)
    best_X = X.copy()
    best_iter = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio0 = _wlc_laplacian_energy_cols(best_X, edges) / obs_energy
    best_ratio = float(np.nanmedian(ratio0))
    if np.isfinite(best_ratio):
        best_score = abs(best_ratio - target)
    else:
        best_score = np.inf
    cur = X.copy()
    for it in range(1, max_iters + 1):
        cur = _wlc_component_graph_smooth_cols(cur, edges, alpha=alpha, n_iters=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = _wlc_laplacian_energy_cols(cur, edges) / obs_energy
        ratio_med = float(np.nanmedian(ratio))
        score = abs(ratio_med - target) if np.isfinite(ratio_med) else np.inf
        if score < best_score:
            best_score = score
            best_ratio = ratio_med
            best_X = cur.copy()
            best_iter = int(it)
    return best_X, best_ratio, best_iter


def _wlc_empirical_p_ge(obs, null_vals_2d):
    """
    One-sided empirical p-value P(null >= obs), with +1 correction.
    """
    obs = np.asarray(obs, dtype=float)
    null_vals_2d = np.asarray(null_vals_2d, dtype=float)
    if null_vals_2d.ndim == 1:
        null_vals_2d = null_vals_2d[:, None]
    M = obs.shape[0]
    p = np.full(M, np.nan, dtype=float)
    for i in range(M):
        oi = obs[i]
        col = null_vals_2d[:, i]
        good = np.isfinite(col)
        if not np.isfinite(oi) or good.sum() == 0:
            continue
        c = int(np.sum(col[good] >= oi))
        p[i] = (c + 1.0) / (good.sum() + 1.0)
    return p


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle, level, spatial, corr]
def within_level_component_shuffle_spatial_similarity(
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    level_col="level",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    n_neurons=None,
    random_state=0,
    train_ids=None,
    test_ids=None,
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    eps=1e-12,
    min_spikes_per_min=None,
    min_spikes_per_min_train=None,
    min_spikes_per_min_test=None,
    rate_source="spike",
    drop_if_below="either",
    n_repeats=1000,
    shuffle_session="test",  # 'test' | 'train'
    transform_scope="global",  # 'global' | 'per_neuron'
    majority_min_frac=0.5,
    drop_unknown_level_bins=True,
    reverse_prob=0.5,
    smoothness_match=False,
    smoothness_target=1.0,
    smoothness_tol=0.2,
    max_attempts_multiplier=20,
    smoothness_filter=False,
    smoothness_filter_mode="adaptive",  # 'adaptive' | 'fixed'
    smoothness_filter_alpha=0.35,
    smoothness_filter_n_iters=1,  # used when mode='fixed'
    smoothness_filter_max_iters=4,  # used when mode='adaptive'
    verbose=False,
    progress_every=25,
    return_null_matrix=True,
):
    """
    Constrained null for place-map similarity:
      shuffle only *within connected components of the same level*.

    This tests whether observed spatial similarity exceeds what is explained
    by level structure + local continuity.

    Workflow:
      1) Build observed spatial maps using cross_session_tuning_similarity(..., mode='spatial')
      2) Infer a shared bin->level codebook by majority vote over train+test frames
      3) Build same-level connected components on kept spatial bins
      4) Shuffle by component-wise circular-shift/reversal transforms
         - transform_scope='global'     : one shared component transform for all neurons per repeat
         - transform_scope='per_neuron' : independent component transforms per neuron per repeat
      5) Compute raw and level-residualized similarities vs null
      6) Report smoothness diagnostics (Laplacian energy ratios) for shuffled session
      7) Optional smoothness-matched null filtering:
         keep only shuffles with median(shuf_energy/obs_energy) within
         smoothness_target +/- smoothness_tol.

    Returns
    -------
    dict with keys:
      - observed_similarity, observed_similarity_resid
      - observed_summary, observed_summary_resid
      - null_mean_raw, null_mean_resid
      - p_mean_raw, p_mean_resid
      - p_neuron_raw, p_neuron_resid
      - null_raw_matrix/null_resid_matrix (optional)
      - components, component_space, smoothness, dropped, ids, masks
    """
    if shuffle_session not in ("test", "train"):
        raise ValueError("shuffle_session must be 'test' or 'train'.")
    if transform_scope not in ("global", "per_neuron"):
        raise ValueError("transform_scope must be 'global' or 'per_neuron'.")
    if smoothness_filter_mode not in ("adaptive", "fixed"):
        raise ValueError("smoothness_filter_mode must be 'adaptive' or 'fixed'.")
    if float(smoothness_tol) < 0:
        raise ValueError("smoothness_tol must be >= 0.")
    if int(max_attempts_multiplier) < 1:
        raise ValueError("max_attempts_multiplier must be >= 1.")
    if float(smoothness_filter_alpha) < 0 or float(smoothness_filter_alpha) > 1:
        raise ValueError("smoothness_filter_alpha must be in [0, 1].")
    if int(smoothness_filter_n_iters) < 0:
        raise ValueError("smoothness_filter_n_iters must be >= 0.")
    if int(smoothness_filter_max_iters) < 0:
        raise ValueError("smoothness_filter_max_iters must be >= 0.")
    if int(progress_every) < 1:
        raise ValueError("progress_every must be >= 1.")
    if bool(verbose):
        from time import perf_counter as _wlc_perf_counter

        _wlc_t0 = _wlc_perf_counter()

        def _wlc_log(msg):
            elapsed = _wlc_perf_counter() - _wlc_t0
            print(f"[WLC {elapsed:7.1f}s] {msg}")

    else:
        _wlc_log = None
    rng = np.random.RandomState(random_state)
    # 1) Observed spatial maps and matched IDs (with same gating policy as main pipeline)
    obs = cross_session_tuning_similarity(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        positions_train,
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        positions_test,
        cimap=cimap,
        mode="spatial",
        level_col=level_col,
        head_col="head_cat",
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        n_neurons=n_neurons,
        random_state=random_state,
        train_ids=train_ids,
        test_ids=test_ids,
        metric=metric,
        normalization=normalization,
        intersect_mask=intersect_mask,
        eps=eps,
        min_spikes_per_min=min_spikes_per_min,
        min_spikes_per_min_train=min_spikes_per_min_train,
        min_spikes_per_min_test=min_spikes_per_min_test,
        rate_source=rate_source,
        drop_if_below=drop_if_below,
    )
    train_ids_used = np.asarray(obs.get("train_ids", []), dtype=int)
    test_ids_used = np.asarray(obs.get("test_ids", []), dtype=int)
    if train_ids_used.size == 0:
        return dict(
            observed=obs,
            observed_similarity=np.array([], dtype=float),
            observed_similarity_resid=np.array([], dtype=float),
            observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            null_mean_raw=np.array([], dtype=float),
            null_mean_resid=np.array([], dtype=float),
            p_mean_raw=np.nan,
            p_mean_resid=np.nan,
            p_neuron_raw=np.array([], dtype=float),
            p_neuron_resid=np.array([], dtype=float),
            null_raw_matrix=np.zeros((0, 0), dtype=float),
            null_resid_matrix=np.zeros((0, 0), dtype=float),
            components=[],
            component_space=dict(log10_n_states=0.0, n_nontrivial=0),
            smoothness=dict(
                observed_energy=np.array([], dtype=float),
                null_ratio=np.array([], dtype=float),
                ratio_mean=np.nan,
                ratio_std=np.nan,
            ),
            dropped=obs.get("dropped", {}),
            ids=dict(train_ids=train_ids_used, test_ids=test_ids_used),
            kept_bins_global=np.array([], dtype=int),
            kept_levels=np.array([], dtype=int),
            majority_meta=None,
        )
    maps_tr = np.asarray(obs["train_map"], dtype=float)
    maps_te = np.asarray(obs["test_map"], dtype=float)
    mask_tr = np.asarray(obs["train_mask"], dtype=bool)
    mask_te = np.asarray(obs["test_mask"], dtype=bool)
    mask = (mask_tr & mask_te) if intersect_mask else (mask_tr | mask_te)
    idx_states = np.where(mask)[0].astype(int)
    if idx_states.size == 0:
        return dict(
            observed=obs,
            observed_similarity=np.array([], dtype=float),
            observed_similarity_resid=np.array([], dtype=float),
            observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            null_mean_raw=np.array([], dtype=float),
            null_mean_resid=np.array([], dtype=float),
            p_mean_raw=np.nan,
            p_mean_resid=np.nan,
            p_neuron_raw=np.array([], dtype=float),
            p_neuron_resid=np.array([], dtype=float),
            null_raw_matrix=np.zeros((0, train_ids_used.size), dtype=float),
            null_resid_matrix=np.zeros((0, train_ids_used.size), dtype=float),
            components=[],
            component_space=dict(log10_n_states=0.0, n_nontrivial=0),
            smoothness=dict(
                observed_energy=np.array([], dtype=float),
                null_ratio=np.array([], dtype=float),
                ratio_mean=np.nan,
                ratio_std=np.nan,
            ),
            dropped=obs.get("dropped", {}),
            ids=dict(train_ids=train_ids_used, test_ids=test_ids_used),
            kept_bins_global=np.array([], dtype=int),
            kept_levels=np.array([], dtype=int),
            majority_meta=None,
        )
    # 2) shared bin->level codebook
    bin_level_global, majority_meta = _wlc_build_majority_bin_levels(
        positions_train,
        selected_df_train,
        positions_test,
        selected_df_test,
        level_col=level_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        majority_min_frac=majority_min_frac,
    )
    if drop_unknown_level_bins:
        keep_lvl = bin_level_global[idx_states] >= 0
        idx_states = idx_states[keep_lvl]
    if idx_states.size == 0:
        return dict(
            observed=obs,
            observed_similarity=np.array([], dtype=float),
            observed_similarity_resid=np.array([], dtype=float),
            observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0),
            null_mean_raw=np.array([], dtype=float),
            null_mean_resid=np.array([], dtype=float),
            p_mean_raw=np.nan,
            p_mean_resid=np.nan,
            p_neuron_raw=np.array([], dtype=float),
            p_neuron_resid=np.array([], dtype=float),
            null_raw_matrix=np.zeros((0, train_ids_used.size), dtype=float),
            null_resid_matrix=np.zeros((0, train_ids_used.size), dtype=float),
            components=[],
            component_space=dict(log10_n_states=0.0, n_nontrivial=0),
            smoothness=dict(
                observed_energy=np.array([], dtype=float),
                null_ratio=np.array([], dtype=float),
                ratio_mean=np.nan,
                ratio_std=np.nan,
            ),
            dropped=obs.get("dropped", {}),
            ids=dict(train_ids=train_ids_used, test_ids=test_ids_used),
            kept_bins_global=np.array([], dtype=int),
            kept_levels=np.array([], dtype=int),
            majority_meta=majority_meta,
        )
    # 3) extract kept rows and level indices
    kept_bins_global = idx_states.astype(int)
    kept_levels = bin_level_global[kept_bins_global].astype(int)
    A_base = maps_tr[kept_bins_global, :]
    B_base = maps_te[kept_bins_global, :]
    Kk, M = A_base.shape
    # 4) components + deterministic order per component
    shuffle_bins_global = kept_bins_global[kept_levels >= 0]
    comps_global = _wlc_build_components(
        bin_level_global, shuffle_bins_global, n_bins_x, n_bins_y
    )
    global_to_local = {g: i for i, g in enumerate(kept_bins_global.tolist())}
    component_orders_local = []
    components = []
    log10_n_states = 0.0
    n_nontrivial = 0
    for c in comps_global:
        ord_global = _wlc_order_component_bins(c["bins_global"], n_bins_x, n_bins_y)
        ord_local = np.array([global_to_local[g] for g in ord_global], dtype=int)
        component_orders_local.append(ord_local)
        sz = int(ord_local.size)
        n_states = 1 if sz <= 1 else (2 if sz == 2 else (2 * sz))
        if n_states > 1:
            n_nontrivial += 1
        log10_n_states += np.log10(float(n_states))
        components.append(
            dict(
                level=int(c["level"]),
                size=sz,
                bins_global=c["bins_global"],
                order_global=ord_global,
                n_states=int(n_states),
            )
        )
    # 5) observed raw + residualized
    obs_raw, obs_valid = _wlc_similarity_per_neuron(
        A_base, B_base, metric=metric, normalization=normalization, eps=eps
    )
    A_res_base = _wlc_residualize_by_level(A_base, kept_levels)
    B_res_base = _wlc_residualize_by_level(B_base, kept_levels)
    obs_resid, obs_resid_valid = _wlc_similarity_per_neuron(
        A_res_base, B_res_base, metric=metric, normalization=normalization, eps=eps
    )
    obs_summary = dict(
        mean=float(np.nanmean(obs_raw)),
        std=float(np.nanstd(obs_raw)),
        median=float(np.nanmedian(obs_raw)),
        n=int(M),
    )
    obs_summary_resid = dict(
        mean=float(np.nanmean(obs_resid)),
        std=float(np.nanstd(obs_resid)),
        median=float(np.nanmedian(obs_resid)),
        n=int(M),
    )
    # 6) smoothness diagnostics
    edges_local = _wlc_edges_local_from_components(
        comps_global, global_to_local, n_bins_x, n_bins_y
    )
    target_obs = B_base if shuffle_session == "test" else A_base
    obs_energy = _wlc_laplacian_energy_cols(target_obs, edges_local)
    # 7) null loop (optionally smoothness-matched)
    target_repeats = int(n_repeats)
    max_attempts = target_repeats * int(max_attempts_multiplier)
    if _wlc_log is not None:
        _wlc_log(
            f"start: target_repeats={target_repeats}, max_attempts={max_attempts}, "
            f"scope={transform_scope}, shuffle_session={shuffle_session}, "
            f"smooth_match={bool(smoothness_match)}, smooth_filter={bool(smoothness_filter)}"
        )
    null_raw_rows = []
    null_resid_rows = []
    null_mean_raw_list = []
    null_mean_resid_list = []
    null_ratio_list = []
    null_filter_iters_list = []
    n_attempts = 0
    while len(null_mean_raw_list) < target_repeats and n_attempts < max_attempts:
        n_attempts += 1
        if transform_scope == "global":
            perm = np.arange(Kk, dtype=int)
            for ord_local in component_orders_local:
                mapped = _wlc_sample_component_mapping(
                    ord_local, rng=rng, reverse_prob=reverse_prob
                )
                perm[ord_local] = mapped
            if shuffle_session == "test":
                A_use = A_base
                B_use = B_base[perm, :]
            else:
                A_use = A_base[perm, :]
                B_use = B_base
        else:
            # Per-neuron independent transforms
            if shuffle_session == "test":
                A_use = A_base
                B_use = B_base.copy()
                for i in range(M):
                    perm_i = np.arange(Kk, dtype=int)
                    for ord_local in component_orders_local:
                        mapped = _wlc_sample_component_mapping(
                            ord_local, rng=rng, reverse_prob=reverse_prob
                        )
                        perm_i[ord_local] = mapped
                    B_use[:, i] = B_base[perm_i, i]
            else:
                A_use = A_base.copy()
                for i in range(M):
                    perm_i = np.arange(Kk, dtype=int)
                    for ord_local in component_orders_local:
                        mapped = _wlc_sample_component_mapping(
                            ord_local, rng=rng, reverse_prob=reverse_prob
                        )
                        perm_i[ord_local] = mapped
                    A_use[:, i] = A_base[perm_i, i]
                B_use = B_base
        # Optional post-shuffle smoothing filter on the shuffled session only.
        n_filter_iters = 0
        if smoothness_filter:
            if shuffle_session == "test":
                if smoothness_filter_mode == "adaptive":
                    B_use, _, n_filter_iters = _wlc_smooth_to_target_ratio(
                        B_use,
                        edges_local,
                        obs_energy,
                        target=smoothness_target,
                        max_iters=int(smoothness_filter_max_iters),
                        alpha=float(smoothness_filter_alpha),
                    )
                else:
                    n_filter_iters = int(smoothness_filter_n_iters)
                    B_use = _wlc_component_graph_smooth_cols(
                        B_use,
                        edges_local,
                        alpha=float(smoothness_filter_alpha),
                        n_iters=n_filter_iters,
                    )
            else:
                if smoothness_filter_mode == "adaptive":
                    A_use, _, n_filter_iters = _wlc_smooth_to_target_ratio(
                        A_use,
                        edges_local,
                        obs_energy,
                        target=smoothness_target,
                        max_iters=int(smoothness_filter_max_iters),
                        alpha=float(smoothness_filter_alpha),
                    )
                else:
                    n_filter_iters = int(smoothness_filter_n_iters)
                    A_use = _wlc_component_graph_smooth_cols(
                        A_use,
                        edges_local,
                        alpha=float(smoothness_filter_alpha),
                        n_iters=n_filter_iters,
                    )
        if shuffle_session == "test":
            A_res_use = A_res_base
            B_res_use = _wlc_residualize_by_level(B_use, kept_levels)
            shuf_energy = _wlc_laplacian_energy_cols(B_use, edges_local)
        else:
            A_res_use = _wlc_residualize_by_level(A_use, kept_levels)
            B_res_use = B_res_base
            shuf_energy = _wlc_laplacian_energy_cols(A_use, edges_local)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = shuf_energy / obs_energy
        ratio_med = float(np.nanmedian(ratio))
        if smoothness_match:
            if (not np.isfinite(ratio_med)) or (
                abs(ratio_med - float(smoothness_target)) > float(smoothness_tol)
            ):
                if (_wlc_log is not None) and (n_attempts % int(progress_every) == 0):
                    _wlc_log(
                        f"attempt={n_attempts}, accepted={len(null_mean_raw_list)} "
                        f"(rejected by smoothness_match)"
                    )
                continue
        s_raw, _ = _wlc_similarity_per_neuron(
            A_use, B_use, metric=metric, normalization=normalization, eps=eps
        )
        s_res, _ = _wlc_similarity_per_neuron(
            A_res_use, B_res_use, metric=metric, normalization=normalization, eps=eps
        )
        null_raw_rows.append(s_raw)
        null_resid_rows.append(s_res)
        null_mean_raw_list.append(float(np.nanmean(s_raw)))
        null_mean_resid_list.append(float(np.nanmean(s_res)))
        null_ratio_list.append(ratio_med)
        null_filter_iters_list.append(int(n_filter_iters))
        if (_wlc_log is not None) and (
            (len(null_mean_raw_list) == 1)
            or (len(null_mean_raw_list) % int(progress_every) == 0)
            or (len(null_mean_raw_list) == target_repeats)
        ):
            _wlc_log(
                f"attempt={n_attempts}, accepted={len(null_mean_raw_list)}/{target_repeats}, "
                f"median_ratio={ratio_med:.3f}"
            )
    if len(null_mean_raw_list) == 0:
        null_raw = np.zeros((0, M), dtype=float)
        null_resid = np.zeros((0, M), dtype=float)
        null_mean_raw = np.array([], dtype=float)
        null_mean_resid = np.array([], dtype=float)
        null_ratio = np.array([], dtype=float)
        null_filter_iters = np.array([], dtype=int)
    else:
        null_raw = np.vstack(null_raw_rows)
        null_resid = np.vstack(null_resid_rows)
        null_mean_raw = np.asarray(null_mean_raw_list, dtype=float)
        null_mean_resid = np.asarray(null_mean_resid_list, dtype=float)
        null_ratio = np.asarray(null_ratio_list, dtype=float)
        null_filter_iters = np.asarray(null_filter_iters_list, dtype=int)
    # 8) p-values
    p_neuron_raw = _wlc_empirical_p_ge(obs_raw, null_raw)
    p_neuron_resid = _wlc_empirical_p_ge(obs_resid, null_resid)
    p_mean_raw = _wlc_empirical_p_ge(
        np.array([obs_summary["mean"]], dtype=float), null_mean_raw
    )[0]
    p_mean_resid = _wlc_empirical_p_ge(
        np.array([obs_summary_resid["mean"]], dtype=float), null_mean_resid
    )[0]
    out = dict(
        observed=obs,
        observed_similarity=obs_raw,
        observed_similarity_resid=obs_resid,
        observed_valid_counts=obs_valid,
        observed_valid_counts_resid=obs_resid_valid,
        observed_summary=obs_summary,
        observed_summary_resid=obs_summary_resid,
        null_mean_raw=null_mean_raw,
        null_mean_resid=null_mean_resid,
        p_mean_raw=float(p_mean_raw),
        p_mean_resid=float(p_mean_resid),
        p_neuron_raw=p_neuron_raw,
        p_neuron_resid=p_neuron_resid,
        components=components,
        component_space=dict(
            log10_n_states=float(log10_n_states), n_nontrivial=int(n_nontrivial)
        ),
        smoothness=dict(
            observed_energy=obs_energy,
            null_ratio=null_ratio,
            ratio_mean=float(np.nanmean(null_ratio)),
            ratio_std=float(np.nanstd(null_ratio)),
            match_enabled=bool(smoothness_match),
            target=float(smoothness_target),
            tol=float(smoothness_tol),
            filter_enabled=bool(smoothness_filter),
            filter_mode=str(smoothness_filter_mode),
            filter_alpha=float(smoothness_filter_alpha),
            filter_n_iters_fixed=int(smoothness_filter_n_iters),
            filter_max_iters=int(smoothness_filter_max_iters),
            filter_iters=null_filter_iters,
            filter_iters_mean=float(np.nanmean(null_filter_iters))
            if null_filter_iters.size > 0
            else np.nan,
            filter_iters_std=float(np.nanstd(null_filter_iters))
            if null_filter_iters.size > 0
            else np.nan,
            n_attempts=int(n_attempts),
            n_accepted=int(null_ratio.size),
            acceptance_rate=(
                float(null_ratio.size) / float(n_attempts) if n_attempts > 0 else np.nan
            ),
        ),
        dropped=obs.get("dropped", {}),
        ids=dict(train_ids=train_ids_used, test_ids=test_ids_used),
        kept_bins_global=kept_bins_global,
        kept_levels=kept_levels,
        majority_meta=majority_meta,
        shuffle_session=shuffle_session,
        transform_scope=transform_scope,
        n_repeats=int(n_repeats),
        metric=metric,
        normalization=normalization,
    )
    if return_null_matrix:
        out["null_raw_matrix"] = null_raw
        out["null_resid_matrix"] = null_resid
    if _wlc_log is not None:
        _wlc_log(
            f"done: accepted={null_ratio.size}/{target_repeats}, attempts={n_attempts}, "
            f"acceptance_rate={out['smoothness']['acceptance_rate']:.3f}"
        )
    return out


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def cross_session_level_shift_similarity(
    # data
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap=None,
    # how to compute tuning
    level_col="level",
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,  # kept for API parity; not used for levels
    # which matched neurons to use
    n_neurons=None,
    random_state=0,
    train_ids=None,  # NEW: explicit ZERO-BASED neuron ids for TRAIN
    test_ids=None,  # NEW: explicit ZERO-BASED neuron ids for TEST
    # normalization & masking
    normalization="auto",  # 'zscore' | 'l2' | 'auto' | None
    intersect_mask=True,  # use only levels present in BOTH sessions (recommended)
    min_levels=3,
    eps=1e-12,
):
    """
    Cross-session **level shift similarity**:
    For each matched neuron, compute the maximum circularly-shifted correlation
    between its TRAIN level-tuning vector and TEST level-tuning vector.
    Also returns the best lag (in level bins) per neuron.
    You can pass explicit ZERO-BASED `train_ids` and `test_ids`. If both are
    provided, they override `cimap`. If `n_neurons` is set and the lists are
    longer, a reproducible subsample is taken with `random_state`.
    """
    rng = np.random.RandomState(random_state)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    # 1) Match or accept explicit neuron IDs (ZERO-BASED)
    n_train = spike_counts_train.shape[1]
    n_test = spike_counts_test.shape[1]
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr.min() < 0 or tr.max() >= n_train:
            raise ValueError(
                "train_ids out of range for TRAIN session (expected zero-based)."
            )
        if te.min() < 0 or te.max() >= n_test:
            raise ValueError(
                "test_ids out of range for TEST session (expected zero-based)."
            )
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        train_ids, test_ids = _pick_matched_neurons_for_level(
            cimap, n_train, n_test, n_neurons=n_neurons, rng=rng
        )
    M = len(train_ids)
    if M == 0:
        raise ValueError("No matched neurons after selection.")
    # 2) Shared level index & labels per session
    labels_tr, labels_te, uniq_lvls = _build_level_labels(
        selected_df_train, selected_df_test, level_col
    )
    K = len(uniq_lvls)
    # 3) Build windowed counts (T x M)
    frames_tr = np.arange(len(positions_train))
    frames_te = np.arange(len(positions_test))
    Q_tr = _windowed_counts(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        frames_tr,
        time_window,
        train_ids,
    )
    Q_te = _windowed_counts(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        frames_te,
        time_window,
        test_ids,
    )
    dt_eff = dt * time_window
    # 4) Fit level maps: lam_* (K x M), pr_* (K,), valid_* (K,)
    lam_tr, pr_tr, valid_tr = _fit_level_map(Q_tr, labels_tr, K, dt_eff, tuner, eps)
    lam_te, pr_te, valid_te = _fit_level_map(Q_te, labels_te, K, dt_eff, tuner, eps)
    # 5) Choose mask of levels to compare
    mask = (valid_tr & valid_te) if intersect_mask else (valid_tr | valid_te)
    idx = np.where(mask)[0]
    if idx.size < min_levels:
        sim = np.full(M, np.nan, float)
        best_lag = np.full(M, np.nan, float)
        valid_counts = np.zeros(M, int)
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=M)
        meta_tr = dict(n_levels=K, dt_eff=dt_eff)
        meta_te = dict(n_levels=K, dt_eff=dt_eff)
        return dict(
            train_ids=train_ids,
            test_ids=test_ids,
            best_corr=sim,
            best_lag=best_lag,
            levels=uniq_lvls,
            valid_counts=valid_counts,
            summary=summary,
            meta_train=meta_tr,
            meta_test=meta_te,
        )
    # 6) Slice to common levels & normalize per neuron (column-wise)
    A = lam_tr[idx, :]  # (K', M)
    B = lam_te[idx, :]  # (K', M)
    A = _normalize_cols_level(A, normalization, eps)
    B = _normalize_cols_level(B, normalization, eps)
    # 7) Per-neuron best circular correlation + lag
    best_corr, best_lag = _best_circ_corr_and_lag(A, B, min_levels=min_levels, eps=eps)
    valid_counts = np.full(M, idx.size, int)
    summary = dict(
        mean=float(np.nanmean(np.abs(best_lag))),
        std=float(np.nanstd(np.abs(best_lag))),
        median=float(np.nanmedian(np.abs(best_lag))),
        n=int(M),
    )
    meta_tr = dict(n_levels=K, dt_eff=dt_eff)
    meta_te = dict(n_levels=K, dt_eff=dt_eff)
    return dict(
        train_ids=train_ids,
        test_ids=test_ids,
        best_corr=best_corr,
        best_lag=best_lag,
        levels=uniq_lvls,
        valid_counts=valid_counts,
        summary=summary,
        meta_train=meta_tr,
        meta_test=meta_te,
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity, shuffle]
def pf_shuffle_level_shift_similarity(
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap=None,
    level_col="level",
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    n_neurons=None,
    random_state=0,
    # NEW: explicit ZERO-BASED neuron selections
    train_ids=None,
    test_ids=None,
    normalization="auto",
    restrict_to_common=True,
    min_levels=3,
    n_repeats=100,
    shuffle_session="test",  # 'test' or 'train'
    shuffle_type="neuron_id",  # 'neuron_id' or 'map'
    eps=1e-12,
):
    """
    PF-shuffle for *level-shift similarity*.
    shuffle_type:
      - 'neuron_id' : permute neuron columns in the chosen session
                      (breaks trainâ†”test neuron identity; preserves each map).
      - 'map'       : permute rows (levels) within each neuron's map in the chosen session
                      (destroys level ordering/topography; preserves neuron identity).
    You can pass explicit ZERO-BASED `train_ids` and `test_ids`. If both are
    provided, they override `cimap`. If `n_neurons` is set and the lists are
    longer, a reproducible subsample is taken with `random_state`.
    """
    if shuffle_session not in ("test", "train"):
        raise ValueError("shuffle_session must be 'test' or 'train'.")
    if shuffle_type not in ("neuron_id", "map"):
        raise ValueError("shuffle_type must be 'neuron_id' or 'map'.")
    rng = np.random.RandomState(random_state)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    # 1) Resolve neuron IDs (explicit ZERO-BASED overrides cimap)
    Ntr = spike_counts_train.shape[1]
    Nte = spike_counts_test.shape[1]
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr.min() < 0 or tr.max() >= Ntr:
            raise ValueError(
                "train_ids out of range for TRAIN session (expected zero-based)."
            )
        if te.min() < 0 or te.max() >= Nte:
            raise ValueError(
                "test_ids out of range for TEST session (expected zero-based)."
            )
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        train_ids, test_ids = _pick_matched_neurons_for_level(
            cimap, Ntr, Nte, n_neurons, rng
        )
    M = len(train_ids)
    if M == 0:
        raise ValueError("No matched neurons for PF-shuffle.")
    # 2) Shared level labels/codebook
    labels_tr, labels_te, uniq_lvls = _build_level_labels(
        selected_df_train, selected_df_test, level_col
    )
    K = len(uniq_lvls)
    # 3) Build Q once for the chosen neuron sets
    frames_tr = np.arange(len(positions_train))
    frames_te = np.arange(len(positions_test))
    dt_eff = dt * time_window
    Q_tr = _windowed_counts(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        frames_tr,
        time_window,
        train_ids,
    )
    Q_te = _windowed_counts(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        frames_te,
        time_window,
        test_ids,
    )
    # 4) Fit level tuning maps once (lam_*: K x M), and valid masks per level
    lam_tr, pr_tr, valid_tr = _fit_level_map(Q_tr, labels_tr, K, dt_eff, tuner, eps)
    lam_te, pr_te, valid_te = _fit_level_map(Q_te, labels_te, K, dt_eff, tuner, eps)
    # 5) Decide which levels to compare (fixed over repeats)
    mask_levels = (valid_tr & valid_te) if restrict_to_common else (valid_tr | valid_te)
    idx = np.where(mask_levels)[0]
    Kp = idx.size
    if Kp < min_levels:
        return dict(
            best_corr_matrix=np.full((n_repeats, M), np.nan),
            best_lag_matrix=np.full((n_repeats, M), np.nan),
            mean=np.nan,
            std=np.nan,
            levels=uniq_lvls,
            meta=dict(
                n_levels=K,
                n_pairs=M,
                shuffle_session=shuffle_session,
                shuffle_type=shuffle_type,
                normalization=normalization,
            ),
            train_ids=train_ids,
            test_ids=test_ids,
        )
    # 6) Keep both full and sliced versions
    A_full = lam_tr  # (K, M)
    B_full = lam_te  # (K, M)
    A_base = A_full[idx, :]  # (K', M)
    B_base = B_full[idx, :]  # (K', M)
    best_corr_mat = np.full((n_repeats, M), np.nan, float)
    best_lag_mat = np.full((n_repeats, M), np.nan, float)
    for r in range(n_repeats):
        # 7) Choose shuffle
        if shuffle_type == "neuron_id":
            if shuffle_session == "test":
                perm = rng.permutation(M)
                A_use = A_base
                B_use = B_base[:, perm]
            else:  # train
                perm = rng.permutation(M)
                A_use = A_base[:, perm]
                B_use = B_base
        else:  # shuffle_type == 'map' â†’ permute level rows within each neuron's map on full K, then slice
            if shuffle_session == "test":
                B_shuf_full = B_full.copy()
                for m in range(M):
                    B_shuf_full[:, m] = B_full[rng.permutation(K), m]
                A_use = A_base
                B_use = B_shuf_full[idx, :]
            else:  # train
                A_shuf_full = A_full.copy()
                for m in range(M):
                    A_shuf_full[:, m] = A_full[rng.permutation(K), m]
                A_use = A_shuf_full[idx, :]
                B_use = B_base
        # 8) Normalize columns
        A_n = _normalize_cols_level(A_use, normalization, eps)  # (K', M)
        B_n = _normalize_cols_level(B_use, normalization, eps)  # (K', M)
        # 9) Best circular correlation & lag per neuron column
        bc, bl = _best_circ_corr_and_lag(A_n, B_n, min_levels=min_levels, eps=eps)
        best_corr_mat[r, :] = bc
        best_lag_mat[r, :] = bl
    return dict(
        best_corr_matrix=best_corr_mat,
        best_lag_matrix=best_lag_mat,
        mean=float(np.nanmean(np.abs(best_lag_mat))),  # summary of |lag|
        std=float(np.nanstd(np.abs(best_lag_mat))),
        levels=uniq_lvls,
        meta=dict(
            n_levels=K,
            n_pairs=M,
            shuffle_session=shuffle_session,
            shuffle_type=shuffle_type,
            normalization=normalization,
        ),
        train_ids=train_ids,
        test_ids=test_ids,
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr, correlation]
def cross_session_pv_correlation(
    # data
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    # what to compare
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    # how to compute tuning
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    # which matched neurons to use
    n_neurons=None,
    random_state=0,
    train_ids=None,  # explicit ZERO-BASED train ids
    test_ids=None,  # explicit ZERO-BASED test ids
    # NEW: explicit exclusions
    exclude_train_ids=None,  # iterable of ZERO-BASED train ids to exclude
    exclude_test_ids=None,  # iterable of ZERO-BASED test ids to exclude
    exclude_pairs=None,  # iterable of (train_id, test_id) pairs to exclude
    # similarity across neurons
    metric="pearson",  # 'pearson' | 'spearman' | 'cosine'
    normalization="auto",  # 'none' | 'zscore' | 'l2' | 'auto'
    intersect_mask=True,  # use only bins/states valid in BOTH sessions
    min_neurons=3,  # minimum neurons needed to compute a PV correlation
    eps=1e-12,
    # firing-rate based masking (spikes per minute)
    min_spikes_per_min=None,  # float or None -> apply to BOTH sessions if set
    min_spikes_per_min_train=None,  # optional override for TRAIN
    min_spikes_per_min_test=None,  # optional override for TEST
    rate_source="spike",  # 'spike' (spike_counts_*) | 'raw' (raw_counts_*)
    drop_if_below="either",  # 'either' (drop if train<thr OR test<thr) | 'both' (keep if >=1 passes)
):
    """
    Compute PV correlation per bin/state across TRAIN vs TEST.
    1D vs 2D positions:
      - 2D: positions_* shape (T, 2), n_bins_x Ã— n_bins_y grid.
      - 1D: positions_* shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and n_bins_x = #bins.
    """
    rng = np.random.RandomState(random_state)
    # --- Normalize positions & detect 1D vs 2D ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # --- Helper: choose counts for rate computation ---
    if rate_source == "spike":
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == "raw":
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, dtype=float).sum(axis=0)  # (N,)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        if total_minutes <= 0:
            return np.zeros(C.shape[1], dtype=float)
        return total_events / total_minutes
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    # 1) Resolve matched neurons (explicit ZERO-BASED overrides cimap)
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError(
                "train_ids out of range for TRAIN session (expected zero-based)."
            )
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError(
                "test_ids out of range for TEST session (expected zero-based)."
            )
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        train_ids, test_ids = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
    if len(train_ids) == 0:
        raise ValueError("No matched neurons selected.")
    # --- 1a) Apply explicit user exclusions on matched PAIRS ---
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    n_pairs_initial = int(train_ids.size)
    keep_user = np.ones(train_ids.size, dtype=bool)
    if exclude_train_ids is not None:
        ex_tr = _to_int_array(exclude_train_ids)
        if ex_tr.size:
            keep_user &= ~np.isin(train_ids, ex_tr)
    if exclude_test_ids is not None:
        ex_te = _to_int_array(exclude_test_ids)
        if ex_te.size:
            keep_user &= ~np.isin(test_ids, ex_te)
    if exclude_pairs is not None:
        ex_pairs = np.asarray(exclude_pairs, dtype=int)
        if not (ex_pairs.ndim == 1 and ex_pairs.size == 0):
            if ex_pairs.ndim != 2 or ex_pairs.shape[1] != 2:
                raise ValueError(
                    "exclude_pairs must be an array-like of (train_id, test_id) pairs."
                )
            ex_set = set(map(tuple, ex_pairs.tolist()))
            pair_mask = np.array(
                [(tr, te) not in ex_set for tr, te in zip(train_ids, test_ids)],
                dtype=bool,
            )
            keep_user &= pair_mask
    # Filter after exclusions
    train_ids = train_ids[keep_user]
    test_ids = test_ids[keep_user]
    n_dropped_by_user = int((~keep_user).sum())
    if train_ids.size == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n_bins=0)
        return dict(
            pvc=np.array([], dtype=float),
            mask=None,
            maps_tr=None,
            maps_te=None,
            metric=metric,
            mode=mode,
            normalization=normalization,
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            valid_counts=None,
            summary=summary,
            meta_train=dict(
                rate_per_min=rate_tr,
                threshold=min_spikes_per_min_train
                if min_spikes_per_min_train is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            meta_test=dict(
                rate_per_min=rate_te,
                threshold=min_spikes_per_min_test
                if min_spikes_per_min_test is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            dropped=dict(
                n_pairs_initial=n_pairs_initial,
                n_pairs_dropped_by_user=n_dropped_by_user,
                n_pairs_dropped_by_rate=0,
                n_pairs_final=0,
                policy=drop_if_below,
            ),
        )
    # --- 1b) Apply firing-rate based masking on matched PAIRS (after user exclusions) ---
    thr_tr = (
        min_spikes_per_min_train
        if min_spikes_per_min_train is not None
        else min_spikes_per_min
    )
    thr_te = (
        min_spikes_per_min_test
        if min_spikes_per_min_test is not None
        else min_spikes_per_min
    )
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == "either":
        keep_rate = pass_tr & pass_te
    elif drop_if_below == "both":
        keep_rate = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    n_dropped_by_rate = int((~keep_rate).sum())
    train_ids = np.asarray(train_ids)[keep_rate]
    test_ids = np.asarray(test_ids)[keep_rate]
    # If nothing left after masking, return empty PV
    if train_ids.size == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n_bins=0)
        return dict(
            pvc=np.array([], dtype=float),
            mask=None,
            maps_tr=None,
            maps_te=None,
            metric=metric,
            mode=mode,
            normalization=normalization,
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            valid_counts=None,
            summary=summary,
            meta_train=dict(
                rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
            ),
            meta_test=dict(
                rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
            ),
            dropped=dict(
                n_pairs_initial=n_pairs_initial,
                n_pairs_dropped_by_user=n_dropped_by_user,
                n_pairs_dropped_by_rate=n_dropped_by_rate,
                n_pairs_final=0,
                policy=drop_if_below,
            ),
        )
    # 2) Fit maps once (train/test)
    maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te = _prefit_train_and_test_maps(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        train_ids,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    # --- Normalization policy ---
    if normalization == "auto":
        norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        norm_mode = normalization
    # 3) PV per bin/state
    pvc, mask = _pv_from_maps(
        maps_tr,
        maps_te,
        mask_tr,
        mask_te,
        metric=metric,
        normalization=norm_mode,  # <-- use resolved mode
        intersect_mask=intersect_mask,
        min_neurons=min_neurons,
        eps=eps,
    )
    # 4) Summary
    summary = dict(
        mean=float(np.nanmean(pvc)) if pvc.size else np.nan,
        std=float(np.nanstd(pvc)) if pvc.size else np.nan,
        median=float(np.nanmedian(pvc)) if pvc.size else np.nan,
        n_bins=int(np.isfinite(pvc).sum()) if pvc.size else 0,
    )
    # Annotate meta with thresholds + per-neuron rates used for masking
    meta_tr = dict(meta_tr)
    meta_te = dict(meta_te)
    meta_tr["rate_per_min"] = rate_tr
    meta_tr["threshold"] = thr_tr
    meta_tr["rate_source"] = rate_source
    meta_te["rate_per_min"] = rate_te
    meta_te["threshold"] = thr_te
    meta_te["rate_source"] = rate_source
    return dict(
        pvc=pvc,
        mask=mask,
        maps_tr=maps_tr,
        maps_te=maps_te,
        metric=metric,
        mode=mode,
        normalization=norm_mode,
        train_ids=train_ids,
        test_ids=test_ids,
        valid_counts=None,
        summary=summary,
        meta_train=meta_tr,
        meta_test=meta_te,
        dropped=dict(
            n_pairs_initial=n_pairs_initial,
            n_pairs_dropped_by_user=n_dropped_by_user,
            n_pairs_dropped_by_rate=n_dropped_by_rate,
            n_pairs_final=int(train_ids.size),
            policy=drop_if_below,
        ),
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity, shuffle]
def pf_shuffle_pv_similarity(
    # same inputs/signature style as cross_session_pv_correlation
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    mode="spatial",
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    eps=1e-12,
    n_neurons=None,
    random_state=0,
    train_ids=None,  # explicit ZERO-BASED train ids
    test_ids=None,  # explicit ZERO-BASED test ids
    # NEW: explicit exclusions
    exclude_train_ids=None,  # iterable of ZERO-BASED train ids to exclude
    exclude_test_ids=None,  # iterable of ZERO-BASED test ids to exclude
    exclude_pairs=None,  # iterable of (train_id, test_id) pairs to exclude
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    min_neurons=3,
    repeats=1000,
    rng=None,
    return_observed=True,
    # Shuffling controls
    shuffle_type="neuron_id",  # 'neuron_id' | 'map'
    shuffle_session="test",  # 'test' | 'train'
    # firing-rate based masking (spikes per minute)
    min_spikes_per_min=None,  # float or None -> apply to BOTH sessions if set
    min_spikes_per_min_train=None,  # optional override for TRAIN
    min_spikes_per_min_test=None,  # optional override for TEST
    rate_source="spike",  # 'spike' (spike_counts_*) | 'raw' (raw_counts_*)
    drop_if_below="either",  # 'either' drop if EITHER day below thr | 'both' keep if >=1 passes
):
    """
    Build a null for PV correlation by either:
      - shuffle_type='neuron_id': permute neuron columns on one session (breaks trainâ†”test ID)
      - shuffle_type='map'      : permute rows (bins/states) within each neuron's map for one session
    1D vs 2D positions:
      - 2D: positions_* shape (T, 2), n_bins_x Ã— n_bins_y grid.
      - 1D: positions_* shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and n_bins_x = #bins.
    NEW:
      - exclude_train_ids / exclude_test_ids / exclude_pairs to drop pairs explicitly
        before rate-based masking.
    """
    # RNG
    if rng is None:
        rng = np.random.RandomState(random_state)
    # --- Normalize positions & detect 1D vs 2D ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    # Validate shuffle args
    if shuffle_type not in ("neuron_id", "map"):
        raise ValueError("shuffle_type must be 'neuron_id' or 'map'")
    if shuffle_session not in ("test", "train"):
        raise ValueError("shuffle_session must be 'test' or 'train'")
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # --- Choose counts for rate computation ---
    if rate_source == "spike":
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == "raw":
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, dtype=float).sum(axis=0)  # (N,)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        if total_minutes <= 0:
            return np.zeros(C.shape[1], dtype=float)
        return total_events / total_minutes
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    # --- Resolve matched neurons (explicit overrides cimap) ---
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError(
                "train_ids out of range for TRAIN session (expected zero-based)."
            )
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError(
                "test_ids out of range for TEST session (expected zero-based)."
            )
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        train_ids, test_ids = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
    if len(train_ids) == 0:
        raise ValueError("No matched neurons selected.")
    # --- Apply explicit user exclusions on matched PAIRS ---
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    n_pairs_initial = int(train_ids.size)
    keep_user = np.ones(train_ids.size, dtype=bool)
    if exclude_train_ids is not None:
        ex_tr = _to_int_array(exclude_train_ids)
        if ex_tr.size:
            keep_user &= ~np.isin(train_ids, ex_tr)
    if exclude_test_ids is not None:
        ex_te = _to_int_array(exclude_test_ids)
        if ex_te.size:
            keep_user &= ~np.isin(test_ids, ex_te)
    if exclude_pairs is not None:
        ex_pairs = np.asarray(exclude_pairs, dtype=int)
        if not (ex_pairs.ndim == 1 and ex_pairs.size == 0):
            if ex_pairs.ndim != 2 or ex_pairs.shape[1] != 2:
                raise ValueError(
                    "exclude_pairs must be array-like of (train_id, test_id) pairs."
                )
            ex_set = set(map(tuple, ex_pairs.tolist()))
            pair_mask = np.array(
                [(tr, te) not in ex_set for tr, te in zip(train_ids, test_ids)],
                dtype=bool,
            )
            keep_user &= pair_mask
    train_ids = train_ids[keep_user]
    test_ids = test_ids[keep_user]
    n_dropped_by_user = int((~keep_user).sum())
    # If nothing left after exclusions
    if train_ids.size == 0:
        out = dict(
            pv_shuffles=np.empty((0, 0), float),
            pv_summary=dict(mean=np.nan, std=np.nan, median=np.nan),
            metric=metric,
            mode=mode,
            normalization=(
                "zscore"
                if metric in ("pearson", "spearman") and normalization == "auto"
                else (
                    "l2"
                    if metric == "cosine" and normalization == "auto"
                    else normalization
                )
            ),
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            meta_train=dict(
                rate_per_min=rate_tr,
                threshold=min_spikes_per_min_train
                if min_spikes_per_min_train is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            meta_test=dict(
                rate_per_min=rate_te,
                threshold=min_spikes_per_min_test
                if min_spikes_per_min_test is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            shuffle_type=shuffle_type,
            shuffle_session=shuffle_session,
            repeats=int(repeats),
            dropped=dict(
                n_pairs_initial=n_pairs_initial,
                n_pairs_dropped_by_user=n_dropped_by_user,
                n_pairs_dropped_by_rate=0,
                n_pairs_final=0,
                policy=drop_if_below,
            ),
        )
        if return_observed:
            out.update(
                dict(pv_obs=np.array([], float), mask=None, maps_tr=None, maps_te=None)
            )
        return out
    # --- Firing-rate based masking on matched PAIRS (after user exclusions) ---
    thr_tr = (
        min_spikes_per_min_train
        if min_spikes_per_min_train is not None
        else min_spikes_per_min
    )
    thr_te = (
        min_spikes_per_min_test
        if min_spikes_per_min_test is not None
        else min_spikes_per_min
    )
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == "either":
        keep_mask = pass_tr & pass_te
    elif drop_if_below == "both":
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    n_dropped_by_rate = int((~keep_mask).sum())
    train_ids = np.asarray(train_ids)[keep_mask]
    test_ids = np.asarray(test_ids)[keep_mask]
    # If nothing left after masking -> return empty/null structure
    if train_ids.size == 0:
        out = dict(
            pv_shuffles=np.empty((0, 0), float),
            pv_summary=dict(mean=np.nan, std=np.nan, median=np.nan),
            metric=metric,
            mode=mode,
            normalization=(
                "zscore"
                if metric in ("pearson", "spearman") and normalization == "auto"
                else (
                    "l2"
                    if metric == "cosine" and normalization == "auto"
                    else normalization
                )
            ),
            train_ids=np.array([], dtype=int),
            test_ids=np.array([], dtype=int),
            meta_train=dict(
                rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
            ),
            meta_test=dict(
                rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
            ),
            shuffle_type=shuffle_type,
            shuffle_session=shuffle_session,
            repeats=int(repeats),
            dropped=dict(
                n_pairs_initial=n_pairs_initial,
                n_pairs_dropped_by_user=n_dropped_by_user,
                n_pairs_dropped_by_rate=n_dropped_by_rate,
                n_pairs_final=0,
                policy=drop_if_below,
            ),
        )
        if return_observed:
            out.update(
                dict(pv_obs=np.array([], float), mask=None, maps_tr=None, maps_te=None)
            )
        return out
    # --- Fit maps on train/test (once) ---
    maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te = _prefit_train_and_test_maps(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        train_ids,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    # --- Compute observed PV ---
    if normalization == "auto":
        norm_mode = "zscore" if metric in ("pearson", "spearman") else "l2"
    else:
        norm_mode = normalization
    pv_obs, mask = _pv_from_maps(
        maps_tr,
        maps_te,
        mask_tr,
        mask_te,
        metric=metric,
        normalization=norm_mode,
        intersect_mask=intersect_mask,
        min_neurons=min_neurons,
        eps=eps,
    )
    # --- Shuffles ---
    pv_shuffles = []
    K = maps_tr.shape[0]
    for _ in range(repeats):
        if shuffle_type == "neuron_id":
            if shuffle_session == "test":
                perm = rng.permutation(maps_te.shape[1])
                maps_te_shuf = maps_te[:, perm]
                maps_tr_use = maps_tr
                maps_te_use = maps_te_shuf
            else:
                perm = rng.permutation(maps_tr.shape[1])
                maps_tr_shuf = maps_tr[:, perm]
                maps_tr_use = maps_tr_shuf
                maps_te_use = maps_te
        else:  # 'map'
            if shuffle_session == "test":
                maps_te_shuf = maps_te.copy()
                for n in range(maps_te.shape[1]):
                    maps_te_shuf[:, n] = maps_te[rng.permutation(K), n]
                maps_tr_use = maps_tr
                maps_te_use = maps_te_shuf
            else:
                maps_tr_shuf = maps_tr.copy()
                for n in range(maps_tr.shape[1]):
                    maps_tr_shuf[:, n] = maps_tr[rng.permutation(K), n]
                maps_tr_use = maps_tr_shuf
                maps_te_use = maps_te
        pv_shuf, _ = _pv_from_maps(
            maps_tr_use,
            maps_te_use,
            mask_tr,
            mask_te,
            metric=metric,
            normalization=norm_mode,
            intersect_mask=intersect_mask,
            min_neurons=min_neurons,
            eps=eps,
        )
        pv_shuffles.append(pv_shuf)
    pv_shuffles = np.vstack(pv_shuffles) if pv_shuffles else np.empty((0, 0), float)
    pv_summary = dict(
        mean=float(np.nanmean(pv_shuffles)) if pv_shuffles.size else np.nan,
        std=float(np.nanstd(pv_shuffles)) if pv_shuffles.size else np.nan,
        median=float(np.nanmedian(pv_shuffles)) if pv_shuffles.size else np.nan,
    )
    out = dict(
        pv_shuffles=pv_shuffles,
        pv_summary=pv_summary,
        metric=metric,
        mode=mode,
        normalization=norm_mode,
        train_ids=train_ids,
        test_ids=test_ids,
        meta_train=dict(
            rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
        ),
        meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source),
        shuffle_type=shuffle_type,
        shuffle_session=shuffle_session,
        repeats=int(repeats),
        dropped=dict(
            n_pairs_initial=n_pairs_initial,
            n_pairs_dropped_by_user=n_dropped_by_user,
            n_pairs_dropped_by_rate=n_dropped_by_rate,
            n_pairs_final=int(train_ids.size),
            policy=drop_if_below,
        ),
    )
    if return_observed:
        out.update(dict(pv_obs=pv_obs, mask=mask, maps_tr=maps_tr, maps_te=maps_te))
    return out


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def cross_session_pv_cross_similarity_matrix(
    # data
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    # what to compare
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    # how to compute tuning
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    # which matched neurons to use
    n_neurons=None,
    random_state=0,
    train_ids=None,  # explicit ZERO-BASED ids for TRAIN
    test_ids=None,  # explicit ZERO-BASED ids for TEST
    # similarity across neurons (row-wise)
    metric="pearson",  # 'pearson' | 'spearman' | 'cosine'
    normalization="auto",  # 'none' | 'zscore' | 'l2' | 'auto'  (row-wise for PV)
    intersect_mask=True,  # compare only bins/states valid in BOTH sessions when K matches
    eps=1e-12,
    # QC
    min_neurons=3,  # minimum finite neurons required in a row to keep it
    drop_rows_with_too_few_neurons=True,
):
    """
    Compute cross-session, binÃ—bin PV similarity:
      matrix[b_train, b_test] = similarity( PV_train[b_train, :], PV_test[b_test, :] )
    1D vs 2D positions
    ------------------
    - 2D: positions_* shape (T, 2), grid n_bins_x Ã— n_bins_y.
    - 1D: positions_* shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and
          n_bins_x = number of spatial bins.
    """
    try:
        from scipy.stats import rankdata as _rankdata
    except Exception:
        def _rankdata(x, method="average"):
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(order), dtype=float)
            return ranks
    # --- 0) Position dimensionality handling ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    rng = np.random.RandomState(random_state)
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # ---------- 1) Resolve matched neurons ----------
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError("train_ids and test_ids must have the same length.")
        if tr.size == 0:
            raise ValueError("Empty train_ids/test_ids.")
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError(
                "train_ids out of range for TRAIN session (zero-based expected)."
            )
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError(
                "test_ids out of range for TEST session (zero-based expected)."
            )
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        train_ids, test_ids = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
    M = len(train_ids)
    if M == 0:
        raise ValueError("No matched neurons after selection.")
    # ---------- 2) Fit PV maps once per session (K Ã— M) ----------
    maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te = _prefit_train_and_test_maps(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        train_ids,
        test_ids,
        mode=mode,
        level_col=level_col,
        head_col=head_col,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        eps=eps,
    )
    K_tr, M_tr = maps_tr.shape
    K_te, M_te = maps_te.shape
    if M_tr != M_te:
        raise ValueError(
            f"Train/Test neuron counts disagree after selection: {M_tr} vs {M_te}"
        )
    # ---------- 3) Choose rows (bins/states) to compare ----------
    if intersect_mask and (K_tr == K_te):
        common_idx = np.where(mask_tr.astype(bool) & mask_te.astype(bool))[0]
        rows_tr = common_idx
        rows_te = common_idx
    else:
        rows_tr = np.where(mask_tr)[0]
        rows_te = np.where(mask_te)[0]
    A = maps_tr[rows_tr, :]  # (K_tr_used, M)
    B = maps_te[rows_te, :]  # (K_te_used, M)
    # ---------- 4) Row QC + neuron-wise imputation ----------
    good_tr = np.isfinite(A)
    good_te = np.isfinite(B)
    min_good = max(1, int(min_neurons))
    keep_tr = good_tr.sum(axis=1) >= min_good
    keep_te = good_te.sum(axis=1) >= min_good
    if drop_rows_with_too_few_neurons:
        A = A[keep_tr, :]
        B = B[keep_te, :]
        rows_tr = rows_tr[keep_tr]
        rows_te = rows_te[keep_te]
        good_tr = good_tr[keep_tr, :]
        good_te = good_te[keep_te, :]
    A = _impute_neuronwise(A, good_tr)
    B = _impute_neuronwise(B, good_te)
    # ---------- 5) Row-wise normalization + similarity ----------
    if normalization == "auto":
        row_norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        row_norm_mode = normalization
    if metric == "spearman":
        # rank per row, then z-score each row
        A = np.apply_along_axis(_rankdata, 1, A)
        B = np.apply_along_axis(_rankdata, 1, B)
        A = np.apply_along_axis(_norm_row, 1, A, mode="zscore", eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode="zscore", eps=eps)
        denom = max(M - 1, 1) + eps
        sim = (A @ B.T) / denom
    elif metric == "pearson":
        A = np.apply_along_axis(_norm_row, 1, A, mode="zscore", eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode="zscore", eps=eps)
        denom = max(M - 1, 1) + eps
        sim = (A @ B.T) / denom
    else:  # 'cosine'
        A = np.apply_along_axis(_norm_row, 1, A, mode="l2", eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode="l2", eps=eps)
        sim = A @ B.T
    return dict(
        matrix=sim,
        rows_train_idx=rows_tr,
        rows_test_idx=rows_te,
        metric=metric,
        mode=mode,
        normalization=row_norm_mode,
        train_ids=train_ids,
        test_ids=test_ids,
        meta_train=meta_tr,
        meta_test=meta_te,
    )


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _skaggs_info(
    lambdas_spatial: np.ndarray, prior_spatial: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """
    Skaggs spatial information per neuron, in bits/spike.
    lambdas_spatial: (K, N) rate map (per-bin firing rates)
    prior_spatial:   (K,)   occupancy probability over K bins (sums to 1)
    """
    L = np.asarray(lambdas_spatial, float) + eps
    p = np.asarray(prior_spatial, float)
    p = p / max(p.sum(), eps)
    r_bar = (p[:, None] * L).sum(axis=0) + eps  # (N,)
    ratio = L / r_bar[None, :]
    # sum_b p_b * (r_b/r_bar) * log2(r_b/r_bar)
    I = (p[:, None] * ratio * np.log2(ratio)).sum(axis=0)
    # Numerical hygiene
    I[~np.isfinite(I)] = 0.0
    return I


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _spikes_per_min(C, dt_sec):
    """events/min computed over the provided rows."""
    C = np.asarray(C, float)
    total_events = C.sum(axis=0)  # (N,)
    total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
    if total_minutes <= 0:
        return np.zeros(C.shape[1], float)
    return total_events / total_minutes


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_spatial_info_session_rategated(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    *,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=5.0,
    eps=1e-12,
    neuron_ids=None,
    # rate gate (same semantics you use elsewhere)
    min_spikes_per_min=None,  # gate by activity if set
    rate_source="spike",  # 'spike' | 'raw'
):
    """
    Returns:
      dict with keys:
        SI_native: (N_kept,)
        prior:     (K,)
        rates:     (K, N_kept)
        mask:      (K,) bins used (prior>0 after min_time gate)
        kept_neuron_ids: (N_kept,)
        meta: grid info
    1D vs 2D positions
    ------------------
    - 2D: positions shape (T, 2), grid n_bins_x Ã— n_bins_y.
    - 1D: positions shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and
          n_bins_x = number of spatial bins.
    """
    # --- 0) Position dimensionality handling ---
    pos = np.asarray(positions)
    is_1d = (pos.ndim == 1) or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    # 1) Discretize position grid (uses your 1D/2D-aware discretize_positions)
    labels_pos, x_edges, y_edges = discretize_positions(pos, n_bins_x, n_bins_y)
    n_bins = int(n_bins_x * n_bins_y)  # for 1D, n_bins_y should be 1
    # 2) Choose neuron set
    if neuron_ids is None:
        neuron_ids = np.arange(spike_counts.shape[1], dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    # 3) Optional rate gating (events/min over the selected rows)
    C_for_rate = spike_counts if rate_source == "spike" else raw_counts
    dt_eff_rate = float(dt)  # same policy as your PV code
    rates_pm_all = _spikes_per_min(C_for_rate, dt_eff_rate)
    if min_spikes_per_min is not None:
        keep = rates_pm_all[neuron_ids] >= float(min_spikes_per_min)
        neuron_ids = neuron_ids[keep]
    if neuron_ids.size == 0:
        return dict(
            SI_native=np.array([], float),
            prior=np.zeros(n_bins, float),
            rates=np.zeros((n_bins, 0), float),
            mask=np.zeros(n_bins, bool),
            kept_neuron_ids=neuron_ids,
            meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
        )
    # 4) Build counts for the kept neurons ONLY
    frames = np.arange(len(pos))
    Q = build_count_matrix(
        spike_counts,
        raw_counts,
        selected_df,
        frames=frames,
        time_window=time_window,
        neuron_ids=neuron_ids,
    ).T  # (T_sel, N_kept)
    # Effective dt for tuning inputs matches your other functions
    dt_eff_tuning = dt * (time_window + 1)
    # 5) Rate map + occupancy prior
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    lam_sp, prior_sp = tuner(
        Q,
        labels_pos,
        None,
        n_bins,
        dt_eff_tuning,
        filter_ratemap,
        sigma,
        min_time=min_time_per_bin_sec,
        eps=eps,
        decode_head="none",
    )
    # 6) SI
    SI_native = _skaggs_info(lam_sp, prior_sp, eps=eps)
    mask = prior_sp > 0
    return dict(
        SI_native=SI_native,
        prior=prior_sp,
        rates=lam_sp,
        mask=mask,
        kept_neuron_ids=neuron_ids,
        meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
    )


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_spatial_info_matched(
    sp1, sp2, *, control: str = "intersection_avg", eps: float = 1e-12
):
    """
    sp1, sp2 are dicts returned by compute_spatial_info_session_rategated (same N if you matched pairs).
    control:
      - 'intersection_avg'      : shared prior = average s1+s2 over intersection
      - 'intersection_uniform'  : shared prior = uniform over intersection
      - 'intersection_s1'       : shared prior = s1 prior over intersection
      - 'intersection_s2'       : shared prior = s2 prior over intersection
      - 'intersection_native'   : NO shared prior; evaluate each session with its own prior (restricted to intersection)
    """
    p1 = sp1["prior"].astype(float)
    p2 = sp2["prior"].astype(float)
    L1 = sp1["rates"].astype(float)  # (K, N1)
    L2 = sp2["rates"].astype(float)  # (K, N2)
    inter = (p1 > 0) & (p2 > 0)
    if not np.any(inter):
        K = p1.shape[0]
        z1 = np.zeros(L1.shape[1], float) if L1.ndim == 2 else np.array([], float)
        z2 = np.zeros(L2.shape[1], float) if L2.ndim == 2 else np.array([], float)
        return dict(
            mask_intersection=np.zeros(K, bool), SI_matched_s1=z1, SI_matched_s2=z2
        )
    p1i = np.where(inter, p1, 0.0)
    p2i = np.where(inter, p2, 0.0)
    if control == "intersection_avg":
        p_common = (p1i + p2i) / max(p1i.sum() + p2i.sum(), eps)
        p_common = p_common / max(p_common.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(
            mask_intersection=inter,
            prior_common=p_common,
            SI_matched_s1=SI1,
            SI_matched_s2=SI2,
        )
    if control == "intersection_uniform":
        k = inter.sum()
        p_common = np.where(inter, 1.0 / k, 0.0)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(
            mask_intersection=inter,
            prior_common=p_common,
            SI_matched_s1=SI1,
            SI_matched_s2=SI2,
        )
    if control == "intersection_s1":
        p_common = p1i / max(p1i.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(
            mask_intersection=inter,
            prior_common=p_common,
            SI_matched_s1=SI1,
            SI_matched_s2=SI2,
        )
    if control == "intersection_s2":
        p_common = p2i / max(p2i.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(
            mask_intersection=inter,
            prior_common=p_common,
            SI_matched_s1=SI1,
            SI_matched_s2=SI2,
        )
    if control in (
        "intersection_native",
        "intersection_own",
        "intersection_per_session",
    ):
        p1n = p1i / max(p1i.sum(), eps)
        p2n = p2i / max(p2i.sum(), eps)
        SI1 = _skaggs_info(L1, p1n, eps=eps)
        SI2 = _skaggs_info(L2, p2n, eps=eps)
        return dict(
            mask_intersection=inter,
            prior_s1_intersection=p1n,
            prior_s2_intersection=p2n,
            SI_matched_s1=SI1,
            SI_matched_s2=SI2,
        )
    raise ValueError(f"Unknown control='{control}'")


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: spatial]
def cross_session_spatial_information_rategated(
    # data
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    # tuning
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=5.0,
    eps=1e-12,
    # matching / overrides
    n_neurons=None,
    random_state=0,
    train_ids=None,
    test_ids=None,
    # RATE GATE â€” identical semantics to your cross_session_pv_correlation
    min_spikes_per_min=None,
    min_spikes_per_min_train=None,
    min_spikes_per_min_test=None,
    rate_source="spike",
    drop_if_below="either",  # 'either' â†’ keep only if BOTH pass; 'both' â†’ keep if EITHER passes
    # SI matching control
    si_control="intersection_avg",
):
    """
    Returns dict with:
      kept train/test ids, per-neuron SI_native (s1,s2), matched SI (s1,s2),
      intersection mask & priors, and small summary.
    1D vs 2D positions
    ------------------
    - 2D: positions_* shape (T, 2), grid n_bins_x Ã— n_bins_y.
    - 1D: positions_* shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and
          n_bins_x = number of spatial bins.
    """
    # --- 0) Position dimensionality handling ---
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = (pos_tr.ndim == 1) or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = (pos_te.ndim == 1) or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            "TRAIN and TEST positions must have the same dimensionality "
            f"(got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te})."
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError(
            "For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 "
            "and n_bins_x = desired number of spatial bins."
        )
    rng = np.random.RandomState(random_state)
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    # 0) Rate arrays (events/min) on chosen source, dt same policy as PV fn
    Ctr_for_rate = spike_counts_train if rate_source == "spike" else raw_counts_train
    Cte_for_rate = spike_counts_test if rate_source == "spike" else raw_counts_test
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    # 1) Resolve matched pairs (explicit overrides cimap)
    if train_ids is not None and test_ids is not None:
        tr = np.asarray(train_ids, int)
        te = np.asarray(test_ids, int)
        if tr.size == 0 or te.size == 0 or tr.size != te.size:
            raise ValueError("train_ids and test_ids must be same-length, non-empty.")
        if tr.min() < 0 or tr.max() >= n_tr or te.min() < 0 or te.max() >= n_te:
            raise ValueError("train_ids/test_ids out of range.")
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr, te = tr[sel], te[sel]
        train_ids, test_ids = tr, te
    else:
        # relies on your existing helper
        tr, te = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
        train_ids, test_ids = np.asarray(tr, int), np.asarray(te, int)
    if train_ids.size == 0:
        return dict(
            train_ids=np.array([], int),
            test_ids=np.array([], int),
            SI_native_s1=np.array([], float),
            SI_native_s2=np.array([], float),
            SI_matched_s1=np.array([], float),
            SI_matched_s2=np.array([], float),
            mask_intersection=None,
            prior_common=None,
            meta_train=dict(
                rate_per_min=rate_tr,
                threshold=min_spikes_per_min_train
                if min_spikes_per_min_train is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            meta_test=dict(
                rate_per_min=rate_te,
                threshold=min_spikes_per_min_test
                if min_spikes_per_min_test is not None
                else min_spikes_per_min,
                rate_source=rate_source,
            ),
            summary=dict(
                median_native_s1=np.nan,
                median_native_s2=np.nan,
                median_matched_s1=np.nan,
                median_matched_s2=np.nan,
            ),
        )
    # 2) Pairwise rate gate with same 'either'/'both' policy
    thr_tr = (
        min_spikes_per_min_train
        if min_spikes_per_min_train is not None
        else min_spikes_per_min
    )
    thr_te = (
        min_spikes_per_min_test
        if min_spikes_per_min_test is not None
        else min_spikes_per_min
    )
    pass_tr = np.ones_like(train_ids, bool)
    pass_te = np.ones_like(test_ids, bool)
    if thr_tr is not None:
        pass_tr = rate_tr[train_ids] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[test_ids] >= float(thr_te)
    if drop_if_below == "either":
        keep_rate = pass_tr & pass_te  # keep only if BOTH pass
    elif drop_if_below == "both":
        keep_rate = pass_tr | pass_te  # keep if EITHER passes
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    train_ids = train_ids[keep_rate]
    test_ids = test_ids[keep_rate]
    if train_ids.size == 0:
        return dict(
            train_ids=np.array([], int),
            test_ids=np.array([], int),
            SI_native_s1=np.array([], float),
            SI_native_s2=np.array([], float),
            SI_matched_s1=np.array([], float),
            SI_matched_s2=np.array([], float),
            mask_intersection=None,
            prior_common=None,
            meta_train=dict(
                rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
            ),
            meta_test=dict(
                rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
            ),
            summary=dict(
                median_native_s1=np.nan,
                median_native_s2=np.nan,
                median_matched_s1=np.nan,
                median_matched_s2=np.nan,
            ),
        )
    # 3) Compute per-session SI for the kept neuron sets (no extra gating inside)
    s1 = compute_spatial_info_session_rategated(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        pos_tr,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        min_time_per_bin_sec=min_time_per_bin_sec,
        eps=eps,
        neuron_ids=train_ids,
        min_spikes_per_min=None,  # already applied pairwise; do NOT gate again
        rate_source=rate_source,
    )
    s2 = compute_spatial_info_session_rategated(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        pos_te,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        min_time_per_bin_sec=min_time_per_bin_sec,
        eps=eps,
        neuron_ids=test_ids,
        min_spikes_per_min=None,
        rate_source=rate_source,
    )
    # 4) Matched SI under requested occupancy control
    sim = compute_spatial_info_matched(s1, s2, control=si_control, eps=eps)
    # 5) Package
    out = dict(
        train_ids=train_ids,
        test_ids=test_ids,
        SI_native_s1=s1["SI_native"],
        SI_native_s2=s2["SI_native"],
        SI_matched_s1=sim["SI_matched_s1"],
        SI_matched_s2=sim["SI_matched_s2"],
        mask_intersection=sim.get("mask_intersection", None),
    )
    if "prior_common" in sim:
        out["prior_common"] = sim["prior_common"]
    if "prior_s1_intersection" in sim:
        out["prior_s1_intersection"] = sim["prior_s1_intersection"]
    if "prior_s2_intersection" in sim:
        out["prior_s2_intersection"] = sim["prior_s2_intersection"]
    out["meta_train"] = dict(
        rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source
    )
    out["meta_test"] = dict(
        rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source
    )
    out["summary"] = dict(
        median_native_s1=float(np.nanmedian(out["SI_native_s1"]))
        if out["SI_native_s1"].size
        else np.nan,
        median_native_s2=float(np.nanmedian(out["SI_native_s2"]))
        if out["SI_native_s2"].size
        else np.nan,
        median_matched_s1=float(np.nanmedian(out["SI_matched_s1"]))
        if out["SI_matched_s1"].size
        else np.nan,
        median_matched_s2=float(np.nanmedian(out["SI_matched_s2"]))
        if out["SI_matched_s2"].size
        else np.nan,
    )
    return out


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def run_similarity_pipeline(
    # â”€â”€â”€ Data â”€â”€â”€
    spike_counts1,
    raw_counts1,
    selected_df1,
    positions1,
    spike_counts2,
    raw_counts2,
    selected_df2,
    positions2,
    cimap=None,
    # â”€â”€â”€ Optional explicit ZERO-BASED neuron ids (override cimap where supported) â”€â”€â”€
    train_ids=None,
    test_ids=None,
    # â”€â”€â”€ Optional explicit EXCLUSIONS for matched pairs â”€â”€â”€
    exclude_train_ids=None,  # iterable of train ids to exclude
    exclude_test_ids=None,  # iterable of test  ids to exclude
    exclude_pairs=None,  # iterable of (train_id, test_id) pairs to exclude
    # â”€â”€â”€ Session name labels â”€â”€â”€
    session1_name="session1",
    session2_name="session2",
    # â”€â”€â”€ Core options â”€â”€â”€
    modes=("spatial", "joint_pos", "level", "joint_lvl"),
    level_col="level",
    head_col="head_cat",
    # â”€â”€â”€ Tuning/Similarity params (shared defaults) â”€â”€â”€
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    n_neurons=None,
    random_state=0,
    eps=1e-12,
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    # â”€â”€â”€ NEW: Firing-rate gating across sessions (events/min) â”€â”€â”€
    min_spikes_per_min=None,  # applies to BOTH if set
    min_spikes_per_min_train=None,  # override for session1 (TRAIN)
    min_spikes_per_min_test=None,  # override for session2 (TEST)
    rate_source="spike",  # 'spike' | 'raw'
    drop_if_below="either",  # 'either' | 'both'
    # â”€â”€â”€ PF shuffle over tuning maps â”€â”€â”€
    run_pf_shuffle=True,
    pf_shuffle_repeats=1000,
    pf_shuffle_session="test",  # 'test' or 'train'
    pf_shuffle_types=("map", "neuron_id"),
    # â”€â”€â”€ PV cross-similarity matrix (unchanged API) â”€â”€â”€
    run_pv_cross_similarity=True,
    pv_metric="pearson",
    pv_normalization="auto",
    min_neurons=3,
    drop_rows_with_too_few_neurons=True,
    # â”€â”€â”€ PV correlation (+ PFâ€“PV shuffles) â”€â”€â”€
    run_pv_correlation=True,
    run_pv_pf_shuffle=True,
    pv_pf_shuffle_repeats=1000,
    pv_pf_shuffle_session="test",  # 'test' or 'train'
    pv_pf_shuffle_types=("neuron_id", "map"),
    # â”€â”€â”€ Level-shift suite (unchanged API) â”€â”€â”€
    run_level_shift=True,
    min_levels=3,
    level_norm="zscore",
    level_shift_pf_shuffle_types=("neuron_id", "map"),
    level_shift_pf_shuffle_session="test",  # 'test' or 'train'
    # â”€â”€â”€ within-session analyses â”€â”€â”€
    run_pivot_tuning=False,
    run_within_session_pv=False,
    # pivot/WS-PV params (shared)
    pivot_win_len_sec=20 * 60,
    pivot_step_sec=60,
    pivot_time_axis="selected",
    frame_dt_sec=None,
    min_time_per_bin_sec=5.0,
    # â”€â”€â”€ Spatial information (new) â”€â”€â”€
    run_spatial_info=False,
    si_min_time_per_bin_sec=5.0,
    si_control="intersection_native",  # 'intersection_avg' | 'intersection_uniform' | 'intersection_s1' | 'intersection_s2' | 'intersection_native'
    # â”€â”€â”€ Saving/plots â”€â”€â”€
    output_dir="similarity_results",
    save_json_summaries=True,
    do_plots=True,
    # â”€â”€â”€ Overwrite switch â”€â”€â”€
    overwrite=False,
):
    """
    If either session name is empty ("" or None), the pipeline enters single-session
    mode and runs ONLY within-session analyses for the non-empty session(s).
    Cross-session analyses are skipped entirely in that case.
    Single-session outputs are saved under:
      <output_dir>/pivot_tuning/mode_<mode>/pivot_tuning_<session>.{pkl,json,png}
      <output_dir>/within_session_pv/mode_<mode>/within_session_pv_<session>.{pkl,json,png}
    """
    import json
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    # â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)
    def _exists(path):
        try:
            return (
                (not overwrite) and os.path.exists(path) and (os.path.getsize(path) > 0)
            )
        except OSError:
            return False
    def _save_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    def _save_json(obj, path):
        with open(path, "w") as f:
            json.dump(
                obj,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, np.generic) else str(x),
            )
    def _save_hist(values, out_png, title, bins=40):
        try:
            v = np.asarray(values).ravel()
            v = v[np.isfinite(v)]
            if v.size == 0:
                return
            plt.figure(figsize=(6, 4))
            plt.hist(v, bins=bins)
            plt.title(title)
            plt.xlabel("value")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[pipeline] (plot skipped: {title}) {e}")
    def _save_heatmap(mat, out_png, title):
        try:
            M = np.asarray(mat)
            if M.ndim != 2 or M.size == 0:
                return
            plt.figure(figsize=(6, 5))
            plt.imshow(M, aspect="auto")
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[pipeline] (heatmap skipped: {title}) {e}")
    def _save_one_shuffle_hist(matrix2d, out_png_prefix, title_prefix, rng):
        try:
            X = np.asarray(matrix2d)
            if X.ndim != 2 or X.shape[0] == 0:
                return
            r_idx = int(rng.randint(X.shape[0]))
            row = X[r_idx]
            row = row[np.isfinite(row)]
            if row.size == 0:
                return
            out_png = f"{out_png_prefix}_one_shuffle_hist_r{r_idx}.png"
            title = f"{title_prefix} â€” one shuffle (r={r_idx})"
            _save_hist(row, out_png, title)
        except Exception as e:
            print(f"[pipeline] (one-shuffle plot skipped: {title_prefix}) {e}")
    _ensure_dir(output_dir)
    # â€”â€”â€” Determine mode: both names present vs. single-session â€”â€”â€”
    name1_ok = bool(session1_name)
    name2_ok = bool(session2_name)
    both_named = name1_ok and name2_ok
    single_session_mode = not both_named
    if single_session_mode:
        print(
            "\n=== Single-session mode: at least one session name is empty â†’ "
            "only within-session analyses will run. Cross-session steps are skipped. ==="
        )
    # Build the list of available sessions for within-session work
    available_sessions = []
    if name1_ok:
        available_sessions.append(
            (session1_name, spike_counts1, raw_counts1, selected_df1, positions1)
        )
    if name2_ok:
        available_sessions.append(
            (session2_name, spike_counts2, raw_counts2, selected_df2, positions2)
        )
    # --- 1D vs 2D position handling (global pipeline check) ---
    def _is_1d_pos(pos):
        arr = np.asarray(pos)
        return (arr.ndim == 1) or (arr.ndim == 2 and arr.shape[1] == 1)
    if available_sessions:
        dims_1d = [_is_1d_pos(POS) for (_, _, _, _, POS) in available_sessions]
        if len(set(dims_1d)) > 1:
            raise ValueError(
                "run_similarity_pipeline: mixed 1D and 2D positions across sessions are not supported. "
                "Please ensure all sessions use either 1D positions (shape (T,) or (T,1)) or 2D positions (T,2)."
            )
        if dims_1d[0] and n_bins_y != 1:
            raise ValueError(
                "run_similarity_pipeline: detected 1D positions (shape (T,) or (T,1)), "
                "but n_bins_y != 1. For 1D, set n_bins_y=1 and use n_bins_x as the number of spatial bins."
            )
    # For cross-session tags/paths (only used if both_named)
    pair_tag = (
        f"{session1_name}_to_{session2_name}"
        if both_named
        else (session1_name or session2_name or "single_session")
    )
    print(f"\n=== Saving similarity pipeline results to: {output_dir} ===")
    rng = np.random.RandomState(random_state)
    # Flip cimap (train/test -> test/train); explicit ids are NOT swapped.
    cimap_swapped = cimap[:, [1, 0]] if cimap is not None else None
    def _has(df, col):
        return (col is not None) and (col in df.columns)
    have_level = (
        _has(selected_df1, level_col) and _has(selected_df2, level_col)
        if both_named
        else (available_sessions and _has(available_sessions[0][3], level_col))
    )
    have_head = (
        _has(selected_df1, head_col) and _has(selected_df2, head_col)
        if both_named
        else (available_sessions and _has(available_sessions[0][3], head_col))
    )
    expanded_modes = []
    for m in modes:
        mm = "joint_pos" if m == "joint" else m
        if mm in ("level", "joint_lvl") and not have_level:
            print(f"[pipeline] Skipping mode={mm} (missing level_col).")
            continue
        if mm in ("hd", "joint_pos", "joint_lvl") and not have_head:
            print(f"[pipeline] Skipping mode={mm} (missing head_col).")
            continue
        expanded_modes.append(mm)
    # Results container: only include keys we will actually fill
    results = {"pivot_tuning": {}, "within_session_pv": {}}
    if both_named:
        results.update(
            {
                "tuning_similarity": {},
                "pf_shuffle": {},
                "pv_cross_similarity": {},
                "pv_correlation": {},
                "pv_pf_shuffle": {},
                "level_shift": {},
            }
        )
    # â”€â”€ 0) Optional single-session analyses (grouped by analysis type) â”€â”€â”€â”€â”€â”€â”€
    if run_pivot_tuning or run_within_session_pv:
        for mode in expanded_modes:
            # create type/mode folders once
            if run_pivot_tuning:
                piv_mode_dir = os.path.join(output_dir, "pivot_tuning", f"mode_{mode}")
                _ensure_dir(piv_mode_dir)
            if run_within_session_pv:
                wsv_mode_dir = os.path.join(
                    output_dir, "within_session_pv", f"mode_{mode}"
                )
                _ensure_dir(wsv_mode_dir)
            for (sess_name, SC, RC, DF, POS) in available_sessions:
                # A) Pivot tuning
                if run_pivot_tuning:
                    piv_base = os.path.join(piv_mode_dir, f"pivot_tuning_{sess_name}")
                    piv_pkl = f"{piv_base}.pkl"
                    piv_json = f"{piv_base}.json"
                    piv_hist = f"{piv_base}_sim_mean_hist.png"
                    if _exists(piv_pkl):
                        print(
                            f"[pipeline] (single) pivot_tuning mode={mode}, session={sess_name} â€¦ exists â†’ skip"
                        )
                        res_pivot = _load_pickle(piv_pkl)
                    else:
                        print(
                            f"[pipeline] (single) pivot_tuning mode={mode}, session={sess_name} â€¦",
                            end="",
                            flush=True,
                        )
                        try:
                            res_pivot = pivot_window_tuning_similarity(
                                SC,
                                RC,
                                DF,
                                POS,
                                mode=mode,
                                level_col=level_col,
                                head_col=head_col,
                                n_bins_x=n_bins_x,
                                n_bins_y=n_bins_y,
                                dt=dt,
                                time_window=time_window,
                                use_peak=use_peak,
                                filter_ratemap=filter_ratemap,
                                sigma=sigma,
                                neuron_ids=None,
                                random_state=random_state,
                                win_len_sec=pivot_win_len_sec,
                                pivot_step_sec=pivot_step_sec,
                                metric=metric,
                                normalization=normalization,
                                intersect_mask=intersect_mask,
                                min_bins=3,
                                eps=eps,
                                min_time_per_bin_sec=min_time_per_bin_sec,
                                min_spikes_per_min=min_spikes_per_min,
                                rate_source=rate_source,
                                pivot_time_axis=pivot_time_axis,
                                frame_dt_sec=frame_dt_sec,
                            )
                            _save_pickle(res_pivot, piv_pkl)
                            if save_json_summaries:
                                _save_json(
                                    {
                                        "n_pivots": int(res_pivot["pivot_times"].size),
                                        "norm": res_pivot.get("normalization", ""),
                                    },
                                    piv_json,
                                )
                            if do_plots and "sim_mean" in res_pivot:
                                _save_hist(
                                    res_pivot["sim_mean"],
                                    piv_hist,
                                    title=f"Pivot tuning mean similarity ({mode}) â€¢ {sess_name}",
                                )
                            print(" done.")
                        except Exception as e:
                            print(f" ERROR: {e}")
                            res_pivot = {"error": str(e)}
                            _save_pickle(res_pivot, piv_pkl)
                    results["pivot_tuning"].setdefault(sess_name, {})[mode] = res_pivot
                # B) Within-session PV temporal similarity
                if run_within_session_pv:
                    wsv_base = os.path.join(
                        wsv_mode_dir, f"within_session_pv_{sess_name}"
                    )
                    wsv_pkl = f"{wsv_base}.pkl"
                    wsv_json = f"{wsv_base}.json"
                    wsv_hist = f"{wsv_base}_sim_mean_hist.png"
                    if _exists(wsv_pkl):
                        print(
                            f"[pipeline] (single) within_session_pv mode={mode}, session={sess_name} â€¦ exists â†’ skip"
                        )
                        res_wsv = _load_pickle(wsv_pkl)
                    else:
                        print(
                            f"[pipeline] (single) within_session_pv mode={mode}, session={sess_name} â€¦",
                            end="",
                            flush=True,
                        )
                        try:
                            res_wsv = within_session_pv_temporal_similarity(
                                SC,
                                RC,
                                DF,
                                POS,
                                mode=mode,
                                level_col=level_col,
                                head_col=head_col,
                                n_bins_x=n_bins_x,
                                n_bins_y=n_bins_y,
                                dt=dt,
                                time_window=time_window,
                                use_peak=use_peak,
                                filter_ratemap=filter_ratemap,
                                sigma=sigma,
                                neuron_ids=None,
                                random_state=random_state,
                                win_len_sec=pivot_win_len_sec,
                                pivot_step_sec=pivot_step_sec,
                                pivot_time_axis=pivot_time_axis,
                                frame_dt_sec=frame_dt_sec,
                                metric=pv_metric,
                                normalization=pv_normalization,
                                intersect_mask=intersect_mask,
                                eps=eps,
                                min_time_per_bin_sec=min_time_per_bin_sec,
                                min_spikes_per_min=min_spikes_per_min,
                                rate_source=rate_source,
                            )
                            _save_pickle(res_wsv, wsv_pkl)
                            if save_json_summaries:
                                _save_json(
                                    {
                                        "n_pivots": int(res_wsv["pivot_times"].size),
                                        "norm": res_wsv.get("normalization", ""),
                                    },
                                    wsv_json,
                                )
                            if do_plots and "sim_mean" in res_wsv:
                                _save_hist(
                                    res_wsv["sim_mean"],
                                    wsv_hist,
                                    title=f"Within-session PV mean ({mode}) â€¢ {sess_name}",
                                )
                            print(" done.")
                        except Exception as e:
                            print(f" ERROR: {e}")
                            res_wsv = {"error": str(e)}
                            _save_pickle(res_wsv, wsv_pkl)
                    results["within_session_pv"].setdefault(sess_name, {})[
                        mode
                    ] = res_wsv
                    # â”€â”€ 0c) Spatial information per session (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    # â”€â”€ 0c) Spatial information per session (rate-gated) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if run_spatial_info:
                    si_root = os.path.join(output_dir, "spatial_info", "mode_spatial")
                    _ensure_dir(si_root)
                    for (sess_name, SC, RC, DF, POS) in available_sessions:
                        si_base = os.path.join(si_root, f"spatial_info_{sess_name}")
                        si_pkl = f"{si_base}.pkl"
                        si_json = f"{si_base}.json"
                        si_hist = f"{si_base}_SI_hist.png"
                        if _exists(si_pkl):
                            print(
                                f"[pipeline] (single) spatial_info â€¢ {sess_name} â€¦ exists â†’ skip"
                            )
                            res_si = _load_pickle(si_pkl)
                        else:
                            print(
                                f"[pipeline] (single) spatial_info â€¢ {sess_name} â€¦",
                                end="",
                                flush=True,
                            )
                            try:
                                res_si = compute_spatial_info_session_rategated(
                                    SC,
                                    RC,
                                    DF,
                                    POS,
                                    n_bins_x=n_bins_x,
                                    n_bins_y=n_bins_y,
                                    dt=dt,
                                    time_window=time_window,
                                    use_peak=use_peak,
                                    filter_ratemap=filter_ratemap,
                                    sigma=sigma,
                                    min_time_per_bin_sec=si_min_time_per_bin_sec,
                                    eps=eps,
                                    neuron_ids=None,
                                    min_spikes_per_min=min_spikes_per_min,  # â† global gate if set
                                    rate_source=rate_source,
                                )
                                _save_pickle(res_si, si_pkl)
                                if save_json_summaries:
                                    _save_json(
                                        {
                                            "n_neurons": int(res_si["SI_native"].size),
                                            "median_bits_per_spike": float(
                                                np.nanmedian(res_si["SI_native"])
                                            )
                                            if res_si["SI_native"].size
                                            else np.nan,
                                        },
                                        si_json,
                                    )
                                if do_plots and res_si["SI_native"].size:
                                    _save_hist(
                                        res_si["SI_native"],
                                        si_hist,
                                        title=f"Spatial information (native, rate-gated) â€¢ {sess_name}",
                                    )
                                print(" done.")
                            except Exception as e:
                                print(f" ERROR: {e}")
                                res_si = {"error": str(e)}
                                _save_pickle(res_si, si_pkl)
                        results.setdefault("spatial_info", {}).setdefault(
                            sess_name, {}
                        )["single"] = res_si
    # â”€â”€ 1) Cross-session per-mode computations (ONLY if both names present) â”€â”€
    if both_named:
        for mode in expanded_modes:
            sub = os.path.join(output_dir, f"mode_{mode}")
            _ensure_dir(sub)
            print(f"\n--- MODE: {mode} ---")
            # 1) Tuning similarity
            tuning_pkl = os.path.join(sub, f"tuning_{pair_tag}.pkl")
            tuning_json = os.path.join(sub, f"tuning_summary_{pair_tag}.json")
            tuning_hist = os.path.join(sub, f"tuning_hist_{pair_tag}.png")
            # 1b) Spatial information (matched; rate-gated pairs)
            if run_spatial_info and mode == "spatial":
                si_root = os.path.join(output_dir, "spatial_info", "mode_spatial")
                _ensure_dir(si_root)
                si_pair_pkl = os.path.join(si_root, f"spatial_info_pair_{pair_tag}.pkl")
                si_pair_json = os.path.join(
                    si_root, f"spatial_info_pair_{pair_tag}.json"
                )
                if _exists(si_pair_pkl):
                    print(f"[pipeline] 1b) spatial_info (matched) â€¦ exists â†’ skip")
                    res_sim = _load_pickle(si_pair_pkl)
                else:
                    print(
                        f"[pipeline] 1b) spatial_info (matched) â€¦", end="", flush=True
                    )
                    try:
                        res_sim = cross_session_spatial_information_rategated(
                            spike_counts1,
                            raw_counts1,
                            selected_df1,
                            positions1,
                            spike_counts2,
                            raw_counts2,
                            selected_df2,
                            positions2,
                            cimap=cimap_swapped,
                            n_bins_x=n_bins_x,
                            n_bins_y=n_bins_y,
                            dt=dt,
                            time_window=time_window,
                            use_peak=use_peak,
                            filter_ratemap=filter_ratemap,
                            sigma=sigma,
                            min_time_per_bin_sec=si_min_time_per_bin_sec,
                            eps=eps,
                            n_neurons=n_neurons,
                            random_state=random_state,
                            train_ids=train_ids,
                            test_ids=test_ids,
                            # exact same rate-gating knobs as the rest of your pipeline:
                            min_spikes_per_min=min_spikes_per_min,
                            min_spikes_per_min_train=min_spikes_per_min_train,
                            min_spikes_per_min_test=min_spikes_per_min_test,
                            rate_source=rate_source,
                            drop_if_below=drop_if_below,
                            si_control=si_control,
                        )
                        _save_pickle(res_sim, si_pair_pkl)
                        if save_json_summaries:
                            _save_json(
                                {
                                    "control": si_control,
                                    "n_pairs": int(res_sim["train_ids"].size),
                                    "median_native_s1": float(
                                        np.nanmedian(res_sim["SI_native_s1"])
                                    )
                                    if res_sim["SI_native_s1"].size
                                    else np.nan,
                                    "median_native_s2": float(
                                        np.nanmedian(res_sim["SI_native_s2"])
                                    )
                                    if res_sim["SI_native_s2"].size
                                    else np.nan,
                                    "median_matched_s1": float(
                                        np.nanmedian(res_sim["SI_matched_s1"])
                                    )
                                    if res_sim["SI_matched_s1"].size
                                    else np.nan,
                                    "median_matched_s2": float(
                                        np.nanmedian(res_sim["SI_matched_s2"])
                                    )
                                    if res_sim["SI_matched_s2"].size
                                    else np.nan,
                                },
                                si_pair_json,
                            )
                        print(" done.")
                    except Exception as e:
                        print(f" ERROR: {e}")
                        res_sim = {"error": str(e)}
                        _save_pickle(res_sim, si_pair_pkl)
                results.setdefault("spatial_info", {})["pair"] = res_sim
            if _exists(tuning_pkl):
                print(f"[pipeline] 1) Tuning similarity â€¦ exists â†’ skip")
                res = _load_pickle(tuning_pkl)
                results["tuning_similarity"][mode] = res
                if save_json_summaries and (not _exists(tuning_json)):
                    _save_json({"summary": res.get("summary", "")}, tuning_json)
                if do_plots and (not _exists(tuning_hist)) and ("similarity" in res):
                    _save_hist(
                        res["similarity"],
                        tuning_hist,
                        title=f"Tuning similarity ({mode}) â€¢ {pair_tag}",
                    )
            else:
                print(f"[pipeline] 1) Tuning similarity â€¦", end="", flush=True)
                try:
                    res = cross_session_tuning_similarity(
                        spike_counts1,
                        raw_counts1,
                        selected_df1,
                        positions1,
                        spike_counts2,
                        raw_counts2,
                        selected_df2,
                        positions2,
                        cimap=cimap_swapped,
                        mode=mode,
                        level_col=level_col,
                        head_col=head_col,
                        n_bins_x=n_bins_x,
                        n_bins_y=n_bins_y,
                        dt=dt,
                        time_window=time_window,
                        use_peak=use_peak,
                        filter_ratemap=filter_ratemap,
                        sigma=sigma,
                        n_neurons=n_neurons,
                        random_state=random_state,
                        train_ids=train_ids,
                        test_ids=test_ids,
                        metric=metric,
                        normalization=normalization,
                        intersect_mask=intersect_mask,
                        eps=eps,
                        min_spikes_per_min=min_spikes_per_min,
                        min_spikes_per_min_train=min_spikes_per_min_train,
                        min_spikes_per_min_test=min_spikes_per_min_test,
                        rate_source=rate_source,
                        drop_if_below=drop_if_below,
                    )
                    results["tuning_similarity"][mode] = res
                    _save_pickle(res, tuning_pkl)
                    if save_json_summaries:
                        _save_json({"summary": res.get("summary", "")}, tuning_json)
                    if do_plots and ("similarity" in res):
                        _save_hist(
                            res["similarity"],
                            tuning_hist,
                            title=f"Tuning similarity ({mode}) â€¢ {pair_tag}",
                        )
                    print(" done.")
                except Exception as e:
                    print(f" ERROR: {e}")
                    results["tuning_similarity"][mode] = {"error": str(e)}
            # 2) PF-shuffle tuning (map & neuron_id)
            if run_pf_shuffle:
                results["pf_shuffle"].setdefault(mode, {})
                for stype in pf_shuffle_types:
                    tag = f"pf_shuffle_{stype}_{pair_tag}"
                    pf_pkl = os.path.join(sub, f"{tag}.pkl")
                    pf_hist = os.path.join(sub, f"{tag}_hist.png")
                    if _exists(pf_pkl):
                        print(
                            f"[pipeline] 2) PF-shuffle (tuning) type={stype} â€¦ exists â†’ skip"
                        )
                        res_pf = _load_pickle(pf_pkl)
                        results["pf_shuffle"][mode][stype] = res_pf
                        if do_plots:
                            if ("sim_per_repeat" in res_pf) and (not _exists(pf_hist)):
                                _save_hist(
                                    res_pf["sim_per_repeat"],
                                    pf_hist,
                                    title=f"PF-shuffle (tuning {stype}) per-repeat means ({mode}) â€¢ {pair_tag}",
                                )
                            if "sim_matrix" in res_pf and isinstance(
                                res_pf["sim_matrix"], np.ndarray
                            ):
                                if not any(
                                    f.startswith(f"{tag}_one_shuffle_hist_r")
                                    for f in os.listdir(sub)
                                ):
                                    _save_one_shuffle_hist(
                                        res_pf["sim_matrix"],
                                        os.path.join(sub, f"{tag}"),
                                        f"PF-shuffle (tuning {stype}) per-neuron corr ({mode}) â€¢ {pair_tag}",
                                        rng,
                                    )
                    else:
                        print(
                            f"[pipeline] 2) PF-shuffle (tuning) type={stype} (repeats={pf_shuffle_repeats}) â€¦",
                            end="",
                            flush=True,
                        )
                        try:
                            res_pf = pf_shuffle_tuning_similarity(
                                spike_counts1,
                                raw_counts1,
                                selected_df1,
                                positions1,
                                spike_counts2,
                                raw_counts2,
                                selected_df2,
                                positions2,
                                cimap=cimap_swapped,
                                mode=mode,
                                level_col=level_col,
                                head_col=head_col,
                                n_bins_x=n_bins_x,
                                n_bins_y=n_bins_y,
                                dt=dt,
                                time_window=time_window,
                                use_peak=use_peak,
                                filter_ratemap=filter_ratemap,
                                sigma=sigma,
                                n_neurons=n_neurons,
                                random_state=random_state,
                                metric=(metric or "pearson"),
                                normalization=(
                                    "auto" if normalization == "none" else normalization
                                ),
                                intersect_mask=intersect_mask,
                                eps=eps,
                                n_repeats=pf_shuffle_repeats,
                                shuffle_session=pf_shuffle_session,
                                shuffle_type=stype,
                                train_ids=train_ids,
                                test_ids=test_ids,
                                min_spikes_per_min=min_spikes_per_min,
                                min_spikes_per_min_train=min_spikes_per_min_train,
                                min_spikes_per_min_test=min_spikes_per_min_test,
                                rate_source=rate_source,
                                drop_if_below=drop_if_below,
                            )
                            results["pf_shuffle"][mode][stype] = res_pf
                            _save_pickle(res_pf, pf_pkl)
                            if do_plots:
                                if "sim_per_repeat" in res_pf:
                                    _save_hist(
                                        res_pf["sim_per_repeat"],
                                        pf_hist,
                                        title=f"PF-shuffle (tuning {stype}) per-repeat means ({mode}) â€¢ {pair_tag}",
                                    )
                                if "sim_matrix" in res_pf and isinstance(
                                    res_pf["sim_matrix"], np.ndarray
                                ):
                                    _save_one_shuffle_hist(
                                        res_pf["sim_matrix"],
                                        os.path.join(sub, f"{tag}"),
                                        f"PF-shuffle (tuning {stype}) per-neuron corr ({mode}) â€¢ {pair_tag}",
                                        rng,
                                    )
                            print(" done.")
                        except Exception as e:
                            print(f" ERROR: {e}")
                            results["pf_shuffle"][mode][stype] = {"error": str(e)}
            # 3) PV cross-similarity matrix
            if run_pv_cross_similarity:
                pv_pkl = os.path.join(sub, f"pv_cross_similarity_{pair_tag}.pkl")
                pv_heatmap = os.path.join(
                    sub, f"pv_cross_similarity_heatmap_{pair_tag}.png"
                )
                if _exists(pv_pkl):
                    print(f"[pipeline] 3) PV cross-similarity matrix â€¦ exists â†’ skip")
                    res_pv = _load_pickle(pv_pkl)
                    results["pv_cross_similarity"][mode] = res_pv
                    if do_plots and (not _exists(pv_heatmap)):
                        mat = None
                        for k in (
                            "matrix",
                            "pv_matrix",
                            "cross_similarity",
                            "similarity_matrix",
                            "M",
                        ):
                            if (
                                isinstance(res_pv.get(k, None), np.ndarray)
                                and res_pv[k].ndim == 2
                            ):
                                mat = res_pv[k]
                                break
                        if mat is None:
                            for v in res_pv.values():
                                if isinstance(v, np.ndarray) and v.ndim == 2:
                                    mat = v
                                    break
                        if mat is not None:
                            _save_heatmap(
                                mat,
                                pv_heatmap,
                                title=f"PV cross-similarity ({mode}) â€¢ {pair_tag}",
                            )
                else:
                    print(
                        f"[pipeline] 3) PV cross-similarity matrix â€¦",
                        end="",
                        flush=True,
                    )
                    try:
                        res_pv = cross_session_pv_cross_similarity_matrix(
                            spike_counts1,
                            raw_counts1,
                            selected_df1,
                            positions1,
                            spike_counts2,
                            raw_counts2,
                            selected_df2,
                            positions2,
                            cimap=cimap_swapped,
                            mode=mode,
                            level_col=level_col,
                            head_col=head_col,
                            n_bins_x=n_bins_x,
                            n_bins_y=n_bins_y,
                            dt=dt,
                            time_window=time_window,
                            use_peak=use_peak,
                            filter_ratemap=filter_ratemap,
                            sigma=sigma,
                            n_neurons=n_neurons,
                            random_state=random_state,
                            metric=pv_metric,
                            normalization=pv_normalization,
                            intersect_mask=intersect_mask,
                            eps=eps,
                            min_neurons=min_neurons,
                            drop_rows_with_too_few_neurons=drop_rows_with_too_few_neurons,
                            train_ids=train_ids,
                            test_ids=test_ids,
                        )
                        results["pv_cross_similarity"][mode] = res_pv
                        _save_pickle(res_pv, pv_pkl)
                        mat = None
                        for k in (
                            "matrix",
                            "pv_matrix",
                            "cross_similarity",
                            "similarity_matrix",
                            "M",
                        ):
                            if (
                                isinstance(res_pv.get(k, None), np.ndarray)
                                and res_pv[k].ndim == 2
                            ):
                                mat = res_pv[k]
                                break
                        if mat is None:
                            for v in res_pv.values():
                                if isinstance(v, np.ndarray) and v.ndim == 2:
                                    mat = v
                                    break
                        if do_plots and (mat is not None):
                            _save_heatmap(
                                mat,
                                pv_heatmap,
                                title=f"PV cross-similarity ({mode}) â€¢ {pair_tag}",
                            )
                        print(" done.")
                    except Exception as e:
                        print(f" ERROR: {e}")
                        results["pv_cross_similarity"][mode] = {"error": str(e)}
            # 4) PV correlation
            if run_pv_correlation:
                pvc_pkl = os.path.join(sub, f"pv_correlation_{pair_tag}.pkl")
                pvc_hist = os.path.join(sub, f"pv_correlation_hist_{pair_tag}.png")
                if _exists(pvc_pkl):
                    print(f"[pipeline] 4) PV correlation â€¦ exists â†’ skip")
                    res_pvc = _load_pickle(pvc_pkl)
                    results["pv_correlation"][mode] = res_pvc
                    if do_plots and (not _exists(pvc_hist)) and ("pvc" in res_pvc):
                        _save_hist(
                            res_pvc["pvc"],
                            pvc_hist,
                            title=f"PV correlation ({mode}) â€¢ {pair_tag}",
                        )
                else:
                    print(f"[pipeline] 4) PV correlation â€¦", end="", flush=True)
                    try:
                        res_pvc = cross_session_pv_correlation(
                            spike_counts_train=spike_counts1,
                            raw_counts_train=raw_counts1,
                            selected_df_train=selected_df1,
                            positions_train=positions1,
                            spike_counts_test=spike_counts2,
                            raw_counts_test=raw_counts2,
                            selected_df_test=selected_df2,
                            positions_test=positions2,
                            cimap=cimap_swapped,
                            mode=mode,
                            level_col=level_col,
                            head_col=head_col,
                            n_bins_x=n_bins_x,
                            n_bins_y=n_bins_y,
                            dt=dt,
                            time_window=time_window,
                            use_peak=use_peak,
                            filter_ratemap=filter_ratemap,
                            sigma=sigma,
                            n_neurons=n_neurons,
                            random_state=random_state,
                            exclude_train_ids=exclude_train_ids,
                            exclude_test_ids=exclude_test_ids,
                            exclude_pairs=exclude_pairs,
                            metric=metric,
                            normalization=(
                                "auto" if normalization == "none" else normalization
                            ),
                            intersect_mask=intersect_mask,
                            min_neurons=min_neurons,
                            eps=eps,
                            train_ids=train_ids,
                            test_ids=test_ids,
                            min_spikes_per_min=min_spikes_per_min,
                            min_spikes_per_min_train=min_spikes_per_min_train,
                            min_spikes_per_min_test=min_spikes_per_min_test,
                            rate_source=rate_source,
                            drop_if_below=drop_if_below,
                        )
                        results["pv_correlation"][mode] = res_pvc
                        _save_pickle(res_pvc, pvc_pkl)
                        if do_plots and ("pvc" in res_pvc):
                            _save_hist(
                                res_pvc["pvc"],
                                pvc_hist,
                                title=f"PV correlation ({mode}) â€¢ {pair_tag}",
                            )
                        print(" done.")
                    except Exception as e:
                        print(f" ERROR: {e}")
                        results["pv_correlation"][mode] = {"error": str(e)}
            # 5) PFâ€“PV shuffle nulls
            if run_pv_pf_shuffle:
                results["pv_pf_shuffle"].setdefault(mode, {})
                for stype in pv_pf_shuffle_types:
                    tag = f"pv_pf_shuffle_{stype}_{pair_tag}"
                    pvpf_pkl = os.path.join(sub, f"{tag}.pkl")
                    pvpf_hist = os.path.join(sub, f"{tag}_hist.png")
                    if _exists(pvpf_pkl):
                        print(
                            f"[pipeline] 5) PFâ€“PV shuffle type={stype} â€¦ exists â†’ skip"
                        )
                        res_pvpf = _load_pickle(pvpf_pkl)
                        results["pv_pf_shuffle"][mode][stype] = res_pvpf
                        if (
                            do_plots
                            and ("pv_shuffles" in res_pvpf)
                            and (not _exists(pvpf_hist))
                        ):
                            means = np.nanmean(res_pvpf["pv_shuffles"], axis=1)
                            _save_hist(
                                means,
                                pvpf_hist,
                                title=f"PFâ€“PV shuffle ({stype}) per-repeat means ({mode}) â€¢ {pair_tag}",
                            )
                            if not any(
                                f.startswith(f"{tag}_one_shuffle_hist_r")
                                for f in os.listdir(sub)
                            ):
                                _save_one_shuffle_hist(
                                    res_pvpf["pv_shuffles"],
                                    os.path.join(sub, f"{tag}"),
                                    f"PFâ€“PV shuffle ({stype}) per-bin PV corr ({mode}) â€¢ {pair_tag}",
                                    rng,
                                )
                    else:
                        print(
                            f"[pipeline] 5) PFâ€“PV shuffle type={stype} (repeats={pv_pf_shuffle_repeats}) â€¦",
                            end="",
                            flush=True,
                        )
                        try:
                            res_pvpf = pf_shuffle_pv_similarity(
                                spike_counts_train=spike_counts1,
                                raw_counts_train=raw_counts1,
                                selected_df_train=selected_df1,
                                positions_train=positions1,
                                spike_counts_test=spike_counts2,
                                raw_counts_test=raw_counts2,
                                selected_df_test=selected_df2,
                                positions_test=positions2,
                                cimap=cimap_swapped,
                                mode=mode,
                                level_col=level_col,
                                head_col=head_col,
                                n_bins_x=n_bins_x,
                                n_bins_y=n_bins_y,
                                dt=dt,
                                time_window=time_window,
                                use_peak=use_peak,
                                filter_ratemap=filter_ratemap,
                                sigma=sigma,
                                eps=eps,
                                n_neurons=n_neurons,
                                random_state=random_state,
                                exclude_train_ids=exclude_train_ids,
                                exclude_test_ids=exclude_test_ids,
                                exclude_pairs=exclude_pairs,
                                metric=metric,
                                normalization=(
                                    "auto" if normalization == "none" else normalization
                                ),
                                intersect_mask=intersect_mask,
                                min_neurons=min_neurons,
                                repeats=pv_pf_shuffle_repeats,
                                rng=None,
                                return_observed=True,
                                shuffle_type=stype,
                                shuffle_session=pv_pf_shuffle_session,
                                train_ids=train_ids,
                                test_ids=test_ids,
                                min_spikes_per_min=min_spikes_per_min,
                                min_spikes_per_min_train=min_spikes_per_min_train,
                                min_spikes_per_min_test=min_spikes_per_min_test,
                                rate_source=rate_source,
                                drop_if_below=drop_if_below,
                            )
                            results["pv_pf_shuffle"][mode][stype] = res_pvpf
                            _save_pickle(res_pvpf, pvpf_pkl)
                            if do_plots and ("pv_shuffles" in res_pvpf):
                                means = np.nanmean(res_pvpf["pv_shuffles"], axis=1)
                                _save_hist(
                                    means,
                                    pvpf_hist,
                                    title=f"PFâ€“PV shuffle ({stype}) per-repeat means ({mode}) â€¢ {pair_tag}",
                                )
                                _save_one_shuffle_hist(
                                    res_pvpf["pv_shuffles"],
                                    os.path.join(sub, f"{tag}"),
                                    f"PFâ€“PV shuffle ({stype}) per-bin PV corr ({mode}) â€¢ {pair_tag}",
                                    rng,
                                )
                            print(" done.")
                        except Exception as e:
                            print(f" ERROR: {e}")
                            results["pv_pf_shuffle"][mode][stype] = {"error": str(e)}
        # â”€â”€ 6) Level-shift suite â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if run_level_shift and have_level:
            sub = os.path.join(output_dir, "level_suite")
            _ensure_dir(sub)
            lvl_pkl = os.path.join(sub, f"level_shift_{pair_tag}.pkl")
            lvl_json = os.path.join(sub, f"level_shift_summary_{pair_tag}.json")
            lvl_hist = os.path.join(sub, f"level_shift_hist_{pair_tag}.png")
            print(f"\n--- LEVEL SUITE ---")
            if _exists(lvl_pkl):
                print(f"[pipeline] 6) Level-shift similarity â€¦ exists â†’ skip")
                res_lvl = _load_pickle(lvl_pkl)
                results["level_shift"]["main"] = res_lvl
                if save_json_summaries and (not _exists(lvl_json)):
                    _save_json({"summary": res_lvl.get("summary", "")}, lvl_json)
                if do_plots:
                    vec = res_lvl.get("best_lag", None)
                    if vec is None:
                        for k in ("best_lags", "lags", "values"):
                            if k in res_lvl:
                                vec = res_lvl[k]
                                break
                    if (vec is not None) and (not _exists(lvl_hist)):
                        _save_hist(
                            vec,
                            lvl_hist,
                            title=f"Level-shift tuning similarity â€¢ {pair_tag}",
                        )
            else:
                print(f"[pipeline] 6) Level-shift similarity â€¦", end="", flush=True)
                try:
                    res_lvl = cross_session_level_shift_similarity(
                        spike_counts1,
                        raw_counts1,
                        selected_df1,
                        positions1,
                        spike_counts2,
                        raw_counts2,
                        selected_df2,
                        positions2,
                        cimap=(cimap[:, [1, 0]] if cimap is not None else None),
                        level_col=level_col,
                        dt=dt,
                        time_window=time_window,
                        use_peak=use_peak,
                        filter_ratemap=filter_ratemap,
                        sigma=sigma,
                        n_neurons=n_neurons,
                        random_state=random_state,
                        intersect_mask=False,
                        min_levels=min_levels,
                        normalization=level_norm,
                        eps=eps,
                    )
                    results["level_shift"]["main"] = res_lvl
                    _save_pickle(res_lvl, lvl_pkl)
                    if save_json_summaries:
                        _save_json({"summary": res_lvl.get("summary", "")}, lvl_json)
                    if do_plots:
                        vec = res_lvl.get("best_lag", None)
                        if vec is None:
                            for k in ("best_lags", "lags", "values"):
                                if k in res_lvl:
                                    vec = res_lvl[k]
                                    break
                        if vec is not None:
                            _save_hist(
                                vec,
                                lvl_hist,
                                title=f"Level-shift tuning similarity â€¢ {pair_tag}",
                            )
                    print(" done.")
                except Exception as e:
                    print(f" ERROR: {e}")
                    results["level_shift"]["main"] = {"error": str(e)}
        elif run_level_shift:
            print(
                "[pipeline] Skipping level-shift suite (level_col missing or single-session mode)."
            )
    else:
        # single-session mode: explicitly skip cross-session suite
        print("\n[pipeline] Cross-session analyses skipped (single-session mode).")
    print("\n[pipeline] Pipeline complete.")
    return results


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def pivot_window_tuning_similarity(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    # what to compare
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    # tuning hyperparams
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,  # sampling interval of the selected rows
    time_window=0,  # used in the one-time Q builder
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    # neurons
    neuron_ids=None,
    random_state=0,
    # pivot windows (seconds)
    win_len_sec=20 * 60,  # 20 minutes
    pivot_step_sec=60,  # 1-minute stride
    # similarity
    metric="pearson",  # 'pearson' | 'spearman' | 'cosine'
    normalization="auto",  # 'none' | 'zscore' | 'l2' | 'auto'
    intersect_mask=True,
    min_bins=3,
    eps=1e-12,
    # frequent-visit / occupancy control (bin inclusion, SECONDS)
    min_time_per_bin_sec=5.0,
    # firing-rate based masking (spikes per minute)
    min_spikes_per_min=None,  # float or None -> drop neurons with rate < threshold
    rate_source="spike",  # 'spike' | 'raw' (raw_counts sampled at selected_df['frame'])
    # time axis control
    pivot_time_axis="selected",  # 'selected' | 'frame' | 'maze'
    frame_dt_sec=None,  # used when pivot_time_axis='frame'; falls back to dt if None
):
    """
    Functionality:
    - Pivot window tuning similarity.
    Inputs:
    - `spike_counts`: Neural activity values.
      Typical shape: Typically (n_frames, n_neurons).
    - `raw_counts`: Raw activity values before processing.
      Typical shape: Typically (n_frames, n_neurons).
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `positions`: Position coordinates over time.
      Typical shape: Typically (n_frames, 2) for x/y.
    - `mode`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `level_col`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `head_col`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `n_bins_x`: Count/size parameter.
      Typical shape: Scalar integer.
    - `n_bins_y`: Count/size parameter.
      Typical shape: Scalar integer.
    - `dt`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `time_window`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `use_peak`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `filter_ratemap`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `sigma`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `neuron_ids`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `random_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `win_len_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pivot_step_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `metric`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `normalization`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `intersect_mask`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_bins`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `eps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_time_per_bin_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_spikes_per_min`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `rate_source`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pivot_time_axis`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `frame_dt_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    T_sel, N_total = spike_counts.shape
    assert "frame" in selected_df.columns, "selected_df must contain a 'frame' column"
    assert T_sel == len(selected_df), "spike_counts rows must match selected_df rows"
    # --- 0) Position dimensionality handling (1D vs 2D) ---
    pos = np.asarray(positions)
    is_1d = (pos.ndim == 1) or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError(
            "pivot_window_tuning_similarity: for 1D positions (shape (T,) or (T,1)), "
            "please set n_bins_y=1 and use n_bins_x for the number of spatial bins."
        )
    if neuron_ids is None:
        neuron_ids = np.arange(N_total, dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    # --- Build the time axis (SECONDS, length T_sel) ---
    if pivot_time_axis == "selected":
        t_sel = np.arange(T_sel, dtype=float) * float(dt)
        frame_dt_used = None
        frame_dt_inferred = False
    elif pivot_time_axis == "frame":
        frame_idx = selected_df["frame"].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)  # fallback
            frame_dt_inferred = True
        else:
            frame_dt_inferred = False
        t_sel = frame_idx * float(frame_dt_sec)
        frame_dt_used = float(frame_dt_sec)
    elif pivot_time_axis == "maze":
        if "maze_time" not in selected_df.columns:
            raise ValueError(
                "pivot_time_axis='maze' requires selected_df['maze_time'] in seconds."
            )
        t_sel = selected_df["maze_time"].to_numpy(dtype=float)
        frame_dt_used = None
        frame_dt_inferred = False
    else:
        raise ValueError("pivot_time_axis must be one of {'selected','frame','maze'}.")
    # Effective dt for tuning inputs
    if t_sel.size > 1 and np.all(np.isfinite(np.diff(t_sel))):
        dt_eff = float(np.mean(np.diff(t_sel)))
    else:
        dt_eff = float(dt)
    # --- Firing-rate based neuron filtering ---
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, dtype=float).sum(axis=0)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        if total_minutes <= 0:
            return np.zeros(C.shape[1], dtype=float)
        return total_events / total_minutes
    if rate_source == "spike":
        C_for_rate = spike_counts
        dt_for_rate = dt
    elif rate_source == "raw":
        frames_cont = selected_df["frame"].to_numpy().astype(int)
        C_for_rate = raw_counts[frames_cont, :]
        dt_for_rate = dt_eff
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    rate_per_min = _spikes_per_min(C_for_rate, dt_for_rate)
    if min_spikes_per_min is not None:
        thr = float(min_spikes_per_min)
        keep_mask = rate_per_min[neuron_ids] >= thr
        neuron_ids = neuron_ids[keep_mask]
        if neuron_ids.size == 0:
            # prepare minimal metadata for pivots
            W = float(win_len_sec)
            if t_sel[-1] - t_sel[0] < 2 * W:
                piv_times = np.array([], dtype=float)
            else:
                t0 = t_sel[0] + W
                t1 = t_sel[-1] - W
                piv_times = np.arange(t0, t1 + 1e-9, float(pivot_step_sec), dtype=float)
            return dict(
                pivot_bounds=np.zeros((piv_times.size, 2, 2), dtype=int),
                pivot_times=piv_times,
                pivot_times_maze=None,
                sim_matrix=np.zeros((piv_times.size, 0), dtype=float),
                sim_mean=np.full(piv_times.size, np.nan, float),
                sim_std=np.full(piv_times.size, np.nan, float),
                mode=mode,
                metric=metric,
                normalization=normalization,
                meta=dict(
                    x_edges=None,
                    y_edges=None,
                    n_bins_x=n_bins_x,
                    n_bins_y=n_bins_y,
                    kept_neuron_ids=np.array([], dtype=int),
                    rate_per_min=rate_per_min,
                    threshold=thr,
                    rate_source=rate_source,
                    pivot_time_axis=pivot_time_axis,
                    frame_dt_used=frame_dt_used,
                    frame_dt_inferred=frame_dt_inferred,
                    min_time_per_bin_sec=min_time_per_bin_sec,
                ),
            )
    M = int(neuron_ids.size)
    # --- Discretize positions once (1D or 2D) ---
    n_bins = n_bins_x * n_bins_y
    labels_pos, x_edges, y_edges = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_lvl = None
    uniq_lvls = None
    if mode in ("level", "joint_lvl"):
        v = selected_df[level_col].values
        uniq_lvls = np.unique(v)
        lvl_map = {val: i for i, val in enumerate(uniq_lvls)}
        labels_lvl = np.array([lvl_map[val] for val in v])
        n_lvls = len(uniq_lvls)
    labels_hd = None
    uniq_hd = None
    if mode in ("hd", "joint_pos"):
        h = selected_df[head_col].values
        uniq_hd = np.unique(h)
        hd_map = {val: i for i, val in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[val] for val in h])
        n_hd = len(uniq_hd) if uniq_hd is not None else None
    # --- Build pivot times & bounds on chosen time axis ---
    W = float(win_len_sec)
    step_sec = float(pivot_step_sec)
    if (t_sel[-1] - t_sel[0]) <= 2 * W:
        raise ValueError(
            "Session too short for chosen win_len_sec on the selected time axis."
        )
    t0 = t_sel[0] + W
    t1 = t_sel[-1] - W
    pivot_times = np.arange(t0, t1 + 1e-9, step_sec, dtype=float)
    n_pivots = pivot_times.size
    bounds = np.zeros((n_pivots, 2, 2), dtype=int)  # [[s_b,e_b],[s_a,e_a]]
    for j, tp in enumerate(pivot_times):
        s_b = int(np.searchsorted(t_sel, tp - W, side="left"))
        e_b = int(np.searchsorted(t_sel, tp, side="left"))
        s_a = int(np.searchsorted(t_sel, tp, side="left"))
        e_a = int(np.searchsorted(t_sel, tp + W, side="left"))
        bounds[j, 0, :] = [s_b, e_b]
        bounds[j, 1, :] = [s_a, e_a]
    # --- ALWAYS compute the maze-time at pivot centers ---
    if "maze_time" in selected_df.columns and n_pivots > 0:
        maze_t = selected_df["maze_time"].to_numpy(dtype=float)
        pivot_center_idx = bounds[:, 1, 0]  # s_a for each pivot (center index)
        pivot_times_maze = maze_t[pivot_center_idx]
    else:
        pivot_times_maze = None
    # --- Normalization per metric ---
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    if normalization == "auto":
        col_norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        col_norm_mode = normalization
    def _normalize_columns(X, mode_):
        if mode_ == "none":
            return X
        if mode_ == "zscore":
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True)
            return (X - mu) / (sd + eps)
        if mode_ == "l2":
            nrm = np.linalg.norm(X, axis=0, keepdims=True)
            return X / (nrm + eps)
        return X
    def _per_neuron_similarity(A, B):
        A = _normalize_columns(A, col_norm_mode)
        B = _normalize_columns(B, col_norm_mode)
        out = np.full(M, np.nan, float)
        for i in range(M):
            a, b = A[:, i], B[:, i]
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() < min_bins:
                continue
            if metric == "pearson":
                out[i] = np.corrcoef(a[good], b[good])[0, 1]
            elif metric == "spearman":
                out[i] = spearmanr(a[good], b[good])[0]
            else:
                na = np.linalg.norm(a[good])
                nb = np.linalg.norm(b[good])
                out[i] = float((a[good] @ b[good]) / (na * nb + eps))
        return out
    # --- Build Q once (NÃ—T_sel) then transpose to (T_selÃ—M) ---
    Q_full = build_count_matrix(
        spike_counts,
        raw_counts,
        selected_df,
        frames=np.arange(T_sel),
        time_window=time_window,
        neuron_ids=neuron_ids,
    ).T  # (T_sel, M)
    # ---------- helpers to compute OCCUPANCY IN SECONDS (not priors) ----------
    def _occ_secs_spatial(labels, n_bins_local):
        return np.bincount(labels, minlength=n_bins_local).astype(float) * dt_eff
    def _occ_secs_level(labels_level, nL):
        return np.bincount(labels_level, minlength=nL).astype(float) * dt_eff
    def _occ_secs_hd(labels_hd_local, nH):
        return np.bincount(labels_hd_local, minlength=nH).astype(float) * dt_eff
    def _occ_secs_joint_pos(pos_labels, hd_labels, nB, nH):
        idx = pos_labels.astype(int) * nH + hd_labels.astype(int)
        occ = np.bincount(idx, minlength=nB * nH).astype(float) * dt_eff
        return occ.reshape(nB, nH)
    def _occ_secs_joint_lvl(pos_labels, lvl_labels, nB, nL):
        occL = np.zeros((nL, nB), dtype=float)
        for L in range(nL):
            m = lvl_labels == L
            if m.any():
                occL[L] = (
                    np.bincount(pos_labels[m], minlength=nB).astype(float) * dt_eff
                )
        return occL  # (nL, nB)
    # --- Compute similarity over pivots ---
    sim_rows = []
    for j in range(n_pivots):
        s_b, e_b = bounds[j, 0]
        s_a, e_a = bounds[j, 1]
        Qb = Q_full[s_b:e_b, :]
        Qa = Q_full[s_a:e_a, :]
        yb_pos = labels_pos[s_b:e_b]
        ya_pos = labels_pos[s_a:e_a]
        if mode == "spatial":
            # occupancy seconds per bin
            occ_b = _occ_secs_spatial(yb_pos, n_bins)
            occ_a = _occ_secs_spatial(ya_pos, n_bins)
            lamb_b, _pr_b = tuner(
                Qb,
                yb_pos,
                None,
                n_bins,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            lamb_a, _pr_a = tuner(
                Qa,
                ya_pos,
                None,
                n_bins,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A, B = lamb_b[mask, :], lamb_a[mask, :]
        elif mode == "level":
            yb_lvl = labels_lvl[s_b:e_b]
            nL = int(np.max(labels_lvl)) + 1
            ya_lvl = labels_lvl[s_a:e_a]
            occ_b = _occ_secs_level(yb_lvl, nL)  # (nL,)
            occ_a = _occ_secs_level(ya_lvl, nL)
            lamb_b, _ = tuner(
                Qb,
                yb_lvl,
                None,
                nL,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            lamb_a, _ = tuner(
                Qa,
                ya_lvl,
                None,
                nL,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A, B = lamb_b[mask, :], lamb_a[mask, :]
        elif mode == "hd":
            yb_hd = labels_hd[s_b:e_b]
            nH = int(np.max(labels_hd)) + 1
            ya_hd = labels_hd[s_a:e_a]
            occ_b = _occ_secs_hd(yb_hd, nH)  # (nH,)
            occ_a = _occ_secs_hd(ya_hd, nH)
            _, _, lamb_b, _ = tuner(
                Qb,
                yb_pos,
                yb_hd,
                n_bins,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="independent",
            )
            _, _, lamb_a, _ = tuner(
                Qa,
                ya_pos,
                ya_hd,
                n_bins,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="independent",
            )
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A, B = lamb_b[mask, :], lamb_a[mask, :]
        elif mode == "joint_pos":
            yb_hd = labels_hd[s_b:e_b]
            nH = int(np.max(labels_hd)) + 1
            ya_hd = labels_hd[s_a:e_a]
            occ_b = _occ_secs_joint_pos(yb_pos, yb_hd, n_bins, nH)  # (n_bins, nH)
            occ_a = _occ_secs_joint_pos(ya_pos, ya_hd, n_bins, nH)
            lamb_b, _ = tuner(
                Qb,
                yb_pos,
                yb_hd,
                n_bins,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="joint",
            )
            lamb_a, _ = tuner(
                Qa,
                ya_pos,
                ya_hd,
                n_bins,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="joint",
            )
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = lamb_b[mask].reshape(-1, lamb_b.shape[-1])
            B = lamb_a[mask].reshape(-1, lamb_a.shape[-1])
        elif mode == "joint_lvl":
            yb_lvl = labels_lvl[s_b:e_b]
            nL = int(np.max(labels_lvl)) + 1
            ya_lvl = labels_lvl[s_a:e_a]
            occ_b = _occ_secs_joint_lvl(yb_pos, yb_lvl, n_bins, nL)  # (nL, n_bins)
            occ_a = _occ_secs_joint_lvl(ya_pos, ya_lvl, n_bins, nL)
            lamL_b, lamL_a = [], []
            for Lidx in range(nL):
                mb = yb_lvl == Lidx
                ma = ya_lvl == Lidx
                if mb.any():
                    lb, _ = tuner(
                        Qb[mb],
                        yb_pos[mb],
                        None,
                        n_bins,
                        dt_eff,
                        filter_ratemap,
                        sigma,
                        min_time=min_time_per_bin_sec,
                        eps=eps,
                        decode_head="none",
                    )
                else:
                    lb = np.zeros((n_bins, M))
                if ma.any():
                    la, _ = tuner(
                        Qa[ma],
                        ya_pos[ma],
                        None,
                        n_bins,
                        dt_eff,
                        filter_ratemap,
                        sigma,
                        min_time=min_time_per_bin_sec,
                        eps=eps,
                        decode_head="none",
                    )
                else:
                    la = np.zeros((n_bins, M))
                lamL_b.append(lb)
                lamL_a.append(la)
            lamb_b = np.stack(lamL_b, axis=0)  # (nL, n_bins, M)
            lamb_a = np.stack(lamL_a, axis=0)  # (nL, n_bins, M)
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = lamb_b[mask].reshape(-1, M)
            B = lamb_a[mask].reshape(-1, M)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        # If nothing survives occupancy threshold, skip (NaNs will be in stats)
        if A.size == 0 or B.size == 0:
            sim_rows.append(np.full(M, np.nan, float))
            continue
        sim_rows.append(_per_neuron_similarity(A, B))
    sim_matrix = np.vstack(sim_rows) if len(sim_rows) else np.zeros((0, M))
    sim_mean = np.nanmean(sim_matrix, axis=1) if sim_matrix.size else np.array([])
    sim_std = np.nanstd(sim_matrix, axis=1) if sim_matrix.size else np.array([])
    meta = dict(
        x_edges=x_edges,
        y_edges=y_edges,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        kept_neuron_ids=neuron_ids,
        rate_per_min=rate_per_min,
        threshold=min_spikes_per_min,
        rate_source=rate_source,
        pivot_time_axis=pivot_time_axis,
        frame_dt_used=frame_dt_used,
        frame_dt_inferred=frame_dt_inferred,
        pivot_times_maze=pivot_times_maze,  # also stored in meta for convenience
        min_time_per_bin_sec=min_time_per_bin_sec,
    )
    if uniq_lvls is not None:
        meta["levels"] = uniq_lvls
    if uniq_hd is not None:
        meta["head_bins"] = uniq_hd
    return dict(
        pivot_bounds=bounds,  # int indices into selected rows
        pivot_times=pivot_times,  # seconds on chosen axis
        pivot_times_maze=pivot_times_maze,  # seconds on maze-time axis (None if not available)
        sim_matrix=sim_matrix,
        sim_mean=sim_mean,
        sim_std=sim_std,
        mode=mode,
        metric=metric,
        normalization=col_norm_mode,
        meta=meta,
    )


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _make_prior(bin_counts, valid_bins, mode, eps=1e-12, reference_prior=None):
    """
    mode:
      - 'empirical' : p(x) âˆ occupancy (bin_counts) on valid_bins
      - 'uniform'   : p(x) uniform on valid_bins
      - 'reference' : p(x) = reference_prior on valid_bins (then renormalize)
    """
    bin_counts = np.asarray(bin_counts, float)
    valid_bins = np.asarray(valid_bins, bool)
    p = np.zeros_like(bin_counts, dtype=float)
    if mode == "empirical":
        p[valid_bins] = bin_counts[valid_bins]
    elif mode == "uniform":
        p[valid_bins] = 1.0
    elif mode == "reference":
        if reference_prior is None:
            raise ValueError("reference_prior is required for prior_mode='reference'")
        rp = np.asarray(reference_prior, float)
        if rp.shape != p.shape:
            raise ValueError("reference_prior must have shape (n_bins,)")
        p[valid_bins] = rp[valid_bins]
    else:
        raise ValueError("prior_mode must be 'empirical'|'uniform'|'reference'")
    s = p.sum()
    if s > eps:
        p /= s
    return p


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin]
def _accumulate_spikes_per_bin(Q, labels, n_bins):
    """
    Functionality:
    - Accumulate spikes per bin.
    Inputs:
    - `Q`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `labels`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `n_bins`: Count/size parameter.
      Typical shape: Scalar integer.
    Outputs:
    - Variable `spike_sum` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    spike_sum = np.zeros((n_bins, Q.shape[1]), dtype=float)
    np.add.at(spike_sum, labels, Q)
    return spike_sum


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def _compute_lam_visited(Q, labels, n_bins, dt_eff, eps):
    """
    Standard occupancy-normalized lambda estimate on visited bins.
    """
    bin_counts = np.bincount(labels, minlength=n_bins).astype(float)
    spike_sum = _accumulate_spikes_per_bin(Q, labels, n_bins)
    lam = spike_sum / (bin_counts[:, None] * dt_eff + eps)
    return lam, bin_counts, spike_sum


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def _compute_lam(Q, labels, n_bins, dt_eff, eps, valid_bins):
    """
    Functionality:
    - Compute lam.
    Inputs:
    - `Q`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `labels`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `n_bins`: Count/size parameter.
      Typical shape: Scalar integer.
    - `dt_eff`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `eps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `valid_bins`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    bin_counts = np.bincount(labels, minlength=n_bins).astype(float)
    spike_sum = _accumulate_spikes_per_bin(Q, labels, n_bins)
    lam = spike_sum / (bin_counts[:, None] * dt_eff + eps)
    lam[~np.asarray(valid_bins, bool), :] = 0.0
    return lam, bin_counts


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _impute_lam_for_unvisited(
    lam, visited_mask, prior_support_mask, Q, dt_eff, eps, policy="global_mean"
):
    """
    Fill lam[b, :] for bins that are in prior_support_mask but not in visited_mask.
    policy:
      - 'global_mean' : set unvisited bins to the neuron's global mean rate in the window
                        (conservative; tends not to inflate SI)
      - 'zero'        : set unvisited bins to 0 (often inflates SI; "optimistic/upper bound")
    """
    lam = np.asarray(lam, float).copy()
    visited_mask = np.asarray(visited_mask, bool)
    prior_support_mask = np.asarray(prior_support_mask, bool)
    unvisited = prior_support_mask & (~visited_mask)
    if not np.any(unvisited):
        return lam
    if policy == "global_mean":
        # global mean firing rate in this window (events/sec) per neuron
        # (sum spikes) / (T * dt_eff)
        T = Q.shape[0]
        total_time = max(T * float(dt_eff), eps)
        lam0 = np.sum(Q, axis=0) / total_time  # (M,)
        lam[unvisited, :] = lam0[None, :]
    elif policy == "zero":
        lam[unvisited, :] = 0.0
    else:
        raise ValueError("unvisited_lam_policy must be 'global_mean' or 'zero'")
    return lam


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _matched_subsample_indices(
    labels_window, support_bins, p_target, K_target, rng, replace_if_needed=True
):
    """
    Functionality:
    - Matched subsample indices.
    Inputs:
    - `labels_window`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `support_bins`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `p_target`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `K_target`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `rng`: Random number generator.
      Typical shape: Object or seed-like scalar.
    - `replace_if_needed`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    - Variable `idx` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    labels_window = np.asarray(labels_window, int)
    support_bins = np.asarray(support_bins, bool)
    p = np.asarray(p_target, float).copy()
    p[~support_bins] = 0.0
    ps = p.sum()
    if ps <= 0:
        return np.array([], dtype=int)
    p /= ps
    bins = np.flatnonzero(p > 0)
    if bins.size == 0 or K_target <= 0:
        return np.array([], dtype=int)
    desired_float = K_target * p[bins]
    desired = np.floor(desired_float).astype(int)
    rem = int(K_target - desired.sum())
    if rem > 0:
        frac = desired_float - np.floor(desired_float)
        desired[np.argsort(-frac)[:rem]] += 1
    picks = []
    for b, k in zip(bins, desired):
        if k <= 0:
            continue
        idx_b = np.where(labels_window == b)[0]
        if idx_b.size == 0:
            continue
        if idx_b.size >= k:
            picks.append(rng.choice(idx_b, size=k, replace=False))
        else:
            picks.append(rng.choice(idx_b, size=k, replace=replace_if_needed))
    if len(picks) == 0:
        return np.array([], dtype=int)
    idx = np.concatenate(picks)
    if idx.size < K_target:
        pool = np.where(support_bins[labels_window])[0]
        if pool.size > 0:
            extra = rng.choice(pool, size=(K_target - idx.size), replace=True)
            idx = np.concatenate([idx, extra])
    if idx.size > K_target:
        idx = rng.choice(idx, size=K_target, replace=False)
    rng.shuffle(idx)
    return idx


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def _compute_reference_prior_from_labels(labels_all, n_bins, min_frames, eps=1e-12):
    """
    Functionality:
    - Compute reference prior from labels.
    Inputs:
    - `labels_all`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `n_bins`: Count/size parameter.
      Typical shape: Scalar integer.
    - `min_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `eps`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    labels_all = np.asarray(labels_all, dtype=int)
    bc = np.bincount(labels_all, minlength=n_bins).astype(float)
    valid = bc >= int(min_frames)
    p = np.zeros(n_bins, float)
    p[valid] = bc[valid]
    s = p.sum()
    if s > eps:
        p /= s
    return p, valid


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _permute_labels(labels, rng):
    """
    Functionality:
    - Permute labels.
    Inputs:
    - `labels`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `rng`: Random number generator.
      Typical shape: Object or seed-like scalar.
    Outputs:
    - Variable `labels` (type/shape from runtime context).
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels
    idx = rng.permutation(labels.size)
    return labels[idx]


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _circular_shift_Q(Q, rng, min_shift_frames=1, per_neuron=True):
    """
    Functionality:
    - Circular shift Q.
    Inputs:
    - `Q`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `rng`: Random number generator.
      Typical shape: Object or seed-like scalar.
    - `min_shift_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `per_neuron`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    - Variable `out` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    Q = np.asarray(Q)
    T = Q.shape[0]
    if T <= 1:
        return Q.copy()
    min_shift = int(min_shift_frames)
    min_shift = max(min_shift, 1)
    def _rand_shift():
        if hasattr(rng, "integers"):
            return int(rng.integers(min_shift, T))
        return int(rng.randint(min_shift, T))
    if per_neuron:
        out = np.empty_like(Q)
        for j in range(Q.shape[1]):
            shift = _rand_shift()
            out[:, j] = np.roll(Q[:, j], shift)
        return out
    shift = _rand_shift()
    return np.roll(Q, shift, axis=0)


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: spatial]
def pivot_window_spatial_information_fast(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    win_len_sec=10 * 60,
    pivot_step_sec=60,
    pivot_time_axis="selected",
    frame_dt_sec=None,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    neuron_ids=None,
    min_spikes_per_min=None,
    rate_source="spike",
    random_state=0,
    # ---- SI weighting (Option 1) ----
    prior_mode="empirical",  # 'empirical'|'uniform'|'reference'
    reference_prior=None,  # (n_bins,) or None -> computed from whole-session occupancy
    # ---- reference scope: conditional vs unconditional ----
    reference_scope="conditional",  # 'conditional' (mask+renorm) | 'unconditional' (use ref support; impute unvisited)
    unvisited_lam_policy="global_mean",  # for reference_scope='unconditional': 'global_mean'|'zero'
    # ---- support control (Option 2) ----
    support_mode="window",  # 'window' | 'common_intersection' | 'common_fraction'
    common_support_frac=1.0,  # used if support_mode='common_fraction'
    # ---- occupancy-matched subsampling (Option 3) ----
    subsample_mode="none",  # 'none' | 'matched'
    subsample_target="uniform",  # 'uniform' | 'reference'
    subsample_K=None,  # None -> min available frames in support across windows
    subsample_n_resamples=50,
    subsample_replace_if_needed=True,
):
    """
    Adds what we discussed:
      - coverage / K_valid per window
      - normalized SI: si_matrix_norm = si_matrix / log2(K_valid)  (per window, broadcast across neurons)
      - reference_scope:
          * 'conditional'   : reference prior restricted to valid bins and renormalized (safe; what you already had)
          * 'unconditional' : use reference prior over its support; for bins not visited in a window, impute lambda
                             (lets you define an explicit policy instead of silently renormalizing)
    NOTE:
      - This function assumes spike_counts/positions are aligned to selected_df rows (T_all == len(selected_df)).
      - Requires existing functions: discretize_positions(...), build_count_matrix(...) (if time_window>0), _skaggs_info(...)
    """
    rng = np.random.default_rng(random_state)
    # --- positions / bins ---
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    if pos.shape[1] == 1 and n_bins_y != 1:
        raise ValueError("1D positions but n_bins_y != 1")
    T_all = spike_counts.shape[0]
    if len(selected_df) != T_all:
        raise ValueError(
            "Expected spike_counts and selected_df to be aligned (same T)."
        )
    n_bins = int(n_bins_x * n_bins_y)
    labels_all, x_edges, y_edges = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_all = np.asarray(labels_all, dtype=int)
    # --- neuron selection ---
    if neuron_ids is None:
        neuron_ids = np.arange(spike_counts.shape[1], dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    if min_spikes_per_min is not None:
        if rate_source == "spike":
            src = spike_counts
            src_dt = float(dt)
        elif rate_source == "raw":
            if raw_counts.shape[0] == T_all:
                src = raw_counts
            else:
                frames = selected_df["frame"].to_numpy(dtype=int)
                src = raw_counts[frames, :]
            src_dt = float(dt)
        else:
            raise ValueError("rate_source must be 'spike' or 'raw'")
        rates = src.sum(axis=0) / ((T_all * src_dt) / 60.0 + eps)
        neuron_ids = neuron_ids[rates[neuron_ids] >= float(min_spikes_per_min)]
    M = int(neuron_ids.size)
    if M == 0:
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([]),
            pivot_times_maze=None,
            si_matrix=np.zeros((0, 0)),
            si_mean=np.array([]),
            si_std=np.array([]),
            si_matrix_norm=np.zeros((0, 0)),
            si_mean_norm=np.array([]),
            si_std_norm=np.array([]),
            kept_neuron_ids=neuron_ids,
            meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
        )
    # --- time axis for windows ---
    if pivot_time_axis == "maze":
        if "maze_time" not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time']")
        t_sel = selected_df["maze_time"].to_numpy(dtype=float)
        maze_times_all = t_sel
    elif pivot_time_axis == "frame":
        if frame_dt_sec is None:
            raise ValueError("pivot_time_axis='frame' requires frame_dt_sec")
        frames = selected_df["frame"].to_numpy(dtype=float)
        t_sel = frames * float(frame_dt_sec)
        maze_times_all = (
            selected_df["maze_time"].to_numpy(dtype=float)
            if "maze_time" in selected_df.columns
            else None
        )
    else:
        t_sel = np.arange(T_all, dtype=float) * float(dt)
        maze_times_all = (
            selected_df["maze_time"].to_numpy(dtype=float)
            if "maze_time" in selected_df.columns
            else None
        )
    # --- pivot centers & bounds ---
    t_min, t_max = float(t_sel[0]), float(t_sel[-1])
    half_win = float(win_len_sec) / 2.0
    pivot_centers = []
    t = t_min + half_win
    while t + half_win <= t_max:
        pivot_centers.append(t)
        t += float(pivot_step_sec)
    if len(pivot_centers) == 0:
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([]),
            pivot_times_maze=None,
            si_matrix=np.zeros((0, M)),
            si_mean=np.array([]),
            si_std=np.array([]),
            si_matrix_norm=np.zeros((0, M)),
            si_mean_norm=np.array([]),
            si_std_norm=np.array([]),
            kept_neuron_ids=neuron_ids,
            meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
        )
    pivot_times = np.asarray(pivot_centers, dtype=float)
    bounds = []
    maze_time_centers = []
    for tc in pivot_centers:
        t_start, t_end = tc - half_win, tc + half_win
        idx = np.where((t_sel >= t_start) & (t_sel < t_end))[0]
        if idx.size == 0:
            bounds.append((0, 0))
            maze_time_centers.append(np.nan)
        else:
            s, e = int(idx[0]), int(idx[-1]) + 1
            bounds.append((s, e))
            maze_time_centers.append(
                float(np.mean(maze_times_all[s:e]))
            ) if maze_times_all is not None else maze_time_centers.append(np.nan)
    pivot_times_maze = (
        np.asarray(maze_time_centers) if maze_times_all is not None else None
    )
    # --- counts, tuning params ---
    spike_counts_filt = spike_counts[:, neuron_ids]
    dt_eff_tuning = float(dt) * float(time_window + 1)
    min_frames = int(np.ceil(float(min_time_per_bin_sec) / dt_eff_tuning))
    # --- reference prior from whole-session occupancy (if needed) ---
    reference_support = None
    if (
        prior_mode == "reference"
        or subsample_target == "reference"
        or reference_prior is not None
    ):
        if reference_prior is None:
            bc_full = np.bincount(labels_all, minlength=n_bins).astype(float)
            # keep only bins with decent sampling in the whole session (optional but usually helps)
            full_valid = bc_full >= min_frames
            rp = np.zeros(n_bins, float)
            rp[full_valid] = bc_full[full_valid]
            s = rp.sum()
            reference_prior = (rp / s) if s > eps else rp
        else:
            reference_prior = np.asarray(reference_prior, float)
            if reference_prior.shape != (n_bins,):
                raise ValueError("reference_prior must have shape (n_bins,)")
        reference_support = reference_prior > 0
    # --- Option 2: common support prepass ---
    common_support = None
    if support_mode in ("common_intersection", "common_fraction"):
        valid_count = np.zeros(n_bins, dtype=int)
        n_w = 0
        for (s_win, e_win) in bounds:
            if e_win <= s_win:
                continue
            bc = np.bincount(labels_all[s_win:e_win], minlength=n_bins)
            vb = bc >= min_frames
            valid_count += vb.astype(int)
            n_w += 1
        if n_w == 0:
            common_support = np.zeros(n_bins, dtype=bool)
        elif support_mode == "common_intersection":
            common_support = valid_count == n_w
        else:
            thr = int(np.ceil(float(common_support_frac) * n_w))
            common_support = valid_count >= thr
    # --- Option 3: choose default K_target if needed ---
    K_target = None
    if subsample_mode == "matched":
        totals = []
        for (s_win, e_win) in bounds:
            if e_win <= s_win:
                continue
            bc = np.bincount(labels_all[s_win:e_win], minlength=n_bins)
            vb = bc >= min_frames
            if common_support is not None:
                vb &= common_support
            if subsample_target == "reference" and reference_support is not None:
                vb &= reference_support
            totals.append(int(bc[vb].sum()))
        K_target = int(np.min(totals)) if len(totals) else 0
        if subsample_K is not None:
            K_target = int(subsample_K)
    # --- loop windows ---
    n_piv = len(bounds)
    si_rows = []
    K_valid = np.full(n_piv, np.nan, float)
    coverage = np.full(n_piv, np.nan, float)
    for w_i, (s_win, e_win) in enumerate(bounds):
        if e_win <= s_win:
            si_rows.append(np.full(M, np.nan, float))
            continue
        # counts
        if time_window == 0:
            Q_window = spike_counts_filt[s_win:e_win, :]
        else:
            frames_in_window = np.arange(s_win, e_win, dtype=int)
            Q_window = build_count_matrix(
                spike_counts,
                raw_counts,
                selected_df,
                frames=frames_in_window,
                time_window=time_window,
                neuron_ids=neuron_ids,
            ).T
        labels_window = labels_all[s_win:e_win]
        bc = np.bincount(labels_window, minlength=n_bins).astype(int)
        visited = bc >= min_frames
        if common_support is not None:
            visited &= common_support
        # bookkeeping
        K = int(np.sum(visited))
        K_valid[w_i] = float(K)
        coverage[w_i] = float(np.mean(visited))
        if K == 0:
            si_rows.append(np.full(M, np.nan, float))
            continue
        # -------- Option 3: matched subsampling --------
        if subsample_mode == "matched":
            if K_target is None or K_target <= 0:
                si_rows.append(np.full(M, np.nan, float))
                continue
            if subsample_target == "uniform":
                p_target = np.zeros(n_bins, float)
                p_target[visited] = 1.0
                support_bins = visited.copy()
            elif subsample_target == "reference":
                if reference_prior is None:
                    raise ValueError(
                        "subsample_target='reference' requires reference_prior (or reference_prior=None to compute it)"
                    )
                p_target = np.asarray(reference_prior, float).copy()
                support_bins = visited & (p_target > 0)
            else:
                raise ValueError("subsample_target must be 'uniform'|'reference'")
            if not np.any(support_bins):
                si_rows.append(np.full(M, np.nan, float))
                continue
            si_accum = np.zeros(M, float)
            n_ok = 0
            for _ in range(int(subsample_n_resamples)):
                idx_sub = _matched_subsample_indices(
                    labels_window,
                    support_bins,
                    p_target,
                    K_target,
                    rng,
                    replace_if_needed=subsample_replace_if_needed,
                )
                if idx_sub.size == 0:
                    continue
                Q_sub = Q_window[idx_sub, :]
                lab_sub = labels_window[idx_sub]
                bc_sub = np.bincount(lab_sub, minlength=n_bins).astype(float)
                # in subsample we allow bins that appear at least once
                visited_sub = bc_sub >= 1
                if common_support is not None:
                    visited_sub &= common_support
                lam_sub, _, _ = _compute_lam_visited(
                    Q_sub, lab_sub, n_bins, dt_eff_tuning, eps
                )
                lam_sub[~visited_sub, :] = 0.0
                if filter_ratemap and n_bins_y > 1:
                    for j in range(M):
                        lam_sub[:, j] = gaussian_filter(
                            lam_sub[:, j].reshape(n_bins_y, n_bins_x),
                            sigma=sigma,
                            mode="constant",
                            cval=0.0,
                        ).ravel()
                if subsample_target == "uniform":
                    prior_use = _make_prior(bc_sub, visited_sub, "uniform", eps=eps)
                else:
                    prior_use = _make_prior(
                        bc_sub,
                        visited_sub,
                        "reference",
                        eps=eps,
                        reference_prior=reference_prior,
                    )
                si_accum += _skaggs_info(lam_sub, prior_use, eps)
                n_ok += 1
            si_rows.append(si_accum / max(n_ok, 1))
            continue
        # -------- Options 1/2 (+ reference unconditional) --------
        lam, bc_float, _ = _compute_lam_visited(
            Q_window, labels_window, n_bins, dt_eff_tuning, eps
        )
        # smoothing (only meaningful in 2D)
        if filter_ratemap and n_bins_y > 1:
            for j in range(M):
                lam[:, j] = gaussian_filter(
                    lam[:, j].reshape(n_bins_y, n_bins_x),
                    sigma=sigma,
                    mode="constant",
                    cval=0.0,
                ).ravel()
        if prior_mode != "reference":
            # empirical/uniform always operate on visited bins and renorm
            prior_use = _make_prior(bc_float, visited, prior_mode, eps=eps)
            lam_use = lam.copy()
            lam_use[~visited, :] = 0.0
            si_rows.append(_skaggs_info(lam_use, prior_use, eps))
        else:
            if reference_prior is None:
                raise ValueError(
                    "prior_mode='reference' requires reference_prior (or reference_prior=None to compute it)"
                )
            if reference_scope == "conditional":
                # safe: restrict to visited bins then renormalize reference prior on that support
                prior_use = _make_prior(
                    bc_float,
                    visited,
                    "reference",
                    eps=eps,
                    reference_prior=reference_prior,
                )
                lam_use = lam.copy()
                lam_use[~visited, :] = 0.0
                si_rows.append(_skaggs_info(lam_use, prior_use, eps))
            elif reference_scope == "unconditional":
                # "whole-reference support": use reference prior across its support;
                # for bins in prior support but not visited, impute lambda explicitly.
                prior_support = reference_prior > 0
                if common_support is not None:
                    prior_support &= (
                        common_support  # optional: keep the same support constraint
                    )
                # prior is the (renormalized) reference prior on prior_support
                p = np.zeros(n_bins, float)
                p[prior_support] = reference_prior[prior_support]
                ps = p.sum()
                prior_use = (p / ps) if ps > eps else p
                visited_for_lam = visited.copy()
                # ensure we can assign lambda to all bins that have prior mass:
                lam_use = _impute_lam_for_unvisited(
                    lam,
                    visited_for_lam,
                    prior_support,
                    Q_window,
                    dt_eff_tuning,
                    eps,
                    policy=unvisited_lam_policy,
                )
                # for bins outside prior_support, set lam to 0 (they have zero prior mass anyway)
                lam_use[~prior_support, :] = 0.0
                si_rows.append(_skaggs_info(lam_use, prior_use, eps))
            else:
                raise ValueError(
                    "reference_scope must be 'conditional' or 'unconditional'"
                )
    # --- assemble ---
    si_matrix = np.asarray(si_rows, dtype=float)  # (n_pivots, M)
    si_mean = np.nanmean(si_matrix, axis=1) if si_matrix.size else np.array([])
    si_std = np.nanstd(si_matrix, axis=1) if si_matrix.size else np.array([])
    # --- normalized SI (what we discussed): divide by log2(K_valid) ---
    denom = np.log2(K_valid)
    denom[
        (~np.isfinite(denom)) | (denom <= 0)
    ] = np.nan  # avoid divide-by-0 / log2(1)=0
    si_matrix_norm = si_matrix / denom[:, None]
    si_mean_norm = (
        np.nanmean(si_matrix_norm, axis=1) if si_matrix_norm.size else np.array([])
    )
    si_std_norm = (
        np.nanstd(si_matrix_norm, axis=1) if si_matrix_norm.size else np.array([])
    )
    meta = dict(
        n_bins=n_bins,
        x_edges=x_edges,
        y_edges=y_edges,
        win_len_sec=win_len_sec,
        pivot_step_sec=pivot_step_sec,
        min_time_per_bin_sec=min_time_per_bin_sec,
        dt_eff_tuning=dt_eff_tuning,
        min_frames=min_frames,
        prior_mode=prior_mode,
        reference_scope=reference_scope if prior_mode == "reference" else None,
        unvisited_lam_policy=unvisited_lam_policy
        if (prior_mode == "reference" and reference_scope == "unconditional")
        else None,
        support_mode=support_mode,
        common_support_frac=common_support_frac
        if support_mode == "common_fraction"
        else None,
        subsample_mode=subsample_mode,
        subsample_target=subsample_target if subsample_mode == "matched" else None,
        subsample_K=K_target if subsample_mode == "matched" else None,
        subsample_n_resamples=subsample_n_resamples
        if subsample_mode == "matched"
        else None,
        K_valid=K_valid,
        coverage=coverage,
        common_support=common_support,
    )
    return dict(
        pivot_bounds=bounds,
        pivot_times=pivot_times,
        pivot_times_maze=pivot_times_maze,
        si_matrix=si_matrix,
        si_mean=si_mean,
        si_std=si_std,
        si_matrix_norm=si_matrix_norm,
        si_mean_norm=si_mean_norm,
        si_std_norm=si_std_norm,
        kept_neuron_ids=neuron_ids,
        meta=meta,
    )


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def window_si_null(
    Q_window,
    labels_window,
    n_bins,
    n_bins_x,
    n_bins_y,
    dt_eff_tuning,
    min_frames=1,
    eps=1e-12,
    valid_bins=None,  # if None -> computed from labels_window and min_frames
    prior_mode="empirical",
    reference_prior=None,
    filter_ratemap=False,
    sigma=1.0,
    n_shuffles=200,
    shuffle_mode="circular",  # 'circular' | 'permute_labels'
    shift_min_sec=10.0,
    per_neuron=True,
    rng=None,
):
    """
    Returns SI_null: (n_shuffles, M)
    Keeps occupancy fixed (labels unchanged) for circular shift; permute_labels keeps histogram fixed too.
    """
    if rng is None:
        rng = np.random.default_rng()
    Q_window = np.asarray(Q_window)
    labels_window = np.asarray(labels_window, dtype=int)
    T, M = Q_window.shape
    if valid_bins is None:
        bc = np.bincount(labels_window, minlength=n_bins).astype(int)
        valid_bins = bc >= int(min_frames)
    # prior is the same across shuffles (because occupancy histogram is unchanged)
    bc0 = np.bincount(labels_window, minlength=n_bins).astype(float)
    prior_use = _make_prior(
        bc0, valid_bins, mode=prior_mode, eps=eps, reference_prior=reference_prior
    )
    # shift in frames
    min_shift_frames = (
        int(np.ceil(float(shift_min_sec) / float(dt_eff_tuning)))
        if shift_min_sec is not None
        else 1
    )
    min_shift_frames = max(min_shift_frames, 1)
    out = np.zeros((int(n_shuffles), M), dtype=float)
    for k in range(int(n_shuffles)):
        if shuffle_mode == "circular":
            Qs = _circular_shift_Q(
                Q_window, rng, min_shift_frames=min_shift_frames, per_neuron=per_neuron
            )
            labs = labels_window
        elif shuffle_mode == "permute_labels":
            Qs = Q_window
            labs = _permute_labels(labels_window, rng)
        else:
            raise ValueError("shuffle_mode must be 'circular'|'permute_labels'")
        lam, _ = _compute_lam(Qs, labs, n_bins, dt_eff_tuning, eps, valid_bins)
        if filter_ratemap and n_bins_y > 1:
            for j in range(M):
                lam[:, j] = gaussian_filter(
                    lam[:, j].reshape(n_bins_y, n_bins_x),
                    sigma=sigma,
                    mode="constant",
                    cval=0.0,
                ).ravel()
        out[k, :] = _skaggs_info(lam, prior_use, eps)
    return out


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def pivot_window_spatial_information_shuffle_z(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    win_len_sec=10 * 60,
    pivot_step_sec=60,
    pivot_time_axis="selected",
    frame_dt_sec=None,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    neuron_ids=None,
    min_spikes_per_min=None,
    rate_source="spike",
    random_state=0,
    # must match the SI settings you want to evaluate
    prior_mode="empirical",
    reference_prior=None,
    support_mode="window",
    common_support_frac=1.0,
    subsample_mode="none",
    subsample_target="uniform",
    subsample_K=None,
    subsample_n_resamples=50,
    subsample_replace_if_needed=True,
    # shuffle settings
    n_shuffles=200,
    shuffle_mode="circular",
    shift_min_sec=10.0,
    per_neuron=True,
):
    """
    Computes real SI via pivot_window_spatial_information_fast(...), then computes a null per window and returns z-scores.
    NOTE:
      - If subsample_mode='matched', the real SI is an average over resamples.
        For a strict null of that estimator you'd ideally also subsample inside each shuffle (expensive).
        Here we still shuffle the full window using the same prior/support settings, which is a strong
        occupancy-preserving diagnostic in practice. If you want the strict estimator-matched null,
        ask and I will provide the heavier version.
    """
    rng = np.random.default_rng(random_state)
    real = pivot_window_spatial_information_fast(
        spike_counts,
        raw_counts,
        selected_df,
        positions,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        win_len_sec=win_len_sec,
        pivot_step_sec=pivot_step_sec,
        pivot_time_axis=pivot_time_axis,
        frame_dt_sec=frame_dt_sec,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        min_time_per_bin_sec=min_time_per_bin_sec,
        eps=eps,
        neuron_ids=neuron_ids,
        min_spikes_per_min=min_spikes_per_min,
        rate_source=rate_source,
        random_state=random_state,
        prior_mode=prior_mode,
        reference_prior=reference_prior,
        support_mode=support_mode,
        common_support_frac=common_support_frac,
        subsample_mode=subsample_mode,
        subsample_target=subsample_target,
        subsample_K=subsample_K,
        subsample_n_resamples=subsample_n_resamples,
        subsample_replace_if_needed=subsample_replace_if_needed,
    )
    bounds = real["pivot_bounds"]
    si_real = real["si_matrix"]  # (n_pivots, M)
    kept = real["kept_neuron_ids"]
    M = int(len(kept))
    n_piv = int(len(bounds))
    # re-create discretized labels and compute dt_eff/min_frames consistent with real
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    labels_all, _, _ = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_all = np.asarray(labels_all, dtype=int)
    dt_eff_tuning = float(dt) * float(time_window + 1)
    min_frames = int(np.ceil(float(min_time_per_bin_sec) / dt_eff_tuning))
    # reference prior if needed
    if prior_mode == "reference" or reference_prior is not None:
        if reference_prior is None:
            reference_prior, _ = _compute_reference_prior_from_labels(
                labels_all, n_bins_x * n_bins_y, min_frames, eps=eps
            )
        else:
            reference_prior = np.asarray(reference_prior, float)
    # common support (if used)
    common_support = real["meta"].get("common_support", None)
    # build filtered counts for windows
    spike_counts_filt = spike_counts[:, kept]
    z = np.full((n_piv, M), np.nan, float)
    null_mean = np.full((n_piv, M), np.nan, float)
    null_std = np.full((n_piv, M), np.nan, float)
    for w_i, (s_win, e_win) in enumerate(bounds):
        if e_win <= s_win:
            continue
        # window data
        if time_window == 0:
            Q_window = spike_counts_filt[s_win:e_win, :]
        else:
            frames_in_window = np.arange(s_win, e_win, dtype=int)
            Q_window = build_count_matrix(
                spike_counts,
                raw_counts,
                selected_df,
                frames=frames_in_window,
                time_window=time_window,
                neuron_ids=kept,
            ).T
        labels_window = labels_all[s_win:e_win]
        bc = np.bincount(labels_window, minlength=n_bins_x * n_bins_y).astype(int)
        valid_bins = bc >= min_frames
        if common_support is not None:
            valid_bins &= np.asarray(common_support, bool)
        if not np.any(valid_bins):
            continue
        si_null = window_si_null(
            Q_window,
            labels_window,
            n_bins=n_bins_x * n_bins_y,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            dt_eff_tuning=dt_eff_tuning,
            min_frames=min_frames,
            eps=eps,
            valid_bins=valid_bins,
            prior_mode=prior_mode,
            reference_prior=reference_prior,
            filter_ratemap=filter_ratemap,
            sigma=sigma,
            n_shuffles=n_shuffles,
            shuffle_mode=shuffle_mode,
            shift_min_sec=shift_min_sec,
            per_neuron=per_neuron,
            rng=rng,
        )
        mu = np.nanmean(si_null, axis=0)
        sd = np.nanstd(si_null, axis=0) + eps
        null_mean[w_i, :] = mu
        null_std[w_i, :] = sd
        z[w_i, :] = (si_real[w_i, :] - mu) / sd
    out = dict(real=real, z_matrix=z, null_mean=null_mean, null_std=null_std)
    return out


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: spatial]
def pivot_window_spatial_information(
    # Data inputs (single session)
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    # Spatial binning (1D or 2D)
    n_bins_x=15,
    n_bins_y=15,
    # Timing parameters
    dt=0.04,
    time_window=0,
    # Window sliding parameters
    win_len_sec=10 * 60,  # 10 minutes window length
    pivot_step_sec=60,  # 1 minute step
    pivot_time_axis="selected",  # 'selected' | 'frame' | 'maze'
    frame_dt_sec=None,
    # SI computation parameters
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,  # Lower threshold for windowed analysis
    eps=1e-12,
    # Neuron filtering
    neuron_ids=None,
    min_spikes_per_min=None,
    rate_source="spike",
    random_state=0,
):
    """
    Compute spatial information within sliding time windows across a session.
    Parameters
    ----------
    spike_counts : (T, N) array
        Spike counts per frame per neuron
    raw_counts : (T, N) array
        Raw calcium counts per frame per neuron
    selected_df : DataFrame
        Must contain 'frame' column; optionally 'maze_time' if pivot_time_axis='maze'
    positions : (T, 2) or (T,) or (T, 1) array
        For 2D: (T, 2) with n_bins_x Ã— n_bins_y grid
        For 1D: (T,) or (T, 1) with n_bins_y=1 and n_bins_x = # spatial bins
    win_len_sec : float
        Length of sliding window in seconds
    pivot_step_sec : float
        Step size for sliding window in seconds
    pivot_time_axis : str
        Time axis to use: 'selected' (row index), 'frame', or 'maze'
    Returns
    -------
    dict with keys:
        pivot_bounds : list of (start, end) tuples
            Frame index boundaries for each window in selected_df rows
        pivot_times : (n_pivots,) array
            Center time of each window on chosen axis (seconds)
        pivot_times_maze : (n_pivots,) array or None
            Center time on maze-time axis if available
        si_matrix : (n_pivots, n_neurons) array
            Spatial information per neuron per window
        si_mean : (n_pivots,) array
            Mean SI across neurons per window
        si_std : (n_pivots,) array
            Std SI across neurons per window
        kept_neuron_ids : array
            IDs of neurons included after filtering
        meta : dict
            Additional metadata (grid info, parameters, etc.)
    """
    T_sel, N_total = spike_counts.shape
    assert "frame" in selected_df.columns
    assert T_sel == len(selected_df)
    # --- 0) Position dimensionality handling (1D vs 2D) ---
    pos = np.asarray(positions)
    is_1d = (pos.ndim == 1) or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError(
            "pivot_window_spatial_information: for 1D positions (shape (T,) or (T,1)), "
            "please set n_bins_y=1 and use n_bins_x for the number of spatial bins."
        )
    # Choose neuron set
    if neuron_ids is None:
        neuron_ids = np.arange(N_total, dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    M = int(neuron_ids.size)
    # Discretize spatial positions (1D or 2D)
    labels_pos, x_edges, y_edges = discretize_positions(pos, n_bins_x, n_bins_y)
    n_bins = int(n_bins_x * n_bins_y)
    # --- Time axis setup ---
    frame_dt_used = None
    frame_dt_inferred = False
    pivot_times_maze = None
    if pivot_time_axis == "selected":
        t_sel = np.arange(T_sel, dtype=float) * float(dt)
    elif pivot_time_axis == "frame":
        frame_idx = selected_df["frame"].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)
            frame_dt_inferred = True
        else:
            frame_dt_inferred = False
        t_sel = frame_idx * float(frame_dt_sec)
        frame_dt_used = float(frame_dt_sec)
    elif pivot_time_axis == "maze":
        if "maze_time" not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time']")
        t_sel = selected_df["maze_time"].to_numpy(dtype=float)
    else:
        raise ValueError("pivot_time_axis must be 'selected'|'frame'|'maze'")
    dt_eff = float(np.mean(np.diff(t_sel))) if t_sel.size > 1 else float(dt)
    # Store maze_time if available (for reference even if not used as pivot axis)
    if "maze_time" in selected_df.columns and pivot_time_axis != "maze":
        maze_times_all = selected_df["maze_time"].to_numpy(dtype=float)
    else:
        maze_times_all = None
    # --- Neuron filtering by overall rate ---
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, float).sum(axis=0)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        return (
            total_events / total_minutes if total_minutes > 0 else np.zeros(C.shape[1])
        )
    if rate_source == "spike":
        C_for_rate = spike_counts
        dt_for_rate = dt
    elif rate_source == "raw":
        frames_cont = selected_df["frame"].to_numpy().astype(int)
        C_for_rate = raw_counts[frames_cont, :]
        dt_for_rate = dt_eff
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'")
    rate_per_min = _spikes_per_min(C_for_rate, dt_for_rate)
    if min_spikes_per_min is not None:
        keep = rate_per_min[neuron_ids] >= float(min_spikes_per_min)
        neuron_ids = neuron_ids[keep]
        M = neuron_ids.size
    if M == 0:
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([]),
            pivot_times_maze=None,
            si_matrix=np.zeros((0, 0)),
            si_mean=np.array([]),
            si_std=np.array([]),
            kept_neuron_ids=neuron_ids,
            meta=dict(
                n_bins=n_bins,
                x_edges=x_edges,
                y_edges=y_edges,
                win_len_sec=win_len_sec,
                pivot_step_sec=pivot_step_sec,
                min_time_per_bin_sec=min_time_per_bin_sec,
            ),
        )
    # --- Compute pivot window boundaries ---
    t_min, t_max = t_sel[0], t_sel[-1]
    half_win = float(win_len_sec) / 2.0
    # Generate pivot center times
    pivot_centers = []
    t = t_min + half_win
    while t + half_win <= t_max:
        pivot_centers.append(t)
        t += float(pivot_step_sec)
    if len(pivot_centers) == 0:
        # Window too large for session
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([]),
            pivot_times_maze=None,
            si_matrix=np.zeros((0, M)),
            si_mean=np.array([]),
            si_std=np.array([]),
            kept_neuron_ids=neuron_ids,
            meta=dict(
                n_bins=n_bins,
                x_edges=x_edges,
                y_edges=y_edges,
                win_len_sec=win_len_sec,
                pivot_step_sec=pivot_step_sec,
                min_time_per_bin_sec=min_time_per_bin_sec,
            ),
        )
    pivot_times = np.array(pivot_centers, dtype=float)
    # Convert time boundaries to frame indices
    bounds = []
    maze_time_centers = []
    for tc in pivot_centers:
        t_start = tc - half_win
        t_end = tc + half_win
        # Find frames within this time window
        mask = (t_sel >= t_start) & (t_sel < t_end)
        indices = np.where(mask)[0]
        if indices.size == 0:
            bounds.append((0, 0))  # Empty window
            maze_time_centers.append(np.nan)
        else:
            s, e = int(indices[0]), int(indices[-1]) + 1
            bounds.append((s, e))
            # Compute maze time center if available
            if maze_times_all is not None:
                maze_time_centers.append(float(np.mean(maze_times_all[s:e])))
            else:
                maze_time_centers.append(np.nan)
    if maze_times_all is not None:
        pivot_times_maze = np.array(maze_time_centers)
    # --- Tuning computation function (choose peak vs standard) ---
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    dt_eff_tuning = dt * (time_window + 1)
    # --- Loop over pivot windows and compute SI ---
    si_rows = []
    for (s_win, e_win) in bounds:
        if e_win <= s_win:
            # Empty window
            si_rows.append(np.full(M, np.nan, dtype=float))
            continue
        # Extract data for this window
        frames_in_window = np.arange(s_win, e_win, dtype=int)
        Q_window = build_count_matrix(
            spike_counts,
            raw_counts,
            selected_df,
            frames=frames_in_window,
            time_window=time_window,
            neuron_ids=neuron_ids,
        ).T  # (T_window, M)
        # Positions and labels for this window
        pos_window = pos[s_win:e_win]
        labels_window = discretize_positions(pos_window, n_bins_x, n_bins_y)[0]
        # Compute rate map and occupancy prior
        lam_window, prior_window = tuner(
            Q_window,
            labels_window,
            None,
            n_bins,
            dt_eff_tuning,
            filter_ratemap,
            sigma,
            min_time=min_time_per_bin_sec,
            eps=eps,
            decode_head="none",
        )
        # Compute spatial information using Skaggs formula
        SI_window = _skaggs_info(lam_window, prior_window, eps)
        si_rows.append(SI_window)
    # --- Package results ---
    si_matrix = np.vstack(si_rows) if len(si_rows) > 0 else np.zeros((0, M))
    si_mean = np.nanmean(si_matrix, axis=1) if si_matrix.size > 0 else np.array([])
    si_std = np.nanstd(si_matrix, axis=1) if si_matrix.size > 0 else np.array([])
    meta = dict(
        x_edges=x_edges,
        y_edges=y_edges,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        n_bins=n_bins,
        kept_neuron_ids=neuron_ids,
        rate_per_min=rate_per_min,
        threshold=min_spikes_per_min,
        rate_source=rate_source,
        pivot_time_axis=pivot_time_axis,
        frame_dt_used=frame_dt_used,
        frame_dt_inferred=frame_dt_inferred,
        pivot_times_maze=pivot_times_maze,
        min_time_per_bin_sec=min_time_per_bin_sec,
        win_len_sec=win_len_sec,
        pivot_step_sec=pivot_step_sec,
    )
    return dict(
        pivot_bounds=bounds,
        pivot_times=pivot_times,
        pivot_times_maze=pivot_times_maze,
        si_matrix=si_matrix,
        si_mean=si_mean,
        si_std=si_std,
        kept_neuron_ids=neuron_ids,
        meta=meta,
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def within_session_pv_temporal_similarity(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    # maps/modes
    mode="spatial",  # 'spatial' | 'level' | 'hd' | 'joint_pos' | 'joint_lvl'
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,  # passed to build_count_matrix
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    neuron_ids=None,
    random_state=0,
    # pivot windows
    win_len_sec=20 * 60,  # 20 min per side
    pivot_step_sec=60,  # 1 min stride
    pivot_time_axis="selected",  # 'selected' | 'frame' | 'maze'
    frame_dt_sec=None,  # if None and axis='frame', falls back to dt
    # PV similarity
    metric="pearson",  # 'pearson'|'spearman'|'cosine'
    normalization="auto",  # 'none'|'zscore'|'l2'|'auto' (row-wise across neurons)
    intersect_mask=True,
    eps=1e-12,
    # frequent-visit / occupancy control (SECONDS)
    min_time_per_bin_sec=5.0,
    # neuron filtering
    min_spikes_per_min=None,  # None or float
    rate_source="spike",  # 'spike' | 'raw'
):
    """
    For each pivot time t*, compute BEFORE [t*-W, t*) and AFTER [t*, t*+W) PVs per bin/state,
    then correlate BEFORE vs AFTER PVs (row-wise across neurons). Bins are included only if
    their OCCUPANCY (in seconds) meets `min_time_per_bin_sec` in BOTH windows when
    intersect_mask=True (else union).
    1D vs 2D positions
    ------------------
    - 2D: positions shape (T, 2), grid n_bins_x Ã— n_bins_y
    - 1D: positions shape (T,) or (T, 1), REQUIRE n_bins_y == 1 and n_bins_x = # spatial bins
    """
    T_sel, N_total = spike_counts.shape
    assert "frame" in selected_df.columns
    assert T_sel == len(selected_df)
    # --- 0) Position dimensionality handling (1D vs 2D) ---
    pos = np.asarray(positions)
    is_1d = (pos.ndim == 1) or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError(
            "within_session_pv_temporal_similarity: for 1D positions (shape (T,) or (T,1)), "
            "please set n_bins_y=1 and use n_bins_x for the number of spatial bins."
        )
    if neuron_ids is None:
        neuron_ids = np.arange(N_total, dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    M = int(neuron_ids.size)
    # labels (1D or 2D handled inside discretize_positions)
    labels_pos, x_edges, y_edges = discretize_positions(pos, n_bins_x, n_bins_y)
    n_bins_pos = n_bins_x * n_bins_y
    labels_lvl = None
    uniq_lvls = None
    if mode in ("level", "joint_lvl"):
        v = selected_df[level_col].values
        uniq_lvls = np.unique(v)
        lvl_map = {val: i for i, val in enumerate(uniq_lvls)}
        labels_lvl = np.array([lvl_map[val] for val in v])
        nL = len(uniq_lvls)
    labels_hd = None
    uniq_hd = None
    if mode in ("hd", "joint_pos"):
        h = selected_df[head_col].values
        uniq_hd = np.unique(h)
        hd_map = {val: i for i, val in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[val] for val in h])
        nH = len(uniq_hd)
    # time axis
    if pivot_time_axis == "selected":
        t_sel = np.arange(T_sel, dtype=float) * float(dt)
        frame_dt_used = None
        frame_dt_inferred = False
    elif pivot_time_axis == "frame":
        frame_idx = selected_df["frame"].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)
            frame_dt_inferred = True
        else:
            frame_dt_inferred = False
        t_sel = frame_idx * float(frame_dt_sec)
        frame_dt_used = float(frame_dt_sec)
    elif pivot_time_axis == "maze":
        if "maze_time" not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time']")
        t_sel = selected_df["maze_time"].to_numpy(dtype=float)
        frame_dt_used = None
        frame_dt_inferred = False
    else:
        raise ValueError("pivot_time_axis must be 'selected'|'frame'|'maze'")
    dt_eff = float(np.mean(np.diff(t_sel))) if t_sel.size > 1 else float(dt)
    # neuron filtering by overall rate
    def _spikes_per_min(C, dt_sec):
        total_events = np.asarray(C, float).sum(axis=0)
        total_minutes = (C.shape[0] * float(dt_sec)) / 60.0
        return (
            total_events / total_minutes if total_minutes > 0 else np.zeros(C.shape[1])
        )
    if rate_source == "spike":
        C_for_rate = spike_counts
        dt_for_rate = dt
    elif rate_source == "raw":
        frames_cont = selected_df["frame"].to_numpy().astype(int)
        C_for_rate = raw_counts[frames_cont, :]
        dt_for_rate = dt_eff
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    rate_per_min = _spikes_per_min(C_for_rate, dt_for_rate)
    if min_spikes_per_min is not None:
        thr = float(min_spikes_per_min)
        neuron_ids = neuron_ids[rate_per_min[neuron_ids] >= thr]
        M = int(neuron_ids.size)
        if M == 0:
            return dict(
                pivot_bounds=np.zeros((0, 2, 2), int),
                pivot_times=np.zeros((0,), float),
                pivot_times_maze=None,
                sim_per_bin=np.zeros((0, 0), float),
                sim_mean=np.zeros((0,), float),
                mode=mode,
                metric=metric,
                normalization=normalization,
                meta=dict(
                    x_edges=x_edges,
                    y_edges=y_edges,
                    n_bins_x=n_bins_x,
                    n_bins_y=n_bins_y,
                    kept_neuron_ids=np.array([], int),
                    rate_per_min=rate_per_min,
                    threshold=thr,
                    rate_source=rate_source,
                    pivot_time_axis=pivot_time_axis,
                    frame_dt_used=frame_dt_used if pivot_time_axis == "frame" else None,
                    frame_dt_inferred=frame_dt_inferred
                    if pivot_time_axis == "frame"
                    else False,
                    min_time_per_bin_sec=min_time_per_bin_sec,
                ),
            )
    # pivots/bounds
    W = float(win_len_sec)
    step_sec = float(pivot_step_sec)
    if (t_sel[-1] - t_sel[0]) <= 2 * W:
        raise ValueError("Session too short for chosen win_len_sec on this axis.")
    t0 = t_sel[0] + W
    t1 = t_sel[-1] - W
    pivot_times = np.arange(t0, t1 + 1e-9, step_sec, dtype=float)
    n_pivots = pivot_times.size
    bounds = np.zeros((n_pivots, 2, 2), dtype=int)
    for j, tp in enumerate(pivot_times):
        s_b = int(np.searchsorted(t_sel, tp - W, side="left"))
        e_b = int(np.searchsorted(t_sel, tp, side="left"))
        s_a = int(np.searchsorted(t_sel, tp, side="left"))
        e_a = int(np.searchsorted(t_sel, tp + W, side="left"))
        bounds[j, 0] = (s_b, e_b)
        bounds[j, 1] = (s_a, e_a)
    # maze-time at pivot centers
    if "maze_time" in selected_df.columns and n_pivots > 0:
        maze_t = selected_df["maze_time"].to_numpy(dtype=float)
        pivot_times_maze = maze_t[bounds[:, 1, 0]]  # s_a
    else:
        pivot_times_maze = None
    # PV normalization choice
    if normalization == "auto":
        row_norm_mode = {"cosine": "l2", "pearson": "zscore", "spearman": "zscore"}.get(
            metric, "l2"
        )
    else:
        row_norm_mode = normalization
    def _rowwise_z(X):
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        return (X - mu) / (sd + eps)
    def _rowwise_l2(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + eps)
    # counts (T_sel x M)
    Q_full = build_count_matrix(
        spike_counts,
        raw_counts,
        selected_df,
        frames=np.arange(T_sel),
        time_window=time_window,
        neuron_ids=neuron_ids,
    ).T
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    # helpers: occupancy in seconds (NOT priors)
    def occ_secs_spatial(lbl, nB):
        return np.bincount(lbl, minlength=nB).astype(float) * dt_eff
    def occ_secs_level(lbl, nL):
        return np.bincount(lbl, minlength=nL).astype(float) * dt_eff
    def occ_secs_hd(lbl, nH):
        return np.bincount(lbl, minlength=nH).astype(float) * dt_eff
    def occ_secs_joint_pos(pos_lbl, hd_lbl, nB, nH):
        idx = pos_lbl.astype(int) * nH + hd_lbl.astype(int)
        occ = np.bincount(idx, minlength=nB * nH).astype(float) * dt_eff
        return occ.reshape(nB, nH)
    def occ_secs_joint_lvl(pos_lbl, lvl_lbl, nB, nL):
        out = np.zeros((nL, nB), float)
        for L in range(nL):
            m = lvl_lbl == L
            if m.any():
                out[L] = np.bincount(pos_lbl[m], minlength=nB).astype(float) * dt_eff
        return out
    # output alloc
    if mode == "spatial":
        K_full = n_bins_pos
    elif mode == "level":
        K_full = nL
    elif mode == "hd":
        K_full = nH
    elif mode == "joint_pos":
        K_full = n_bins_pos * nH
    elif mode == "joint_lvl":
        K_full = nL * n_bins_pos
    else:
        raise ValueError(f"Unknown mode: {mode}")
    sim_per_bin = np.full((n_pivots, K_full), np.nan, float)
    sim_mean = np.full(n_pivots, np.nan, float)
    for j in range(n_pivots):
        s_b, e_b = bounds[j, 0]
        s_a, e_a = bounds[j, 1]
        Qb = Q_full[s_b:e_b]
        Qa = Q_full[s_a:e_a]
        yb_pos = labels_pos[s_b:e_b]
        ya_pos = labels_pos[s_a:e_a]
        yb_lvl = labels_lvl[s_b:e_b] if labels_lvl is not None else None
        ya_lvl = labels_lvl[s_a:e_a] if labels_lvl is not None else None
        yb_hd = labels_hd[s_b:e_b] if labels_hd is not None else None
        ya_hd = labels_hd[s_a:e_a] if labels_hd is not None else None
        if mode == "spatial":
            occ_b = occ_secs_spatial(yb_pos, n_bins_pos)
            occ_a = occ_secs_spatial(ya_pos, n_bins_pos)
            Ab, _ = tuner(
                Qb,
                yb_pos,
                None,
                n_bins_pos,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            Aa, _ = tuner(
                Qa,
                ya_pos,
                None,
                n_bins_pos,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            keep = (
                (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
                if intersect_mask
                else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            )
            A = Ab[keep]
            B = Aa[keep]
        elif mode == "level":
            occ_b = occ_secs_level(yb_lvl, nL)
            occ_a = occ_secs_level(ya_lvl, nL)
            Ab, _ = tuner(
                Qb,
                yb_lvl,
                None,
                nL,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            Aa, _ = tuner(
                Qa,
                ya_lvl,
                None,
                nL,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="none",
            )
            keep = (
                (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
                if intersect_mask
                else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            )
            A = Ab[keep]
            B = Aa[keep]
        elif mode == "hd":
            occ_b = occ_secs_hd(yb_hd, nH)
            occ_a = occ_secs_hd(ya_hd, nH)
            _, _, Ab, _ = tuner(
                Qb,
                yb_pos,
                yb_hd,
                n_bins_pos,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="independent",
            )
            _, _, Aa, _ = tuner(
                Qa,
                ya_pos,
                ya_hd,
                n_bins_pos,
                dt_eff,
                False,
                0.0,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="independent",
            )
            keep = (
                (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
                if intersect_mask
                else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            )
            A = Ab[keep]
            B = Aa[keep]
        elif mode == "joint_pos":
            occ_b = occ_secs_joint_pos(yb_pos, yb_hd, n_bins_pos, nH)
            occ_a = occ_secs_joint_pos(ya_pos, ya_hd, n_bins_pos, nH)
            AJ_b, _ = tuner(
                Qb,
                yb_pos,
                yb_hd,
                n_bins_pos,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="joint",
            )
            AJ_a, _ = tuner(
                Qa,
                ya_pos,
                ya_hd,
                n_bins_pos,
                dt_eff,
                filter_ratemap,
                sigma,
                min_time=min_time_per_bin_sec,
                eps=eps,
                decode_head="joint",
            )
            keep = (
                (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
                if intersect_mask
                else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            )
            A = AJ_b[keep].reshape(-1, AJ_b.shape[-1])
            B = AJ_a[keep].reshape(-1, AJ_a.shape[-1])
        elif mode == "joint_lvl":
            occ_b = occ_secs_joint_lvl(yb_pos, yb_lvl, n_bins_pos, nL)
            occ_a = occ_secs_joint_lvl(ya_pos, ya_lvl, n_bins_pos, nL)
            lamL_b = []
            lamL_a = []
            for L in range(nL):
                mb = yb_lvl == L
                ma = ya_lvl == L
                if mb.any():
                    lb, _ = tuner(
                        Qb[mb],
                        yb_pos[mb],
                        None,
                        n_bins_pos,
                        dt_eff,
                        filter_ratemap,
                        sigma,
                        min_time=min_time_per_bin_sec,
                        eps=eps,
                        decode_head="none",
                    )
                else:
                    lb = np.zeros((n_bins_pos, M))
                if ma.any():
                    la, _ = tuner(
                        Qa[ma],
                        ya_pos[ma],
                        None,
                        n_bins_pos,
                        dt_eff,
                        filter_ratemap,
                        sigma,
                        min_time=min_time_per_bin_sec,
                        eps=eps,
                        decode_head="none",
                    )
                else:
                    la = np.zeros((n_bins_pos, M))
                lamL_b.append(lb)
                lamL_a.append(la)
            Ab = np.stack(lamL_b, 0)
            Aa = np.stack(lamL_a, 0)
            keep = (
                (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
                if intersect_mask
                else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            )
            A = Ab[keep].reshape(-1, M)
            B = Aa[keep].reshape(-1, M)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if A.size == 0 or B.size == 0:
            continue
        # row-wise normalization across neurons
        if metric == "spearman":
            A = np.apply_along_axis(rankdata, 1, A)
            B = np.apply_along_axis(rankdata, 1, B)
            if normalization == "auto" or row_norm_mode == "zscore":
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == "l2":
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            denom = max(A.shape[1] - 1, 1) + eps
            s = np.sum(A * B, axis=1) / denom
        elif metric == "pearson":
            if normalization == "auto" or row_norm_mode == "zscore":
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == "l2":
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            denom = max(A.shape[1] - 1, 1) + eps
            s = np.sum(A * B, axis=1) / denom
        else:  # cosine
            if row_norm_mode == "zscore":
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == "l2":
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            na = np.linalg.norm(A, axis=1)
            nb = np.linalg.norm(B, axis=1)
            s = np.sum(A * B, axis=1) / (na * nb + eps)
        # stash
        if mode == "joint_pos":
            keep_flat = keep.reshape(-1)
            sim_per_bin[j, keep_flat] = s
        elif mode == "joint_lvl":
            sim_per_bin[j, keep.reshape(-1)] = s
        else:
            sim_per_bin[j, np.where(keep)[0]] = s
        sim_mean[j] = float(np.nanmean(s)) if s.size else np.nan
    meta = dict(
        x_edges=x_edges,
        y_edges=y_edges,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        kept_neuron_ids=neuron_ids,
        rate_per_min=rate_per_min,
        threshold=min_spikes_per_min,
        rate_source=rate_source,
        pivot_time_axis=pivot_time_axis,
        frame_dt_used=frame_dt_used if pivot_time_axis == "frame" else None,
        frame_dt_inferred=frame_dt_inferred if pivot_time_axis == "frame" else False,
        min_time_per_bin_sec=min_time_per_bin_sec,
        levels=uniq_lvls if uniq_lvls is not None else None,
        head_bins=uniq_hd if uniq_hd is not None else None,
    )
    return dict(
        pivot_bounds=bounds,
        pivot_times=pivot_times,
        pivot_times_maze=pivot_times_maze,
        sim_per_bin=sim_per_bin,  # (n_pivots, K_full) with NaN for excluded bins
        sim_mean=sim_mean,  # per-pivot average
        mode=mode,
        metric=metric,
        normalization=row_norm_mode,
        meta=meta,
    )


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: apply]
def _apply_frame_lag(spike_counts, selected_df, positions, lag_frames):
    """
    Shift alignment between spikes (rows) and positions using absolute 'frame' index
    from selected_df. Positive lag means: spikes at frame f are paired with position at frame f+lag.
    Only rows where f+lag exists in selected_df are kept (no wrap across gaps).
    """
    # map absolute frame -> row index in selected_df / positions
    frames = np.asarray(selected_df["frame"].values, dtype=int)
    frame_to_row = {int(f): i for i, f in enumerate(frames)}
    keep_spk_rows = []
    pos_rows = []
    for i, f in enumerate(frames):
        g = int(f) + int(lag_frames)
        j = frame_to_row.get(g, None)
        if j is not None:
            keep_spk_rows.append(i)
            pos_rows.append(j)
    if len(keep_spk_rows) == 0:
        # nothing survives this lag
        return None, None, None
    # build shifted views
    spk_shift = spike_counts[np.asarray(keep_spk_rows, dtype=int), :]
    pos_shift = positions[np.asarray(pos_rows, dtype=int), :]
    df_shift = selected_df.iloc[pos_rows].reset_index(drop=True)
    return spk_shift, df_shift, pos_shift


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def scan_temporal_shifts_for_tuning_similarity(
    # original (unshifted) inputs
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    cimap,
    # search grid (frames)
    lags_train_frames=None,  # iterable of ints; if None -> symmetric range by max_shift_sec
    lags_test_frames=None,  # iterable of ints; if None -> symmetric range by max_shift_sec
    # convenience: build symmetric lag grid in seconds if lists above are None
    dt=0.04,
    max_shift_sec=0.40,
    step_frames=1,
    # args forwarded to cross_session_tuning_similarity
    mode="spatial",
    level_col="level",
    head_col="head_cat",
    n_bins_x=15,
    n_bins_y=15,
    time_window=1,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    n_neurons=None,
    random_state=0,
    train_ids=None,
    test_ids=None,
    metric="pearson",
    normalization="auto",
    intersect_mask=True,
    eps=1e-12,
    min_spikes_per_min=None,
    min_spikes_per_min_train=None,
    min_spikes_per_min_test=None,
    rate_source="spike",
    drop_if_below="either",
    # the base function to call (dependency injection; default assumes it's in scope)
    base_fn=cross_session_tuning_similarity,
):
    """
    Grid-search small temporal lags (in **frames**) between spikes and positions
    for TRAIN and TEST sessions independently, then compute place-map similarity
    using your existing `cross_session_tuning_similarity`.
    Positive lag k: spikes at absolute frame f are paired with position at frame f+k.
    Non-contiguous data are respected: rows that would cross gaps are dropped.
    Returns:
      dict with:
        - 'grid_mean' : 2D array [len(lags_train) x len(lags_test)] of mean similarity
        - 'grid_median' : same for median similarity
        - 'lags_train', 'lags_test' : the lag values scanned (frames)
        - 'coverage_train', 'coverage_test' : fraction of rows kept per lag (lists)
        - 'best' : dict with best lags, result object from base_fn, and stats
        - 'all_results' : dict[(lag_tr, lag_te)] -> full result from base_fn
    """
    # Build lag grids if not provided
    if lags_train_frames is None or lags_test_frames is None:
        max_shift_frames = int(round(max_shift_sec / float(dt)))
        lag_grid = np.arange(
            -max_shift_frames, max_shift_frames + 1, int(step_frames), dtype=int
        )
        if lags_train_frames is None:
            lags_train_frames = lag_grid
        if lags_test_frames is None:
            lags_test_frames = lag_grid
    lags_train_frames = list(map(int, lags_train_frames))
    lags_test_frames = list(map(int, lags_test_frames))
    # Precompute coverage per lag in each session
    cov_train = []
    cov_test = []
    # Prepare outputs
    G_tr = len(lags_train_frames)
    G_te = len(lags_test_frames)
    grid_mean = np.full((G_tr, G_te), np.nan, float)
    grid_median = np.full((G_tr, G_te), np.nan, float)
    all_results = {}
    # cache original lengths
    Ttr0 = selected_df_train.shape[0]
    Tte0 = selected_df_test.shape[0]
    # precompute shifted variants for each lag per session
    shifted_train = {}
    shifted_test = {}
    for i, lag_tr in enumerate(lags_train_frames):
        spk_tr_s, df_tr_s, pos_tr_s = _apply_frame_lag(
            spike_counts_train, selected_df_train, positions_train, lag_tr
        )
        shifted_train[lag_tr] = (spk_tr_s, df_tr_s, pos_tr_s)
        cov_train.append(
            0.0 if spk_tr_s is None else float(spk_tr_s.shape[0]) / max(1, Ttr0)
        )
    for j, lag_te in enumerate(lags_test_frames):
        spk_te_s, df_te_s, pos_te_s = _apply_frame_lag(
            spike_counts_test, selected_df_test, positions_test, lag_te
        )
        shifted_test[lag_te] = (spk_te_s, df_te_s, pos_te_s)
        cov_test.append(
            0.0 if spk_te_s is None else float(spk_te_s.shape[0]) / max(1, Tte0)
        )
    # scan grid
    best = dict(
        mean=-np.inf, median=-np.inf, lag_train=None, lag_test=None, result=None
    )
    for i, lag_tr in enumerate(lags_train_frames):
        spk_tr_s, df_tr_s, pos_tr_s = shifted_train[lag_tr]
        if spk_tr_s is None:
            continue  # nothing survives this lag in TRAIN
        for j, lag_te in enumerate(lags_test_frames):
            spk_te_s, df_te_s, pos_te_s = shifted_test[lag_te]
            if spk_te_s is None:
                continue  # nothing survives this lag in TEST
            # call your existing function on the shifted views
            res = base_fn(
                spk_tr_s,
                raw_counts_train,
                df_tr_s,
                pos_tr_s,
                spk_te_s,
                raw_counts_test,
                df_te_s,
                pos_te_s,
                cimap,
                mode=mode,
                level_col=level_col,
                head_col=head_col,
                n_bins_x=n_bins_x,
                n_bins_y=n_bins_y,
                dt=dt,
                time_window=time_window,
                use_peak=use_peak,
                filter_ratemap=filter_ratemap,
                sigma=sigma,
                n_neurons=n_neurons,
                random_state=random_state,
                train_ids=train_ids,
                test_ids=test_ids,
                metric=metric,
                normalization=normalization,
                intersect_mask=intersect_mask,
                eps=eps,
                min_spikes_per_min=min_spikes_per_min,
                min_spikes_per_min_train=min_spikes_per_min_train,
                min_spikes_per_min_test=min_spikes_per_min_test,
                rate_source=rate_source,
                drop_if_below=drop_if_below,
            )
            grid_mean[i, j] = res["summary"]["mean"]
            grid_median[i, j] = res["summary"]["median"]
            all_results[(lag_tr, lag_te)] = res
            # pick best by mean, tie-break by median
            if np.isfinite(grid_mean[i, j]):
                if (grid_mean[i, j] > best["mean"]) or (
                    grid_mean[i, j] == best["mean"]
                    and grid_median[i, j] > best["median"]
                ):
                    best = dict(
                        mean=grid_mean[i, j],
                        median=grid_median[i, j],
                        lag_train=lag_tr,
                        lag_test=lag_te,
                        result=res,
                    )
    return dict(
        grid_mean=grid_mean,
        grid_median=grid_median,
        lags_train=lags_train_frames,
        lags_test=lags_test_frames,
        coverage_train=cov_train,
        coverage_test=cov_test,
        best=best,
        all_results=all_results,
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def order_neurons_by_similarity(
    sim_out: Dict,
    *,
    descending: bool = True,
    by_abs: bool = False,
    nan_policy: Literal["drop", "first", "last"] = "last",
    return_indices: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Order matched neurons (train_ids/test_ids) AND their maps by the similarity
    values produced by cross_session_tuning_similarity(...).
    Parameters
    ----------
    sim_out : dict
        Return value of cross_session_tuning_similarity(...).
        Must contain keys: 'train_ids', 'test_ids', 'similarity',
        'train_map', 'test_map'.
    descending : bool, default True
        Sort order (high-to-low if True).
    by_abs : bool, default False
        If True, sort by absolute similarity.
    nan_policy : {'drop','first','last'}, default 'last'
        How to handle NaNs in similarity.
    return_indices : bool, default True
        If True, also return the permutation indices ('order').
    Returns
    -------
    result : dict
        {
          'train_ids_ordered': np.ndarray,
          'test_ids_ordered': np.ndarray,
          'similarity_ordered': np.ndarray,
          'train_map_ordered': np.ndarray,  # (K, M') reordered columns
          'test_map_ordered' : np.ndarray,  # (K, M') reordered columns
          'order': np.ndarray (if return_indices=True),
          'kept_mask': np.ndarray (bool)
        }
    """
    sim = np.asarray(sim_out["similarity"], dtype=float)
    train_ids = np.asarray(sim_out["train_ids"], dtype=int)
    test_ids = np.asarray(sim_out["test_ids"], dtype=int)
    train_map = np.asarray(sim_out["train_map"])
    test_map = np.asarray(sim_out["test_map"])
    if train_ids.shape != test_ids.shape or train_ids.shape != sim.shape:
        raise ValueError(
            "Mismatched lengths among train_ids, test_ids, and similarity."
        )
    if train_map.shape[1] != train_ids.shape[0]:
        raise ValueError("train_map second dimension must match number of train_ids.")
    if test_map.shape[1] != test_ids.shape[0]:
        raise ValueError("test_map second dimension must match number of test_ids.")
    # Sorting values
    vals = np.abs(sim) if by_abs else sim
    finite_mask = np.isfinite(vals)
    kept_mask = finite_mask.copy()
    if nan_policy == "drop":
        kept_mask = finite_mask
        vals_keep = vals[kept_mask]
    else:
        vals_keep = vals.copy()
        is_nan = ~finite_mask
        if nan_policy == "first":
            sentinel = -np.inf if not descending else np.inf
        elif nan_policy == "last":
            sentinel = np.inf if not descending else -np.inf
        else:
            raise ValueError("nan_policy must be 'drop', 'first', or 'last'")
        vals_keep[is_nan] = sentinel
        kept_mask[:] = True
    if descending:
        order_local = np.argsort(-vals_keep[kept_mask], kind="stable")
    else:
        order_local = np.argsort(vals_keep[kept_mask], kind="stable")
    # Apply mask + order
    tr_kept = train_ids[kept_mask]
    te_kept = test_ids[kept_mask]
    sim_kept = sim[kept_mask]
    map_tr_kept = train_map[:, kept_mask]
    map_te_kept = test_map[:, kept_mask]
    train_ids_ordered = tr_kept[order_local]
    test_ids_ordered = te_kept[order_local]
    similarity_ordered = sim_kept[order_local]
    train_map_ordered = map_tr_kept[:, order_local]
    test_map_ordered = map_te_kept[:, order_local]
    out = {
        "train_ids_ordered": train_ids_ordered,
        "test_ids_ordered": test_ids_ordered,
        "similarity_ordered": similarity_ordered,
        "train_map_ordered": train_map_ordered,
        "test_map_ordered": test_map_ordered,
        "kept_mask": kept_mask,
    }
    if return_indices:
        orig_idx = np.nonzero(kept_mask)[0]
        out["order"] = orig_idx[order_local]
    return out


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def plot_similarity_curve(out, use_maze_time=False, title=None, ax=None):
    """
    out: dict returned by pivot_window_tuning_similarity(...) or
         within_session_pv_temporal_similarity(...)
    use_maze_time: if True and available, plot vs out['pivot_times_maze']
    """
    # x-axis
    x = out.get("pivot_times_maze") if use_maze_time else out.get("pivot_times")
    if x is None:
        raise ValueError("Requested time axis not available in 'out'.")
    # y-axis (mean curve)
    y = out["sim_mean"]  # shape: (n_pivots,)
    y = np.asarray(y, float)
    # optional std (present in tuning-sim output)
    ystd = out.get("sim_std", None)
    if ystd is not None:
        ystd = np.asarray(ystd, float)
    # mask NaNs so the line breaks over missing segments
    good = np.isfinite(x) & np.isfinite(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
    if ystd is not None and np.any(np.isfinite(ystd)):
        good_std = good & np.isfinite(ystd)
        ax.fill_between(
            x[good_std],
            (y - ystd)[good_std],
            (y + ystd)[good_std],
            alpha=0.2,
            linewidth=0,
        )
    ax.plot(x[good], y[good], linewidth=2)
    # labels
    axis_name = "Maze time (s)" if use_maze_time else "Time (s)"
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Average similarity")
    if title is None:
        title = out.get("mode", "similarity").capitalize() + " â€” mean over bins/neurons"
    ax.set_title(title)
    # nice y-limits for correlations
    if out.get("metric") in ("pearson", "spearman", "cosine"):
        ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _bootstrap_ci(samples, ci=0.95, n_boot=2000, rng=None):
    """samples: 1D array with NaNs allowed."""
    x = samples[np.isfinite(samples)]
    if x.size == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (1 - ci) / 2.0)
    hi = np.percentile(boots, 100 * (1 - (1 - ci) / 2.0))
    return lo, hi


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_curve_and_ci(
    out, source="auto", ci=0.95, method="sem", n_boot=2000, random_state=None
):
    """
    Build mean curve and (lower, upper) CI bands per pivot from model output.
    Parameters
    ----------
    out : dict
        Output of pivot_window_tuning_similarity(...) or within_session_pv_temporal_similarity(...).
    source : {'auto','neurons','bins'}
        - 'auto'     : use 'neurons' if 'sim_matrix' in out, else 'bins'
        - 'neurons'  : CI across neurons using out['sim_matrix'] (shape: pivots x neurons)
        - 'bins'     : CI across bins using out['sim_per_bin'] (shape: pivots x bins)
    ci : float      : confidence level, e.g., 0.95
    method : {'sem','bootstrap'}
        - 'sem'      : normal-approx (mean Â± z * SEM), zâ‰ˆ1.96 for 95%
        - 'bootstrap': percentile bootstrap over available samples per pivot
    n_boot : int    : bootstrap draws
    random_state : int|None : RNG seed for bootstrap
    Returns
    -------
    dict with keys: x_default, x_maze, y_mean, y_lo, y_hi
    """
    # Choose x-axis
    if out.get("pivot_times_maze") is not None:
        x_maze = np.asarray(out["pivot_times_maze"], float)
    else:
        x_maze = None
    x_default = np.asarray(out["pivot_times"], float)
    # Determine source matrix
    if source == "auto":
        source = "neurons" if "sim_matrix" in out else "bins"
    if source == "neurons":
        if "sim_matrix" not in out:
            raise ValueError("No 'sim_matrix' in output; use source='bins' instead.")
        M = np.asarray(out["sim_matrix"], float)  # (P, N_neurons)
        y_mean = np.nanmean(M, axis=1)
        if method == "sem":
            z = (
                1.96 if np.isclose(ci, 0.95) else abs(np.sqrt(2) * erfinv(ci))
            )  # fallback
            n_eff = np.sum(np.isfinite(M), axis=1).astype(float)
            sem = np.nanstd(M, axis=1, ddof=1) / np.maximum(np.sqrt(n_eff), 1.0)
            y_lo, y_hi = y_mean - z * sem, y_mean + z * sem
        else:
            rng = np.random.default_rng(random_state)
            y_lo = np.full(M.shape[0], np.nan)
            y_hi = np.full(M.shape[0], np.nan)
            for i in range(M.shape[0]):
                lo, hi = _bootstrap_ci(M[i, :], ci=ci, n_boot=n_boot, rng=rng)
                y_lo[i], y_hi[i] = lo, hi
    elif source == "bins":
        if "sim_per_bin" not in out:
            raise ValueError(
                "No 'sim_per_bin' in output; use source='neurons' instead."
            )
        K = np.asarray(out["sim_per_bin"], float)  # (P, K_bins)
        y_mean = np.nanmean(K, axis=1)
        if method == "sem":
            z = (
                1.96 if np.isclose(ci, 0.95) else abs(np.sqrt(2) * erfinv(ci))
            )  # fallback
            n_eff = np.sum(np.isfinite(K), axis=1).astype(float)
            sem = np.nanstd(K, axis=1, ddof=1) / np.maximum(np.sqrt(n_eff), 1.0)
            y_lo, y_hi = y_mean - z * sem, y_mean + z * sem
        else:
            rng = np.random.default_rng(random_state)
            y_lo = np.full(K.shape[0], np.nan)
            y_hi = np.full(K.shape[0], np.nan)
            for i in range(K.shape[0]):
                lo, hi = _bootstrap_ci(K[i, :], ci=ci, n_boot=n_boot, rng=rng)
                y_lo[i], y_hi[i] = lo, hi
    else:
        raise ValueError("source must be 'auto', 'neurons', or 'bins'.")
    return dict(x_default=x_default, x_maze=x_maze, y_mean=y_mean, y_lo=y_lo, y_hi=y_hi)


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def plot_similarity_curve_with_ci(
    out,
    use_maze_time=False,
    source="auto",
    ci=0.95,
    method="sem",
    n_boot=2000,
    random_state=None,
    title=None,
    ax=None,
):
    """Plot mean curve with CI band."""
    r = compute_curve_and_ci(
        out,
        source=source,
        ci=ci,
        method=method,
        n_boot=n_boot,
        random_state=random_state,
    )
    x = r["x_maze"] if use_maze_time and (r["x_maze"] is not None) else r["x_default"]
    if x is None:
        raise ValueError("Requested time axis not available in output.")
    y, y_lo, y_hi = r["y_mean"], r["y_lo"], r["y_hi"]
    good = np.isfinite(x) & np.isfinite(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
    # CI band
    good_ci = good & np.isfinite(y_lo) & np.isfinite(y_hi)
    ax.fill_between(x[good_ci], y_lo[good_ci], y_hi[good_ci], alpha=0.2, linewidth=0)
    # Mean curve
    ax.plot(x[good], y[good], linewidth=2)
    # labels
    axis_name = (
        "Maze time (s)" if (use_maze_time and r["x_maze"] is not None) else "Time (s)"
    )
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Average similarity")
    if title is None:
        src = {"neurons": "neuronal", "bins": "positional", "auto": "auto"}[source]
        title = f"Mean similarity with {int(ci * 100)}% CI ({method}, {src})"
    ax.set_title(title)
    if out.get("metric") in ("pearson", "spearman", "cosine"):
        ax.set_ylim(0.1, 1)
    ax.grid(True, alpha=0.3)
    return ax


# === Functions moved from 251110_correlation.ipynb ===

# Style presets moved with the notebook plotting helpers.
PRISM_PALETTE = [
    "#4E79A7",
    "#E15759",
    "#76B7B2",
    "#F28E2B",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

STYLES = {
    "prism": {
        "figure.figsize": (4.0, 3.2),
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.prop_cycle": cycler("color", PRISM_PALETTE[:6]),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
    "pub": {
        "figure.figsize": (3.4, 2.8),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    },
}


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _pick_sans():
    """Choose a sensible sans-serif installed on this machine."""
    have = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: style]
def use_style(name="prism", overrides=None):
    """Apply a style globally; optionally supply a dict of runtime overrides."""
    cfg = STYLES.get(name)
    if cfg is None:
        raise ValueError(f"Unknown style {name!r}. Choose from {list(STYLES)}")
    mpl.rcParams.update(cfg)
    if overrides:
        mpl.rcParams.update(overrides)


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: style]
@contextmanager
def temporary_style(name="pub", overrides=None):
    """
    Functionality:
    - Temporary style.
    Inputs:
    - `name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `overrides`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    _old = mpl.rcParams.copy()
    use_style(name, overrides)
    try:
        yield
    finally:
        mpl.rcParams.update(_old)


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def load_behav_neural(nickname, behav_wd=None, neural_wd=None, video=None, ma=None):
    """Load behavior + neural arrays for one session.
    Parameters can be passed explicitly, or resolved from module globals for
    backward compatibility with notebook workflows.
    """
    if behav_wd is None:
        behav_wd = globals().get("behav_wd")
    if neural_wd is None:
        neural_wd = globals().get("neural_wd")
    if behav_wd is None or neural_wd is None:
        raise ValueError(
            "behav_wd and neural_wd are required. Pass them explicitly or set module globals."
        )
    mouseID, exp_type, exp_date = extract_info(nickname)
    tr = LoadTraj(nickname + "-tf")
    try:
        selected_df = pd.read_csv(
            os.path.join(behav_wd, "selected_data", nickname + "_selected_data.csv")
        )
        nose_interp_df = pd.read_csv(
            os.path.join(behav_wd, "selected_data", nickname + "_nose_interp_data.csv")
        )
    except Exception:
        if video is None:
            video = globals().get("video")
        if ma is None:
            ma = globals().get("ma")
        if video is None or ma is None:
            raise ValueError(
                "process_behav_data requires video and ma. Pass them explicitly or set module globals."
            )
        selected_df, nose_interp_df = process_behav_data(
            behav_wd, video, nickname, tr, ma
        )
    C_raw = C = S = Cn = None
    neural_dir = os.path.join(neural_wd, mouseID)
    if os.path.exists(neural_dir):
        if exp_type != 0:
            mouseID = mouseID + "-" + str(exp_type)
            neural_dir = os.path.join(neural_wd, mouseID)
        datasets = ["C_raw", "C", "S", "Cn"]
        data = read_mat_file(neural_dir, exp_date, datasets)
        if "C_raw" in data:
            C_raw = data["C_raw"].T
            print(f"C_raw: {C_raw.shape}")
        if "C" in data:
            C = data["C"].T
            print(f"C: {C.shape}")
        if "S" in data:
            S = data["S"].T
            print(f"S: {S.shape}")
        if "Cn" in data:
            Cn = data["Cn"].T
            print(f"Cn: {Cn.shape}")
        if "A" in data:
            A = np.transpose(data["A"], (0, 2, 1))
            print(f"A: {A.shape}")
    else:
        print(f"Neural data for {mouseID} not found in {neural_wd}")
    return tr, selected_df, nose_interp_df, C_raw, C, S, Cn


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, pickle]
def load_all_res_pickles(ex_list, path):
    """
    Functionality:
    - Load all res pickles.
    Inputs:
    - `ex_list`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `path`: Filesystem path.
      Typical shape: Scalar string/path.
    Outputs:
    - Variable `all_res` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    all_res = {}
    for ex in ex_list:
        try:
            with open(os.path.join(path, f"res_{ex}.pkl"), "rb") as f:
                res = pickle.load(f)
                all_res[ex] = res
        except FileNotFoundError:
            print(f"File not found: {os.path.join(path, f'res_{ex}.pkl')}")
    return all_res


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, pickle]
def load_all_shuf_pickles(ex_list, path):
    """
    Functionality:
    - Load all shuf pickles.
    Inputs:
    - `ex_list`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `path`: Filesystem path.
      Typical shape: Scalar string/path.
    Outputs:
    - Variable `all_shuf` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    all_shuf = {}
    for ex in ex_list:
        try:
            with open(os.path.join(path, f"shuf_res_{ex}.pkl"), "rb") as f:
                shuf_res = pickle.load(f)
                all_shuf[ex] = shuf_res
        except FileNotFoundError:
            print(f"File not found: {os.path.join(path, f'shuf_res_{ex}.pkl')}")
    return all_shuf


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def extract_similarities(all_res):
    """
    Functionality:
    - Extract similarities.
    Inputs:
    - `all_res`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `sim_values` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    sim_values = []
    for key, res in all_res.items():
        if "similarity" in res:
            sim_values.extend(res["similarity"].flatten())
    return sim_values


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def extract_shuffled_similarities(all_shuf):
    """
    Functionality:
    - Extract shuffled similarities.
    Inputs:
    - `all_shuf`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `shuf_sim_values` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    shuf_sim_values = []
    for key, shuf_res in all_shuf.items():
        if "sim_per_repeat" in shuf_res:
            shuf_sim_values.extend(shuf_res["sim_per_repeat"].flatten())
    return shuf_sim_values


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def extract_shuffled_sim_matrix(all_shuf):
    """
    Functionality:
    - Extract shuffled sim matrix.
    Inputs:
    - `all_shuf`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    shuf_sim_matrix = []
    for key, shuf_res in all_shuf.items():
        if "sim_matrix" in shuf_res:
            shuf_sim_matrix.extend(shuf_res["sim_matrix"].flatten())
    return np.array(shuf_sim_matrix)


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: build]
def build_1d_session(
    tr,
    selected_df,
    S,
    track_graph,
    start_state=127,
    end_state=94,
    offset=25,
    n_bins_x=22,
):
    """
    From a full session, extract all subsequences of the path start_stateâ†’end_state
    across bouts, build a 1D linearized trajectory, and clean NaN frames.
    Returns:
        S_tr        : (N_neurons, T_sel) spike counts along subsequences
        positions_1d: (T_sel,) positions in [0, 1)
        df_tr       : DataFrame with rows corresponding to those T_sel frames
                      (includes original 'frame' column)
    or (None, None, None) if no valid subsequences / frames.
    """
    if tr is None or selected_df is None or S is None:
        return None, None, None
    # How long the canonical path is
    threshold = len(find_path_node_based(start_state, end_state))
    subsequences = find_subsequences_in_bouts(
        tr.no, tr.fr, start_state, end_state, threshold
    )
    if len(subsequences) == 0:
        return None, None, None
    S_chunks = []
    pos_chunks = []
    df_chunks = []
    for trial in subsequences:
        start_frame = trial[0, 1]
        end_frame = trial[-1, 1] + offset  # extend a bit past the end
        temp_df = selected_df.loc[
            selected_df["frame"].between(start_frame, end_frame)
        ].copy()
        if temp_df.empty:
            continue
        frame_idx = temp_df["frame"].values.astype(int)
        # spikes: S is (N_neurons, T_full)
        trial_frames = S[:, frame_idx]  # (N_neurons, T_trial)
        # use x/y nose (or whatever you've stored in selected_df)
        pos_x = temp_df["mass_x"].values * 15
        pos_y = 15 - temp_df["mass_y"].values * 15
        pos = np.column_stack([pos_x, pos_y])  # (T_trial, 2)
        S_chunks.append(trial_frames)
        pos_chunks.append(pos)
        df_chunks.append(temp_df)
    if not S_chunks:
        return None, None, None
    # Concatenate over all subsequences
    S_tr = np.concatenate(S_chunks, axis=1)  # (N_neurons, T_sel)
    pos_2d = np.concatenate(pos_chunks, axis=0)  # (T_sel, 2)
    df_tr = pd.concat(df_chunks, axis=0, ignore_index=True)
    # Drop any frames where ANY neuron has NaN in S_tr
    valid_mask = ~np.isnan(S_tr).any(axis=0)
    if not np.any(valid_mask):
        return None, None, None
    S_tr = S_tr[:, valid_mask]
    pos_2d = pos_2d[valid_mask, :]
    df_tr = df_tr.loc[valid_mask].reset_index(drop=True)
    # Linearize along the predefined track graph
    position_df = get_linearized_position(position=pos_2d, track_graph=track_graph)
    # position_df['linear_position'] is in [0, n_bins_x); rescale to [0,1)
    positions_1d = position_df["linear_position"].to_numpy() / float(n_bins_x)
    return S_tr, positions_1d, df_tr


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _norm_id(x):
    """Return a clean session id string; turn NaN/None/'nan'/'None' into ''."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, session]
def _safe_load_session(session_id):
    """Load a session only if session_id is non-empty; else return Nones."""
    if not session_id:
        return (None, None, None, None, None, None, None)
    return load_behav_neural(session_id)


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: collect]
def collect_var_from_variant(registry, variant, var_key, which="data"):
    """
    Collect `var_key` from all pairs in one variant, ignoring groups.
    Returns:
        out: {pair_key: value}
    """
    out = {}
    # registry[variant] is {group: {pair_key: entry}}
    for group, pair_key, entry in flatten_loaded(registry[variant]):
        # Handle merged {'data','shuffle'} vs bare object
        obj = entry
        if isinstance(entry, dict) and ("data" in entry or "shuffle" in entry):
            if which is None:
                # take data if exists, else shuffle
                obj = entry.get("data", entry.get("shuffle"))
            else:
                obj = entry.get(which)
        if obj is None:
            continue
        # Support dotted paths like "metrics.similarity"
        cur = obj
        for part in var_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is None:
            continue
        out[pair_key] = cur
    return out


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map, level]
def stack_level_maps(d, sort_keys=False, return_slices=False, expected_n_levels=8):
    """
    Stack a dict of (levels, N_i) arrays horizontally into a single
    (levels, sum N_i) array.
    Parameters
    ----------
    d : dict
        {key: array_like}, each value must be shape (levels, N_i).
    sort_keys : bool, optional
        If True, stack in sorted(key) order. Otherwise use dict insertion order.
    return_slices : bool, optional
        If True, also return a dict mapping each key to its (start, end) column indices.
    expected_n_levels : int | None, optional
        Required first dimension. Set to None to infer it from the first block.
    Returns
    -------
    stacked : np.ndarray
        Array of shape (levels, sum_i N_i).
    slices : dict, optional
        {key: (start_col, end_col)} only if return_slices=True.
    """
    keys = sorted(d.keys()) if sort_keys else list(d.keys())
    blocks = []
    slices = {}
    col_start = 0
    inferred_n_levels = None if expected_n_levels is None else int(expected_n_levels)
    for k in keys:
        arr = np.asarray(d[k])
        if arr.ndim != 2:
            raise ValueError(
                f"{k}: expected 2D array, got {arr.ndim}D with shape {arr.shape}"
            )
        if inferred_n_levels is None:
            inferred_n_levels = int(arr.shape[0])
        if arr.shape[0] != inferred_n_levels:
            raise ValueError(
                f"{k}: first dimension must be {inferred_n_levels}, "
                f"got {arr.shape[0]} for key '{k}'"
            )
        blocks.append(arr)
        n = arr.shape[1]
        if return_slices:
            slices[k] = (col_start, col_start + n)
        col_start += n
    stacked = np.concatenate(blocks, axis=1)
    if return_slices:
        return stacked, slices
    return stacked, None


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def order_maps_by_preferred_index_and_similarity(maps, normalize=True):
    """
    Order columns of an (8, N) matrix so that:
      1) Columns are grouped by preferred index (argmax over the 8 rows),
         with earlier indices first (0 -> 7).
      2) Within each preferred-index group, similar maps are placed next to
         each other (based on cosine similarity after optional normalization).
    Parameters
    ----------
    maps : np.ndarray
        2D array of shape (8, N). Each column is one map (length 8).
    normalize : bool
        If True, L2-normalize each column before computing similarities so
        clustering is based on shape, not absolute amplitude.
    Returns
    -------
    order : np.ndarray
        1D array of length N; new column order.
    maps_sorted : np.ndarray
        maps[:, order], same shape as input but reordered.
    """
    maps = np.asarray(maps)
    if maps.ndim != 2 or maps.shape[0] != 8:
        raise ValueError(f"`maps` must be shape (8, N); got {maps.shape}")
    N = maps.shape[1]
    # preferred index (0..7) for each column
    pref = np.argmax(maps, axis=0)
    order = []
    for idx in range(8):  # enforce: earlier pref index comes first
        cols = np.where(pref == idx)[0]
        if cols.size == 0:
            continue
        if cols.size == 1:
            # only one map with this preferred index
            order.append(cols[0])
            continue
        # Sub-matrix for this group: shape (8, k)
        sub = maps[:, cols].astype(float)
        # Optional: normalize columns so similarity depends on pattern, not scale
        if normalize:
            norms = np.linalg.norm(sub, axis=0)
            norms[norms == 0] = 1.0
            sub /= norms
        # Cosine similarity matrix (since columns are normalized)
        sim = sub.T @ sub  # (k, k)
        k = cols.size
        used = np.zeros(k, dtype=bool)
        # Start from the map most selective at this preferred index
        start = np.argmax(sub[idx, :])
        used[start] = True
        local_order = [start]
        current = start
        # Greedy nearest-neighbor: each next map is most similar to current
        for _ in range(k - 1):
            remaining = np.where(~used)[0]
            # pick remaining column with highest similarity to current
            j = remaining[np.argmax(sim[current, remaining])]
            used[j] = True
            local_order.append(j)
            current = j
        # Map local indices back to global column indices
        order.extend(cols[j] for j in local_order)
    order = np.array(order, dtype=int)
    maps_sorted = maps[:, order]
    return order, maps_sorted


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def normalize_maps(maps, mode="l2", eps=1e-12):
    """
    Normalize columns of an (8, N) array (each column = one map).
    Parameters
    ----------
    maps : np.ndarray
        Shape (8, N)
    mode : str
        "l2"   : divide by sqrt(sum(x^2)) per column
        "max"  : divide by max(x) per column
        "sum1" : divide by sum(x) per column
        "z"    : z-score per column (mean 0, std 1)
    eps : float
        Small value to avoid division by zero.
    Returns
    -------
    norm_maps : np.ndarray
        Same shape as maps, column-normalized.
    """
    m = np.asarray(maps, float)
    if m.ndim != 2:
        raise ValueError("maps must be 2D (8, N)")
    if mode == "l2":
        norms = np.linalg.norm(m, axis=0)
        norms = np.maximum(norms, eps)
        return m / norms
    elif mode == "max":
        maxv = np.max(m, axis=0)
        maxv = np.maximum(maxv, eps)
        return m / maxv
    elif mode == "sum1":
        s = np.sum(m, axis=0)
        s = np.maximum(s, eps)
        return m / s
    elif mode == "z":
        mu = np.mean(m, axis=0)
        sd = np.std(m, axis=0)
        sd = np.maximum(sd, eps)
        return (m - mu) / sd
    else:
        raise ValueError(f"Unknown mode: {mode}")


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def order_by_index_and_similarity(maps, similarity, normalize=True):
    """
    Order columns of an (8, N) map matrix using:
      1) Preferred index (argmax) ascending (0..7),
      2) Within each index group: start from highest `similarity`,
         then add columns by greedy cosine-similarity to keep shapes close.
    Parameters
    ----------
    maps : (8, N) array
        Train maps stacked; each column is one neuron/map.
    similarity : (N,) or (1, N) array
        Per-neuron similarity score (e.g. train-test corr).
    normalize : bool
        If True, L2-normalize columns before computing similarities.
    Returns
    -------
    order : (N,) int array
        New column order.
    maps_sorted : (8, N) array
        maps[:, order]
    """
    maps = np.asarray(maps, float)
    sim = np.asarray(similarity).reshape(-1)
    if maps.ndim != 2 or maps.shape[0] != 8:
        raise ValueError(f"`maps` must be (8, N), got {maps.shape}")
    if sim.shape[0] != maps.shape[1]:
        raise ValueError(f"`similarity` length {sim.shape[0]} != N {maps.shape[1]}")
    # Optional normalization so cosine similarity depends on pattern, not power
    if normalize:
        norms = np.linalg.norm(maps, axis=0)
        norms[norms == 0] = 1.0
        norm_maps = maps / norms
    else:
        norm_maps = maps
    # Preferred bin for each map
    pref = np.argmax(norm_maps, axis=0)
    order = []
    # Loop in fixed bin order: 0 (top) -> 7 (bottom)
    for idx in range(norm_maps.shape[0]):
        cols = np.where(pref == idx)[0]
        if cols.size == 0:
            continue
        # If only one in this group, just append.
        if cols.size == 1:
            order.append(cols[0])
            continue
        sub = norm_maps[:, cols]  # (8, k)
        k = cols.size
        # Cosine similarity matrix inside this group
        sim_mat = sub.T @ sub  # (k, k)
        used = np.zeros(k, dtype=bool)
        # Start from neuron with highest external similarity in this group
        start = np.argmax(sim[cols])
        used[start] = True
        local = [start]
        cur = start
        # Greedy nearest-neighbor walk by shape similarity
        for _ in range(k - 1):
            remaining = np.where(~used)[0]
            j = remaining[np.argmax(sim_mat[cur, remaining])]
            used[j] = True
            local.append(j)
            cur = j
        # Map local indices back to global column indices
        order.extend(cols[j] for j in local)
    order = np.asarray(order, int)
    maps_sorted = maps[:, order]
    return order, maps_sorted


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: similarity]
def order_purely_by_similarity(similarity, descending=True):
    """
    Order indices using ONLY the 1D similarity array.
    Parameters
    ----------
    similarity : array-like, shape (N,) or (1, N)
        Per-neuron similarity score.
    descending : bool
        True  -> highest similarity first
        False -> lowest similarity first
    Returns
    -------
    order : (N,) int
        Permutation of [0..N-1].
    """
    sim = np.asarray(similarity).reshape(-1)
    order = np.argsort(-sim) if descending else np.argsort(sim)
    return order


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: classify, peak]
def classify_peaks_and_pref(train_maps, frac=0.5, min_max=1e-3):
    """
    For each column (8-bin map), classify how many strong bins it has
    and assign a preferred position.
    Parameters
    ----------
    train_maps : (8, N) array
    frac : float
        Threshold relative to each neuron's own max to call a bin "strong".
        E.g. 0.5 means bins >= 50% of max are counted.
    min_max : float
        If max < min_max -> treat as non-selective (class 0).
    Returns
    -------
    peak_class : (N,) int
        0 = non-selective / too weak
        1 = single-peak
        2 = two-peak
        3 = >=3 peaks (broad / complex)
    pref_pos : (N,) float
        Preferred position used for ordering within class.
        (0..7-ish; larger = later index; non-selective pushed to large value)
    """
    maps = np.asarray(train_maps, float)
    if maps.ndim != 2 or maps.shape[0] != 8:
        raise ValueError(f"`train_maps` must be (8, N), got {maps.shape}")
    N = maps.shape[1]
    maxv = maps.max(axis=0)
    peak_class = np.zeros(N, dtype=int)
    pref_pos = np.zeros(N, dtype=float)
    idx = np.arange(8)
    for j in range(N):
        mj_max = maxv[j]
        if mj_max < min_max:
            # non-selective / basically silent
            peak_class[j] = 0
            pref_pos[j] = 999.0
            continue
        mj = maps[:, j] / mj_max
        strong = np.where(mj >= frac)[0]
        k = strong.size
        if k == 1:
            # single selective bin
            peak_class[j] = 1
            pref_pos[j] = strong[0]
        elif k == 2:
            # two strong bins: use their midpoint
            peak_class[j] = 2
            pref_pos[j] = strong.mean()
        else:
            # 3+ strong bins: broad; use COM
            peak_class[j] = 3
            w = maps[:, j]
            com = (idx * w).sum() / w.sum()
            pref_pos[j] = com
    return peak_class, pref_pos


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: order, peak]
def order_by_peak_classes(train_maps, frac=0.5, min_max=1e-3):
    """
    Build an ordering:
      1) all single-peak cells (class 1), early->late by pref_pos
      2) all two-peak cells (class 2), early->late by pref_pos
      3) all broad cells (class 3), early->late by pref_pos
      4) all non-selective (class 0) at the bottom
    Uses only train_maps.
    Returns
    -------
    order : (N,) int
    """
    peak_class, pref_pos = classify_peaks_and_pref(train_maps, frac, min_max)
    # priority for each class
    # (smaller = higher up in the plot)
    class_priority = {1: 0, 2: 1, 3: 2, 0: 3}
    pri = np.vectorize(class_priority.get)(peak_class)
    # sort by (class_priority, pref_pos)
    # np.lexsort: last key is primary, so pass (pref_pos, pri)
    order = np.lexsort((pref_pos, pri))
    return order


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: classify, peak]
def classify_peaks_4groups(train_maps, frac=0.5, min_max=1e-3):
    """
    Classify each 8-bin map by number of strong peaks.
    Strong bin: value >= frac * max(map).
    Parameters
    ----------
    train_maps : (8, N) array
    frac : float
        Threshold as fraction of each neuron's own max.
    min_max : float
        If max(map) < min_max -> treat as non-selective.
    Returns
    -------
    n_peaks : (N,) int
        Number of strong bins (0 for non-selective).
    group : (N,) int
        0 = non-selective
        1 = exactly 1 strong bin
        2 = exactly 2 strong bins
        3 = exactly 3 strong bins
        4 = >=4 strong bins
    com : (N,) float
        Center-of-mass over bins (0..7-ish). For 0-peaks, set to big value (999).
    """
    maps = np.asarray(train_maps, float)
    if maps.ndim != 2 or maps.shape[0] != 8:
        raise ValueError(f"`train_maps` must be (8, N), got {maps.shape}")
    N = maps.shape[1]
    maxv = maps.max(axis=0)
    idx = np.arange(8)
    n_peaks = np.zeros(N, dtype=int)
    group = np.zeros(N, dtype=int)
    com = np.full(N, 999.0, dtype=float)
    for j in range(N):
        mj_max = maxv[j]
        if mj_max < min_max:
            # non-selective (group 0, com stays large)
            continue
        mj = maps[:, j] / mj_max
        strong = np.where(mj >= frac)[0]
        k = strong.size
        n_peaks[j] = k
        if k == 1:
            group[j] = 1
            com[j] = float(strong[0])
        elif k == 2:
            group[j] = 2
            com[j] = strong.mean()
        elif k == 3:
            group[j] = 3
            com[j] = strong.mean()
        else:  # k >= 4
            group[j] = 4
            # broad: use true COM
            w = maps[:, j]
            com[j] = float((idx * w).sum() / w.sum())
    return n_peaks, group, com


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: order, peak]
def order_by_peak_groups_and_com(train_maps, frac=0.5, min_max=1e-3, drop_ge4=False):
    """
    Build neuron order from train_maps using 4 peak groups.
    If drop_ge4 == False:
        Order by:
          1) peak group priority: 1 -> 2 -> 3 -> 4 -> 0
          2) within-group: increasing COM
    If drop_ge4 == True:
        Discard all neurons with group == 4 (>=4 peaks),
        then order the remaining neurons purely by COM
        (0-peaks have COM=999 so they fall to the bottom).
    Returns
    -------
    order : (M,) int
        Indices of columns in desired order.
        If drop_ge4, M <= N (since some are discarded).
    n_peaks, group, com : arrays
        Returned for convenience so you can inspect / mask.
    """
    n_peaks, group, com = classify_peaks_4groups(train_maps, frac, min_max)
    N = train_maps.shape[1]
    if not drop_ge4:
        # priority: 1-peak, 2-peak, 3-peak, >=4-peak, non-selective
        priority_map = {1: 0, 2: 1, 3: 2, 4: 3, 0: 4}
        pri = np.array([priority_map[g] for g in group], dtype=int)
        # non-selective already have com=999, so they sink
        order = np.lexsort((com, pri))
        return order, n_peaks, group, com
    # drop_ge4 == True
    keep = group != 4
    keep_idx = np.where(keep)[0]
    # COM already set so that:
    # - selective (1/2/3 peaks) have real COM (0..7-ish)
    # - non-selective (0) have 999 -> bottom
    order_keep = keep_idx[np.argsort(com[keep_idx])]
    return order_keep, n_peaks, group, com


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: order]
def order_hier_shape_index(
    train_maps,
    similarity=None,
    lam_index=0.5,
    lam_sim=0.0,
    eps=1e-12,
    _condensed_threshold=4096,
):
    """
    Build a distance that combines:
      - map-shape similarity (1 - corr),
      - COM proximity along the level bins,
      - optional similarity proximity,
    then use hierarchical clustering to get a 1D ordering.
    Parameters
    ----------
    train_maps : (levels, N) array
        Each column is a tuning map.
    similarity : (N,), optional
        Per-neuron cross-session similarity. If None, lam_sim is ignored.
    lam_index : float
        Weight on COM term. Larger -> stronger grouping by index.
    lam_sim : float
        Weight on similarity term. 0 = ignore similarity in clustering.
    eps : float
        Small constant for numerical stability.
    Returns
    -------
    order : (N,) int array
        New column order.
    """
    maps = np.asarray(train_maps, float)
    if maps.ndim != 2 or maps.shape[0] < 2:
        raise ValueError(f"`train_maps` must be (levels>=2, N), got {maps.shape}")
    N = maps.shape[1]
    n_levels = maps.shape[0]
    if similarity is not None:
        sim = np.asarray(similarity).reshape(-1)
        if sim.size != N:
            raise ValueError("similarity length must match N")
        use_sim = True
        # normalize similarity into ~[0,1]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + eps)
    else:
        use_sim = False
    # --- normalize maps (per column) for shape similarity ---
    norms = np.linalg.norm(maps, axis=0)
    norms[norms == 0] = 1.0
    nmaps = maps / norms
    # --- COM as continuous preferred position in [0, 1] ---
    idx = np.arange(n_levels)[:, None]
    num = (maps * idx).sum(axis=0)
    den = maps.sum(axis=0)
    den[den < eps] = 1.0
    com = num / den
    com = com / float(n_levels - 1)  # rescale to [0,1]
    # --- pairwise pieces and hierarchical clustering ---
    # The historical implementation built several full N x N matrices before
    # converting the result to SciPy's condensed upper triangle.  At the
    # maintained Figure 5 support (16,901 neurons) that needlessly requires
    # many tens of GB and can spend hours paging.  Build the *same* upper
    # triangle directly, in row-major blocks, so the scientific distance and
    # average-linkage estimator stay unchanged while peak memory is bounded by
    # the one condensed vector required by scipy.cluster.hierarchy.linkage.
    if linkage is not None and N >= int(_condensed_threshold):
        # The combined distance is exactly squared Euclidean distance in an
        # augmented feature space:
        #   (1-cosine) + lam_index*dCOM^2 + lam_sim*dSimilarity^2.
        # Because every normalized map has unit norm,
        # 1-cosine == ||u_i-u_j||^2 / 2.  ``pdist`` therefore builds the same
        # condensed vector in optimized C without any N x N intermediates.
        features = [nmaps.T / np.sqrt(2.0)]
        if lam_index != 0:
            features.append((np.sqrt(lam_index) * com)[:, None])
        if use_sim and lam_sim != 0:
            features.append((np.sqrt(lam_sim) * sim)[:, None])
        condensed = pdist(np.column_stack(features), metric="sqeuclidean")
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    else:
        # Preserve the established small-N path byte-for-byte for exploratory
        # analyses and for the no-SciPy greedy fallback.
        cos = nmaps.T @ nmaps
        cos = np.clip(cos, -1.0, 1.0)
        d_shape = 1.0 - cos
        dc = com[:, None] - com[None, :]
        d_index = dc * dc
        if use_sim:
            ds = sim[:, None] - sim[None, :]
            d_sim = ds * ds
        else:
            d_sim = 0.0
        D = d_shape + lam_index * d_index + lam_sim * d_sim
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        if linkage is None:
            order = _greedy_seriation(D)
        else:
            iu = np.triu_indices(N, k=1)
            condensed = D[iu]
            Z = linkage(condensed, method="average")
            order = leaves_list(Z)
    return np.asarray(order, int)


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: seriation]
def _greedy_seriation(D):
    """Fallback: simple greedy path through distance matrix."""
    N = D.shape[0]
    remaining = np.arange(N)
    used = np.zeros(N, bool)
    # start from the neuron with smallest average distance (most "central")
    start = np.argmin(D.mean(axis=1))
    order = [start]
    used[start] = True
    for _ in range(N - 1):
        cur = order[-1]
        cand = np.where(~used)[0]
        nxt = cand[np.argmin(D[cur, cand])]
        used[nxt] = True
        order.append(nxt)
    return np.array(order, int)


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_com(train_maps, eps=1e-12):
    """
    Center-of-mass over the 8 bins for each column.
    train_maps: (8, N)
    returns: (N,) COM in [0, 7] (roughly)
    """
    maps = np.asarray(train_maps, float)
    if maps.ndim != 2 or maps.shape[0] != 8:
        raise ValueError(f"`train_maps` must be (8, N), got {maps.shape}")
    idx = np.arange(8)[:, None]  # (8,1)
    num = (maps * idx).sum(axis=0)  # (N,)
    den = maps.sum(axis=0)  # (N,)
    zero_mask = den <= eps
    den[zero_mask] = 1.0  # avoid /0
    com = num / den  # (N,)
    # if a map is all zeros, drop it in the middle
    com[zero_mask] = 3.5
    return com


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: order, refine]
def refine_order_by_com_bands(train_maps, base_order=None, n_bands=8):
    """
    Slightly reorganize an existing order so that neurons are arranged
    along their preferred position (COM), creating a diagonal structure.
    - Uses COM bands as a coarse constraint.
    - Uses a *stable* sort, so within each band the original order is kept.
    Parameters
    ----------
    train_maps : (8, N) array
    base_order : (N,) array or None
        Existing permutation (e.g. from similarity/clustering).
        If None, start from [0..N-1].
    n_bands : int
        Number of COM bands across [0, 7]. Use 8 (per bin) or larger for finer gradation.
    Returns
    -------
    new_order : (N,) int
    """
    maps = np.asarray(train_maps, float)
    N = maps.shape[1]
    com = compute_com(maps)  # in [0,7]
    # map COM -> band index in [0, n_bands-1]
    bands = np.floor((com / 7.0) * n_bands).astype(int)
    bands = np.clip(bands, 0, n_bands - 1)
    if base_order is None:
        base_order = np.arange(N, dtype=int)
    else:
        base_order = np.asarray(base_order, int)
    # stable sort by band label applied to the *current* order
    # â†’ minimal changes, but early-band neurons move toward top, late-band toward bottom
    band_for_base = bands[base_order]
    new_order = base_order[np.argsort(band_for_base, kind="stable")]
    return new_order


# [FUNC_CATEGORY: ordering_classification] [SEARCH_TAGS: order, refine]
def refine_order_by_argmax(train_maps, base_order=None, eps=1e-6):
    """
    Slightly reorganize an existing neuron order so rows form a clearer diagonal
    by preferred bin (argmax over level bins).
    - Uses stable sort: preserves relative order within each argmax bin.
    - Neurons with earlier argmax bins move toward the top.
    Parameters
    ----------
    train_maps : (levels, N) array
        Each column is one tuning map for the train session.
    base_order : (N,) array or None
        Existing permutation (e.g. from similarity/clustering).
        If None, start from np.arange(N).
    eps : float
        Threshold to treat maps as "non-selective" (very low max).
    Returns
    -------
    new_order : (N,) int array
        Refined permutation.
    """
    maps = np.asarray(train_maps, float)
    if maps.ndim != 2 or maps.shape[0] < 2:
        raise ValueError(f"`train_maps` must be (levels>=2, N), got {maps.shape}")
    N = maps.shape[1]
    if base_order is None:
        base_order = np.arange(N, dtype=int)
    else:
        base_order = np.asarray(base_order, int)
        if base_order.size != N:
            raise ValueError(
                "base_order length must match number of columns in train_maps"
            )
    # Preferred bin by argmax
    pref = np.argmax(maps, axis=0)
    # Optionally push non-selective maps to the bottom:
    maxv = np.max(maps, axis=0)
    non_sel = maxv <= eps
    if np.any(non_sel):
        pref = pref.astype(float)
        pref[non_sel] = 999  # large label -> bottom, keeps them grouped
    # Apply a *stable* sort on the base order by these labels
    pref_on_base = pref[base_order]
    new_order = base_order[np.argsort(pref_on_base, kind="stable")]
    return new_order


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_node_positions(depth, tree_width=10.0, tree_height=5.0):
    """
    Compute (x, y) positions for a complete binary tree.
    """
    pos = {}
    n_leaves = 2 ** (depth - 1)
    xs_leaves = np.linspace(0, tree_width, n_leaves)
    y_step = tree_height / (depth - 1) if depth > 1 else 0.0
    # leaves at y=0
    for i in range(n_leaves):
        pos[(depth - 1, i)] = (xs_leaves[i], 0.0)
    # work upwards
    for level in range(depth - 2, -1, -1):
        n_nodes = 2**level
        for i in range(n_nodes):
            left_child = (level + 1, 2 * i)
            right_child = (level + 1, 2 * i + 1)
            x = (pos[left_child][0] + pos[right_child][0]) / 2.0
            y = pos[left_child][1] + y_step
            pos[(level, i)] = (x, y)
    return pos


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: draw]
def draw_binary_tree(
    depth=7,
    tree_width=8.0,
    tree_height=5.0,
    figsize=(24, 12),
    draw_root_line=True,
    root_line_length=2.0,
    root_line_direction="up",  # "up" or "down"
    cmap_name="magma",
    edge_width=5,
    root_line_width=5.0,
    node_size=75,
):
    """
    Draw a complete binary tree:
      * edges between level â„“ and â„“+1 share a color
      * nodes at level â„“ share that same color
      * leaf nodes use a separate colormap color
      * root vertical line uses the root's color
    """
    pos = compute_node_positions(depth, tree_width, tree_height)
    fig, ax = plt.subplots(figsize=figsize)
    # colors: depth internal levels + 1 leaf color
    cmap = plt.get_cmap(cmap_name)
    all_colors = [cmap(i / depth) for i in range(depth + 1)]
    level_colors = all_colors[:-1]  # for levels 0..depth-1 (internal)
    leaf_color = all_colors[-1]  # for level depth-1 nodes
    # --- draw edges (color by parent level) ---
    for level in range(depth - 1):
        n_nodes = 2**level
        color = level_colors[level + 1]
        for i in range(n_nodes):
            parent = (level, i)
            left = (level + 1, 2 * i)
            right = (level + 1, 2 * i + 1)
            x0, y0 = pos[parent]
            for child in (left, right):
                x1, y1 = pos[child]
                ax.plot(
                    [x0, x1], [y0, y1], linestyle="-", linewidth=edge_width, color=color
                )
    # --- draw nodes (color by their own level) ---
    for level in range(depth):
        n_nodes = 2**level
        if level == depth - 1:
            color = leaf_color
        else:
            color = level_colors[level + 1]
        xs = []
        ys = []
        for i in range(n_nodes):
            x, y = pos[(level, i)]
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=node_size, facecolor=color, edgecolor=color, zorder=3)
    # --- vertical line from the root (same color as level 0) ---
    extra_ymax = 0.0
    extra_ymin = 0.0
    if draw_root_line:
        root_x, root_y = pos[(0, 0)]
        if root_line_direction.lower() == "up":
            y_end = root_y + root_line_length
            extra_ymax = max(extra_ymax, root_line_length)
        else:
            y_end = root_y - root_line_length
            extra_ymin = min(extra_ymin, -root_line_length)
        ax.plot(
            [root_x, root_x],
            [root_y, y_end],
            linestyle="-",
            linewidth=root_line_width,
            color=level_colors[0],
        )
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, tree_width + 0.3 * tree_width)
    ax.set_ylim(-0.5 + extra_ymin, tree_height + 0.4 * tree_height + extra_ymax)
    plt.tight_layout()
    return fig, ax


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map, spatial]
def stack_spatial_maps(d, sort_keys=False, return_slices=False):
    """
    Stack a dict of (n_bins, N_i) arrays horizontally into a single (n_bins, sum N_i) array.
    For spatial maps: n_bins = n_bins_x * n_bins_y (flattened spatial bins)
    Parameters
    ----------
    d : dict
        {key: array_like}, each value must be shape (n_bins, N_i).
    sort_keys : bool, optional
        If True, stack in sorted(key) order. Otherwise use dict insertion order.
    return_slices : bool, optional
        If True, also return a dict mapping each key to its (start, end) column indices.
    Returns
    -------
    stacked : np.ndarray
        Array of shape (n_bins, sum_i N_i).
    slices : dict, optional
        {key: (start_col, end_col)} only if return_slices=True.
    """
    keys = sorted(d.keys()) if sort_keys else list(d.keys())
    blocks = []
    slices = {}
    col_start = 0
    for k in keys:
        arr = np.asarray(d[k])
        if arr.ndim != 2:
            raise ValueError(
                f"{k}: expected 2D array, got {arr.ndim}D with shape {arr.shape}"
            )
        blocks.append(arr)
        n = arr.shape[1]  # number of neurons
        if return_slices:
            slices[k] = (col_start, col_start + n)
        col_start += n
    # Concatenate along the second axis (neurons)
    stacked = np.concatenate(blocks, axis=1)
    if return_slices:
        return stacked, slices
    return stacked, None


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def find_universal_mask(masks_dict):
    """
    Find the intersection of all masks across sessions.
    Parameters
    ----------
    masks_dict : dict
        Dictionary of {session_key: mask_array}
    Returns
    -------
    universal_mask : np.ndarray
        Boolean mask of shape (n_bins,) that is True only where ALL sessions have valid data
    """
    # Start with all True
    first_key = list(masks_dict.keys())[0]
    universal_mask = np.ones_like(masks_dict[first_key], dtype=bool)
    # Take intersection with all masks
    for key, mask in masks_dict.items():
        universal_mask = universal_mask & mask
    return universal_mask


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr, correlation]
def columnwise_correlation(train, test):
    """
    Vectorized Pearson correlation between columns of train and test,
    handling NaNs (per-column valid mask).
    Parameters
    ----------
    train : (n_bins, n_neurons)
    test  : (n_bins, n_neurons)
    Returns
    -------
    mean_corr : float
        Mean correlation across neurons (ignoring NaNs)
    corrs : (n_neurons,) array
        Per-neuron correlations
    """
    # valid where both finite
    valid = np.isfinite(train) & np.isfinite(test)  # (n_bins, n_neurons)
    n_valid = valid.sum(axis=0)  # (#valid bins per neuron)
    n_neurons = train.shape[1]
    corrs = np.full(n_neurons, np.nan, dtype=float)
    ok = n_valid > 1  # need at least 2 points
    if not np.any(ok):
        return np.nan, corrs
    # Restrict computation to neurons with enough valid points
    v = valid[:, ok]  # (n_bins, n_ok)
    x = np.where(v, train[:, ok], 0.0)  # zero-out invalid entries
    y = np.where(v, test[:, ok], 0.0)
    n = n_valid[ok].astype(float)  # (n_ok,)
    # Sufficient statistics
    sum_x = x.sum(axis=0)
    sum_y = y.sum(axis=0)
    sum_x2 = (x**2).sum(axis=0)
    sum_y2 = (y**2).sum(axis=0)
    sum_xy = (x * y).sum(axis=0)
    # n-1, guarding against n<=1
    denom_n = n - 1.0
    denom_n[denom_n <= 0] = np.nan
    # Covariance and variances (per neuron)
    cov = (sum_xy - (sum_x * sum_y) / n) / denom_n
    var_x = (sum_x2 - (sum_x**2) / n) / denom_n
    var_y = (sum_y2 - (sum_y**2) / n) / denom_n
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    denom = std_x * std_y
    corr_ok = cov / denom
    # kill pathological cases
    corr_ok[(denom <= 0) | ~np.isfinite(corr_ok)] = np.nan
    # write back into full vector
    corrs[ok] = corr_ok
    mean_corr = np.nanmean(corrs)
    return mean_corr, corrs


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def cover_time(seq, num_states=64):
    """
    Returns the step-index (1-based) at which all `num_states`
    have been seen in `seq`.  Returns None if coverage never occurs.
    """
    seen = set()
    for idx, state in enumerate(seq, start=1):
        seen.add(state)
        if len(seen) == num_states:
            return idx
    return None


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def exploration_chunks(
    ns: List[Hashable],
    k_new: int = 32,
    include_boundary: bool = False,
    keep_incomplete_last: bool = False,
) -> Dict[str, Any]:
    """
    Segment a node-visit sequence into chunks that each accumulate `k_new`
    unique nodes (unique set resets each chunk).
    Parameters
    ----------
    ns : list
        Sequence of visited node IDs (hashable).
    k_new : int, default=32
        Target number of new (unique) nodes per chunk.
    include_boundary : bool, default=True
        If True, the next chunk starts at the *same* index where the previous
        chunk reached `k_new`. This matches "start over from that index".
        If False, the next chunk starts from the *next* index.
    keep_incomplete_last : bool, default=True
        If True, include the final partial chunk (with < k_new uniques).
    Returns
    -------
    out : dict with
        - 'segments': list of dicts, each with:
            {'start': int, 'end': int, 'steps': int,
             'unique_found': int, 'efficiency': float}
          where 'end' is exclusive and 'steps' == end - start.
        - 'per_step_efficiency': list[float]
          piecewise-constant array aligned to time steps indices in `ns`;
          each sample equals unique_found/steps for its chunk.
          For the last partial chunk, efficiency uses (unique_found/steps).
        - 'k_new': int (echo)
        - 'params': dict of parameters
    """
    if k_new <= 0:
        raise ValueError("k_new must be positive.")
    N = len(ns)
    segments: List[Dict[str, Any]] = []
    per_step_eff = [float("nan")] * N
    i = 0
    while i < N:
        seen = set()
        start = i
        j = i
        while j < N and len(seen) < k_new:
            seen.add(ns[j])
            j += 1
        unique_found = len(seen)
        if unique_found < k_new:
            # ran out of data
            if keep_incomplete_last and j > start:
                steps = j - start
                eff = unique_found / steps
                segments.append(
                    {
                        "start": start,
                        "end": j,
                        "steps": steps,
                        "unique_found": unique_found,
                        "efficiency": eff,
                    }
                )
                for t in range(start, j):
                    per_step_eff[t] = eff
            break
        # Completed a full chunk reaching k_new uniques at index j-1
        steps = j - start
        eff = k_new / steps
        segments.append(
            {
                "start": start,
                "end": j,
                "steps": steps,
                "unique_found": k_new,
                "efficiency": eff,
            }
        )
        for t in range(start, j):
            per_step_eff[t] = eff
        # advance to next chunk start
        i = j if include_boundary else min(j + 1, N)  # both cases end up the same index
        # NOTE: If you ever want to *skip* the boundary sample in the next
        # chunk, set i = j (done) vs. i = j (same). To start next at j+1, do:
        # i = min(j + 1, N)
        # But your wording says "start over from that index", i.e., j.
    return {
        "segments": segments,
        "per_step_efficiency": per_step_eff,
        "k_new": k_new,
        "params": {
            "include_boundary": include_boundary,
            "keep_incomplete_last": keep_incomplete_last,
        },
    }


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def end_level_nodes(level: int) -> np.ndarray:
    """Nodes at a specific tree level `level` (your ln list)."""
    # inclusive range: [2^level - 1, 2^(level+1) - 1)
    return np.arange(2**level - 1, 2 ** (level + 1) - 1, dtype=int)


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def safe_cover_efficiency(ns: np.ndarray, num_states: int) -> float:
    """
    Compute num_states / cover_time(ns) with safe fallback.
    If ns is empty or cover time is 0/None, fall back to len(ns) as in your snippet.
    """
    if ns is None or len(ns) == 0:
        return 0.0
    ct = cover_time(ns, num_states=num_states)
    if ct is None or ct == 0:
        return float(len(ns))
    return float(num_states) / float(ct)


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: process]
def process_one_animal(
    nickname, ma, mode_indices, end_level=6, num_states=32, k_new=32
):
    """
    Returns (half1_value, half2_value, ff_list) for one animal.
    Assumes:
      - SplitModeClips(tf, ma) returns an array with 4th col = mode index
      - tf.no[c[0]] is indexable as [start:end, 0] for node vector; [*, 1] for frame times
    """
    leave_idx, drink_idx, explore_idx = mode_indices
    tf = LoadTraj(nickname + "-tf")
    # Find clips
    cl = SplitModeClips(tf, ma)
    # Optional: these were computed in your code but not used downstream; left for parity
    # ti = np.array([tf.no[c[0]][c[1]+c[2],1] - tf.no[c[0]][c[1],1] for c in cl])  # duration in frames
    # nn = np.array([
    #     np.sum(cl[np.where(cl[:,3]==leave_idx)][:,2]),
    #     np.sum(cl[np.where(cl[:,3]==drink_idx)][:,2]),
    #     np.sum(cl[np.where(cl[:,3]==explore_idx)][:,2])
    # ])
    # nf = np.array([
    #     np.sum(ti[np.where(cl[:,3]==leave_idx)]),
    #     np.sum(ti[np.where(cl[:,3]==drink_idx)]),
    #     np.sum(ti[np.where(cl[:,3]==explore_idx)])
    # ])
    # transitions between 3 modes (kept for completeness; not used later)
    # tr = np.zeros((3,3))
    # for i in range(1, len(cl)):
    #     tr[cl[i-1,3], cl[i,3]] += 1
    # Exploration clips only
    ce = cl[np.where(cl[:, 3] == explore_idx)]
    if ce.size == 0:
        return 0.0, 0.0, []  # no exploration; return safe blanks
    # Concatenate nodes over exploration clips
    # Each ce row is [c0, start, length, mode]
    ne = np.concatenate([tf.no[c[0]][c[1] : c[1] + c[2], 0] for c in ce])  # node ids
    # Restrict to end nodes at level `end_level`
    end_nodes = end_level_nodes(end_level)
    ns = ne[np.isin(ne, end_nodes)]
    # Half1/half2 cover efficiency (safe)
    half1_val = safe_cover_efficiency(ns, num_states=num_states)
    mid = len(ns) // 2
    ns_second_half = ns[mid:] if mid > 0 else ns  # second half (non-empty fallback)
    half2_val = safe_cover_efficiency(ns_second_half, num_states=num_states)
    # Exploration chunks â†’ per-segment efficiency list
    et = exploration_chunks(ns, k_new=k_new)
    ff_list = [seg["efficiency"] for seg in et.get("segments", [])]
    return half1_val, half2_val, ff_list


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def dest_and_mean(similarities, index, use_nanmean=True):
    """
    Given a dict like:
        {"468-9day1â†’468-12day1": np.array([...]), ...}
    return a list of (dst_name, mean_value).
    If multiple entries share the same dst_name, they will each appear separately.
    """
    out = []
    for name, vals in similarities.items():
        if "â†’" not in name:
            continue  # or raise, depending on how strict you want
        dst = name.split("â†’", 1)[index].strip()
        arr = np.asarray(vals, float)
        mean_val = np.nanmean(arr) if use_nanmean else float(arr.mean())
        out.append([dst, float(mean_val)])
    return out


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: scatter]
def scatter_with_fit_and_stats(
    x,
    y,
    ax=None,
    xlabel="exp_eff",
    ylabel="avg_sim",
    title=None,
    alpha=0.6,
    xlim=None,
    ylim=None,
):
    """
    Scatter of x vs y with:
      - least-squares linear fit
      - Pearson r, RÂ², p-value
      - significance stars on the plot
    Stars:
      *   p < 0.05
      **  p < 0.01
      *** p < 0.001
      n.s. otherwise
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    # remove NaNs / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    # set figure size
    # scatter
    ax.scatter(x, y, alpha=alpha, c="#DA2323", s=50)
    # linear fit y = m x + b
    if x.size >= 2:
        m, b = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = m * x_fit + b
        ax.plot(
            x_fit,
            y_fit,
            color="black",
            linestyle="--",
            label="Fit",
            linewidth=5,
            alpha=0.7,
        )
        # Pearson r and R^2
        r = np.corrcoef(x, y)[0, 1]
        R2 = r**2
        # p-value for slope / correlation
        p_val = np.nan
        try:
            from scipy import stats
            res = stats.linregress(x, y)
            r = res.rvalue
            R2 = r**2
            p_val = res.pvalue
        except Exception:
            # fallback: if scipy not available, leave p_val as nan
            pass
        # significance stars
        if np.isnan(p_val):
            stars = ""
        elif p_val < 1e-3:
            stars = "***"
        elif p_val < 1e-2:
            stars = "**"
        elif p_val < 5e-2:
            stars = "*"
        else:
            stars = "n.s."
        # annotation in axes coordinates (top-left)
        if np.isnan(p_val):
            stats_txt = f"r = {r:.2f}, RÂ² = {R2:.2f}"
        else:
            stats_txt = f"r = {r:.2f}, RÂ² = {R2:.2f}, p = {p_val:.2g} {stars}"
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(labelsize=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, 0.7)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-0.1, 0.7)
    if title is not None:
        ax.set_title(title)
    return ax


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: sem]
def sem(x):
    """
    Functionality:
    - Compute results.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.asarray(x, dtype=float)
    return (
        np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x)))
        if np.sum(np.isfinite(x))
        else np.nan
    )


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def sig_label_from_p(p):
    """
    Functionality:
    - Sig label from p.
    Inputs:
    - `p`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin]
def bin_to_2d_coords(bin_indices, n_bins_x=15, n_bins_y=15):
    """
    Convert 1D bin indices to 2D (ix, iy) coordinates.
    Following the formula from discretize_positions:
        labels = ix + iy * n_bins_x
    Therefore:
        ix = bin_index % n_bins_x
        iy = bin_index // n_bins_x
    Parameters
    ----------
    bin_indices : array-like
        1D bin indices in range [0, n_bins_x * n_bins_y - 1]
    n_bins_x : int
        Number of bins along x
    n_bins_y : int
        Number of bins along y
    Returns
    -------
    ix, iy : arrays
        2D coordinates (bin indices)
    """
    bin_indices = np.asarray(bin_indices)
    ix = bin_indices % n_bins_x
    iy = bin_indices // n_bins_x
    return ix, iy


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin]
def euclidean_distance_bins(bin1, bin2, n_bins_x=15, n_bins_y=15):
    """
    Calculate Euclidean distance between two bin indices in 2D space.
    Parameters
    ----------
    bin1, bin2 : array-like
        1D bin indices
    n_bins_x, n_bins_y : int
        Spatial grid dimensions
    Returns
    -------
    distances : array
        Euclidean distances in bin units
    """
    ix1, iy1 = bin_to_2d_coords(bin1, n_bins_x, n_bins_y)
    ix2, iy2 = bin_to_2d_coords(bin2, n_bins_x, n_bins_y)
    distances = np.sqrt((ix1 - ix2) ** 2 + (iy1 - iy2) ** 2)
    return distances


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def find_first_endnode_visit(tf, ma):
    """
    Find the time when the animal first visits any end node.
    Parameters
    ----------
    tf : Traj
        Trajectory structure with node trajectory data (tf.no)
    ma : Maze
        Maze structure with level information (ma.le)
    Returns
    -------
    dict
        Dictionary with:
        - 'bout': bout number where first end node was visited
        - 'node': the end node number that was visited
        - 'frame_in_bout': frame within the bout
        - 'absolute_frame': absolute frame in the experiment
        - 'time_seconds': time in seconds (at 25 fps)
    Returns None if no end node was visited.
    """
    # End nodes are at the highest level (ma.le)
    # They have node numbers from 2^ma.le - 1 to 2^(ma.le+1) - 2
    endnode_start = 2**ma.le - 1
    endnode_end = 2 ** (ma.le + 1) - 1
    # Search through all bouts in chronological order
    for bout_idx, bout in enumerate(tf.no):
        # bout is an array with shape (n_states, 2): [node_number, frame_in_bout]
        # Exclude the exit state (last row)
        for state_idx in range(len(bout) - 1):
            node_num = bout[state_idx, 0]
            frame_in_bout = bout[state_idx, 1]
            # Check if this is an end node
            if endnode_start <= node_num < endnode_end:
                # Found the first end node visit!
                absolute_frame = tf.fr[bout_idx, 0] + frame_in_bout
                time_seconds = absolute_frame / 25.0  # Assuming 25 fps
                return {
                    "bout": bout_idx,
                    "node": int(node_num),
                    "frame_in_bout": int(frame_in_bout),
                    "absolute_frame": int(absolute_frame),
                    "time_seconds": time_seconds,
                }
    # No end node was visited
    return None


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, pickle]
def _default_pickle_loader(p: Path) -> Any:
    """
    Functionality:
    - Default pickle loader.
    Inputs:
    - `p`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    with p.open("rb") as f:
        return pickle.load(f)


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def _try_pandas_loader() -> Optional[Callable[[Path], Any]]:
    """Use pandas.read_pickle if available, else None."""
    try:
        import pandas as pd  # type: ignore
        return lambda p: pd.read_pickle(p)
    except Exception:
        return None


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: path]
def _find_one_path(
    name: str, roots: Iterable[Path], pattern_template: str = "wtn_{name}_spatial.pkl"
) -> Optional[Path]:
    """
    Look for a file named exactly per the template under any of the roots.
    1) Try direct join: root / pattern
    2) If not found, do a recursive exact-name search with rglob
    Returns the first match found (root order precedence). Warn on duplicates.
    """
    exact_rel = pattern_template.format(name=name)
    found_paths: List[Path] = []
    for root in roots:
        root = Path(root)
        # Direct path first
        direct = root / exact_rel
        if direct.exists():
            found_paths.append(direct)
            continue
        # rglob fallback for nested layouts
        # Use exact filename match to avoid catching other *_spatial.pkl
        target_name = Path(exact_rel).name
        matches = [p for p in root.rglob(target_name) if p.name == target_name]
        found_paths.extend(matches)
    if not found_paths:
        return None
    # Deduplicate same inode if the same file is visible via multiple paths
    unique = []
    seen = set()
    for p in found_paths:
        key = p.resolve().as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) > 1:
        warnings.warn(
            f"Multiple matches for name '{name}':\n  "
            + "\n  ".join(str(p) for p in unique)
            + "\nUsing the first one found. Consider narrowing your roots."
        )
    return unique[0]


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, pickle]
def load_group_pickles(
    groups: Dict[str, List[str]],
    data_roots: Iterable[str | Path],
    *,
    pattern_template: str = "wtn_{name}_spatial.pkl",
    loader: Optional[Callable[[Path], Any]] = None,
    strict: bool = False,
    workers: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Load files named like 'wtn_{name}_spatial.pkl' for multiple groups.
    Parameters
    ----------
    groups : dict
        Mapping from group label -> list of names (e.g., {"G1": ["163RLday", "325-6day1"], ...})
    data_roots : iterable of paths
        One or more root directories to search (first match wins).
    pattern_template : str
        Filename pattern. Default "wtn_{name}_spatial.pkl".
    loader : callable(path) -> object
        Function to load a file. Default tries pandas.read_pickle, else pickle.load.
    strict : bool
        If True, raise FileNotFoundError for any missing name. If False, warn and skip.
    workers : int
        If > 0, use ThreadPoolExecutor with this many worker threads (I/O bound = good speedup).
    Returns
    -------
    result : dict
        { group_label: { name: loaded_object, ... }, ... }
    Notes
    -----
    - Hyphens in names (e.g., '325-6day1') are fine; theyâ€™re inserted verbatim.
    - If a name appears in multiple roots, the first discovered path is used (warning emitted).
    """
    roots = [Path(r) for r in data_roots]
    if loader is None:
        loader = _try_pandas_loader() or _default_pickle_loader
    # Plan all jobs first
    jobs: List[Tuple[str, str, Optional[Path]]] = []
    for g, name_list in groups.items():
        for nm in name_list:
            p = _find_one_path(nm, roots, pattern_template=pattern_template)
            jobs.append((g, nm, p))
    missing = [(g, nm) for (g, nm, p) in jobs if p is None]
    if missing:
        msg = "Missing files for names: " + ", ".join(
            [f"{g}:{nm}" for g, nm in missing]
        )
        if strict:
            raise FileNotFoundError(msg)
        else:
            warnings.warn(msg)
    # Load
    result: Dict[str, Dict[str, Any]] = {g: {} for g in groups}
    to_load = [(g, nm, p) for (g, nm, p) in jobs if p is not None]
    if workers and workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2key = {ex.submit(loader, p): (g, nm, p) for (g, nm, p) in to_load}
            for fut in as_completed(fut2key):
                g, nm, p = fut2key[fut]
                try:
                    obj = fut.result()
                except Exception as e:
                    if strict:
                        raise
                    warnings.warn(f"Failed to load {p} ({g}:{nm}): {e}")
                    continue
                result[g][nm] = obj
    else:
        for g, nm, p in to_load:
            try:
                result[g][nm] = loader(p)  # type: ignore[arg-type]
            except Exception as e:
                if strict:
                    raise
                warnings.warn(f"Failed to load {p} ({g}:{nm}): {e}")
    return result


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def flatten_loaded(
    groups_loaded: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, str, Any]]:
    """Convert {group:{name:obj}} â†’ [(group, name, obj), ...] for easy iteration."""
    out = []
    for g, d in groups_loaded.items():
        for nm, obj in d.items():
            out.append((g, nm, obj))
    return out


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def print_load_summary(
    groups_loaded: Dict[str, Dict[str, Any]], original_groups: Dict[str, List[str]]
) -> None:
    """Print how many loaded per group (vs requested)."""
    for g, requested in original_groups.items():
        got = len(groups_loaded.get(g, {}))
        print(f"{g}: loaded {got}/{len(requested)}")


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _clean_series(t, y) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted, strictly increasing times with aligned values.
    - removes NaNs
    - sorts by time
    - drops duplicate times (keeps first)
    - truncates to min length if lengths mismatch
    """
    t = np.asarray(t).astype(float).ravel()
    y = np.asarray(y).astype(float).ravel()
    m = min(len(t), len(y))
    t, y = t[:m], y[:m]
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) == 0:
        return t, y
    order = np.argsort(t)
    t, y = t[order], y[order]
    # drop duplicated times
    uniq, idx = np.unique(t, return_index=True)
    return uniq, y[idx]


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _median_dt(t: np.ndarray) -> Optional[float]:
    """
    Functionality:
    - Median dt.
    Inputs:
    - `t`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None in some branches.
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if t.size < 2:
        return None
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else None


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_group_simmean_averages(
    loaded: Dict[str, Dict[str, Any]],
    *,
    window: str = "union",  # "union" or "overlap"
    grid_dt: Optional[float] = None,  # if None, use median of per-animal median dts
    min_points: int = 2,  # ignore animals with < min_points usable samples
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns:
      { group: {
          't_grid': np.ndarray,
          'mean':   np.ndarray,
          'sem':    np.ndarray,
          'n':      np.ndarray,   # samples used at each grid time
        }, ...
      }
    """
    out = {}
    for group, name2obj in loaded.items():
        # collect clean (t, y) series for each animal
        series: List[Tuple[np.ndarray, np.ndarray]] = []
        for name, obj in name2obj.items():
            t, y = obj.get("pivot_times_maze"), obj.get("sim_mean")
            if t is None or y is None:  # skip bad entries
                continue
            t, y = _clean_series(t, y)
            if t.size >= min_points:
                series.append((t, y))
        if not series:
            # nothing usable
            out[group] = dict(
                t_grid=np.array([]), mean=np.array([]), sem=np.array([]), n=np.array([])
            )
            continue
        # choose time window and grid step
        t_mins = [t[0] for (t, _) in series]
        t_maxs = [t[-1] for (t, _) in series]
        if window == "overlap":
            t0 = max(t_mins)
            t1 = min(t_maxs)
            if t1 <= t0:  # no overlap; fall back to union
                t0, t1 = min(t_mins), max(t_maxs)
        else:
            t0, t1 = min(t_mins), max(t_maxs)
        # decide grid_dt if not provided
        if grid_dt is None:
            dts = [
                dt for (t, _) in series if (dt := _median_dt(t)) is not None and dt > 0
            ]
            # fallback to 100 bins if we cannot infer a sensible dt
            if dts:
                grid_dt_use = float(np.median(dts))
            else:
                grid_dt_use = (t1 - t0) / 100.0 if t1 > t0 else 1.0
        else:
            grid_dt_use = float(grid_dt)
        if grid_dt_use <= 0:
            grid_dt_use = (t1 - t0) / 100.0 if t1 > t0 else 1.0
        # build grid (inclusive right edge with small epsilon)
        eps = grid_dt_use * 1e-6
        t_grid = np.arange(t0, t1 + eps, grid_dt_use)
        # interpolate each animal to grid (no extrapolation â†’ NaN outside animalâ€™s range)
        interp_stack = []
        for (t, y) in series:
            yi = np.interp(t_grid, t, y)
            outside = (t_grid < t[0]) | (t_grid > t[-1])
            yi[outside] = np.nan
            interp_stack.append(yi)
        M = np.vstack(interp_stack)  # shape: (n_animals, n_times)
        mean = np.nanmean(M, axis=0)
        sem = _nansem(M, axis=0)
        n = np.sum(np.isfinite(M), axis=0)
        out[group] = dict(t_grid=t_grid, mean=mean, sem=sem, n=n)
    return out


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: map]
def _cmap_from_base(base) -> Colormap:
    """Build a Colormap from a name, a single color, or a list/tuple of colors."""
    if isinstance(base, Colormap):
        return base
    if isinstance(base, (list, tuple)):
        cols = [to_rgb(c) for c in base]
        return LinearSegmentedColormap.from_list("custom", cols)
    # try as a named colormap
    try:
        return cm.get_cmap(str(base))
    except Exception:
        pass
    # fallback: single color -> whiteâ†’color gradient; else tab10
    if is_color_like(base):
        col = to_rgb(base)
        return LinearSegmentedColormap.from_list("custom", [(1, 1, 1), col])
    return cm.get_cmap("tab10")


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: color]
def _resolve_colors(labels, colors_arg, default_colors):
    """
    Accepts:
      - dict: {label -> color}
      - list/tuple: sequence of colors (cycled over labels)
      - str/Colormap: a colormap name or object to sample lightâ†’dark
      - single color string: creates whiteâ†’that-color gradient
    Returns {label -> color}.
    """
    # 1) No colors provided â†’ defaults
    if colors_arg is None:
        return {
            lab: default_colors[i % len(default_colors)] for i, lab in enumerate(labels)
        }
    # 2) Dict mapping
    if isinstance(colors_arg, dict):
        return {
            lab: colors_arg.get(lab, default_colors[i % len(default_colors)])
            for i, lab in enumerate(labels)
        }
    # 3) Sequence of explicit colors
    if isinstance(colors_arg, (list, tuple)):
        seq = list(colors_arg)
        return {lab: seq[i % len(seq)] for i, lab in enumerate(labels)}
    # 4) Colormap name/object or single color
    cmap = _cmap_from_base(colors_arg)
    n = max(1, len(labels))
    # sample away from extremes so lines arenâ€™t too white or too dark
    vals = [0.35] if n == 1 else np.linspace(0.35, 0.85, n)
    return {lab: to_hex(cmap(v)) for lab, v in zip(labels, vals)}


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_group_simmean(
    group_stats: Dict[str, Dict[str, np.ndarray]],
    *,
    colors: Optional[
        dict
    ] = None,  # can be dict | list/tuple | colormap name | single color
    show_n: bool = False,
    title: str = "Group-averaged sim_mean (mean Â± SEM)",
    xlabel: str = "Time in maze (s)",
    ylabel: str = "Pearson correlation",
    alpha_individual: float = 0.10,
):
    """
    Functionality:
    - Plot group simmean.
    Inputs:
    - `group_stats`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `colors`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `show_n`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `xlabel`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `ylabel`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `alpha_individual`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    plt.figure(figsize=(8, 8))
    default_colors = [
        "tab:blue",
        "tab:red",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    labels = list(group_stats.keys())
    color_map = _resolve_colors(labels, colors, default_colors)
    for i, lab in enumerate(labels):
        gs = group_stats[lab]
        t, m, s = gs["t_grid"], gs["mean"], gs["sem"]
        if t.size == 0:
            continue
        c = color_map[lab]
        plt.plot(t, m, lw=2.5, ls="--", color=c, label=lab)
        plt.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        if show_n:
            n = gs["n"].astype(float)
            n = n / (np.nanmax(n) if np.nanmax(n) > 0 else 1.0)
            plt.plot(
                t,
                np.min(m) + (np.max(m) - np.min(m)) * 0.08 * n,
                color=c,
                alpha=alpha_individual,
                lw=1.0,
            )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.2, 0.9)
    plt.legend()
    plt.tight_layout()
    plt.show()


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: color]
def make_colors(labels, base="Reds", start=0.30, end=0.85, reverse=False):
    """
    Map each label to a distinct shade.
    Parameters
    ----------
    labels : list[str]
        Group names in the order you want light â†’ dark (or vice versa if reverse=True).
    base : str | tuple[list[str]|tuple[str,...]]
        - Colormap name (e.g., "Reds", "Blues", "viridis"), or
        - A single color (e.g., "teal", "#d62728"), or
        - A list/tuple of colors to create your own gradient (e.g., ("white", "teal") or ("#fff","#f88","#900")).
    start, end : float in [0,1]
        Range inside the colormap to sample (avoid 0.0 if you want to skip near-white).
    reverse : bool
        Reverse the sampled order (dark â†’ light).
    Returns
    -------
    dict[label -> "#RRGGBB"]
    """
    # Build a colormap object
    if isinstance(base, (list, tuple)):  # custom gradient from multiple colors
        colors = [to_rgb(c) for c in base]
        cmap = LinearSegmentedColormap.from_list("custom", colors)
    else:
        try:
            # Try as a named cmap
            cmap = cm.get_cmap(base)
        except ValueError:
            # Treat as a single color â†’ gradient from white to that color
            target = to_rgb(base)
            cmap = LinearSegmentedColormap.from_list("custom", [(1, 1, 1), target])
    if reverse:
        cmap = cmap.reversed()
    n = max(1, len(labels))
    if n == 1:
        vals = [0.5 * (start + end)]
    else:
        vals = np.linspace(start, end, n)
    return {lab: to_hex(cmap(v)) for lab, v in zip(labels, vals)}


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def smooth_signal(signal, window_size=50):
    """Apply moving average smoothing with FFT (10-100x faster for large windows)"""
    T = len(signal)
    kernel = np.ones(window_size)
    sum_signal = fftconvolve(signal, kernel, mode="same")
    counts = fftconvolve(np.ones(T), kernel, mode="same")
    smoothed = sum_signal / counts
    return smoothed


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: file]
def _find_si_file(base_dir, session):
    """
    Return the first path matching */spatial_info_<session>.pkl under base_dir.
    """
    pats = [
        os.path.join(base_dir, f"spatial_info_{session}.pkl"),
        os.path.join(base_dir, "**", f"spatial_info_{session}.pkl"),
    ]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load, session]
def load_session_si(base_dir, session, si_key="SI_native"):
    """
    Load one session's spatial-info vector (e.g., 'SI_native') as a 1D numpy array.
    Returns None if file missing or key absent.
    """
    path = _find_si_file(base_dir, session)
    if path is None or (not os.path.exists(path)):
        print(f"[warn] missing: spatial_info_{session}.pkl (searched under {base_dir})")
        return None
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        v = d.get(si_key, None)
        if v is None:
            # some pair-level pkls might store 'SI_matched_s1' or 'SI_matched_s2'
            print(
                f"[warn] key '{si_key}' not in {os.path.basename(path)}; keys={list(d.keys())[:6]}..."
            )
            return None
        v = np.asarray(v, float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return v
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: collect]
def collect_group_values(groups, base_dir, si_key="SI_native"):
    """
    groups: dict like {'ChemoBIday1': ['755day1', ...], ...}
    Returns:
      per_animal: dict[group] -> list of per-session means (one scalar per session)
      per_neuron: dict[group] -> 1D array of all neurons pooled
      counts:     dict[group] -> {'animals':k, 'neurons':m}
    """
    per_animal = {}
    per_neuron = {}
    counts = {}
    for gname, sessions in groups.items():
        sess_means = []
        pooled = []
        for s in sessions:
            v = load_session_si(base_dir, s, si_key=si_key)
            if v is None:
                continue
            pooled.append(v)
            sess_means.append(float(np.nanmean(v)))
        if pooled:
            per_neuron[gname] = np.concatenate(pooled, axis=0)
        else:
            per_neuron[gname] = np.array([], float)
        per_animal[gname] = sess_means
        counts[gname] = {
            "animals": len(sess_means),
            "neurons": int(per_neuron[gname].size),
        }
    return per_animal, per_neuron, counts


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: sem]
def _sem(x):
    """
    Functionality:
    - Compute results.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.nanstd(x, ddof=1) / math.sqrt(max(1, x.size))) if x.size else np.nan


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_si_groups(
    groups,
    base_dir,
    si_key="SI_native",
    mode="animal",  # 'animal' | 'neuron'
    figsize=(8, 5),
    point_size=18,
    jitter=0.12,
    bar_alpha=0.5,
    save_path=None,
    title=None,
    max_points_per_group=None,  # for 'neuron' mode, downsample for DISPLAY ONLY
    # colors
    bar_color=None,
    point_color=None,
    errorbar_color=None,
    # font sizes
    title_fontsize=None,
    label_fontsize=None,
    tick_fontsize=None,
    count_fontsize=8,
    # --- stats options ---
    stat_test=None,  # None | 'ttest' | 'mannwhitney' | 'paired_ttest' | 'wilcoxon'
    comparisons="all_vs_first",  # "all_vs_first" | "all_pairs" | "pairwise" | list of pairs
    # pairs can be (i,j) indices or (name1,name2)
    alpha=0.05,
    show_ns=False,  # show 'n.s.' labels
    stat_text_height=0.03,  # vertical padding per bracket (fraction of y-range)
    stat_line_width=1.2,
    stat_color="k",
):
    """
    Bar = mean; error = SEM; points = per-session means ('animal') or pooled neurons ('neuron').
    Stats:
      - stat_test:
          'ttest'        -> Welch's t-test (independent; stats.ttest_ind, equal_var=False)
          'mannwhitney'  -> Mann-Whitney U, independent
          'paired_ttest' -> Paired t-test (stats.ttest_rel)
          'wilcoxon'     -> Wilcoxon signed-rank (paired)
          None           -> no tests
      - comparisons:
          "all_vs_first": compare each other group vs the first (gnames[0])
          "all_pairs" or "pairwise": all i<j group comparisons
          list of pairs: [(0,1), (2,3)] or [("ChemoBIday1","ChemoCBIday1"), ...]
    """
    from scipy import stats
    # Collect SI values
    per_animal, per_neuron, counts = collect_group_values(
        groups, base_dir, si_key=si_key
    )
    gnames = list(groups.keys())
    means, sems = [], []
    pts = []  # values used for plotting (may be downsampled)
    vals_for_test = []  # full values used for stats
    for g in gnames:
        if mode == "animal":
            vals_all = np.asarray(per_animal[g], float)
        elif mode == "neuron":
            vals_all = np.asarray(per_neuron[g], float)
        else:
            raise ValueError("mode must be 'animal' or 'neuron'")
        vals_all = vals_all[np.isfinite(vals_all)]
        # store for stats (full, no display downsampling)
        vals_for_test.append(vals_all)
        # optional downsampling ONLY for scatter display (only for neurons)
        if (
            mode == "neuron"
            and max_points_per_group is not None
            and vals_all.size > max_points_per_group
        ):
            rng = np.random.default_rng(0)
            vals_plot = rng.choice(vals_all, size=max_points_per_group, replace=False)
        else:
            vals_plot = vals_all
        pts.append(vals_plot)
        # summary stats from full data
        means.append(float(np.nanmean(vals_all)) if vals_all.size else np.nan)
        sems.append(_sem(vals_all))
    x = np.arange(len(gnames))
    width = 0.6
    # ---- color helpers ----
    def _resolve_colors(spec, default=None):
        if spec is None:
            return None
        if isinstance(spec, dict):
            return [spec.get(g, default) for g in gnames]
        if isinstance(spec, (str, tuple)):
            return [spec for _ in gnames]
        if isinstance(spec, (list, tuple)):
            if len(spec) == len(gnames):
                return list(spec)
            c0 = spec[0] if len(spec) > 0 else default
            return [c0 for _ in gnames]
        return None
    bar_colors = _resolve_colors(bar_color)
    point_colors = _resolve_colors(point_color)
    if point_colors is None and bar_colors is not None:
        point_colors = list(bar_colors)
    fig, ax = plt.subplots(figsize=figsize)
    # ---- bars ----
    if bar_colors is None:
        ax.bar(x, means, width=width, alpha=bar_alpha)
    else:
        for i, (xi, m) in enumerate(zip(x, means)):
            ax.bar(xi, m, width=width, alpha=bar_alpha, color=bar_colors[i])
    # ---- error bars ----
    ax.errorbar(
        x,
        means,
        yerr=sems,
        fmt="none",
        elinewidth=1.2,
        capsize=4,
        ecolor=(errorbar_color if errorbar_color is not None else None),
    )
    # ---- decide if we do paired plotting with connecting lines ----
    paired_mode = (
        stat_test in ("paired_ttest", "wilcoxon")
        and mode == "animal"
        and len(gnames) == 2  # simple, common case: two conditions
    )
    # ---- points (and optionally connecting lines) ----
    if paired_mode:
        # assume pts[0] and pts[1] correspond index-wise to the same animals
        arr0 = pts[0]
        arr1 = pts[1]
        n0, n1 = len(arr0), len(arr1)
        if n0 != n1:
            print(
                f"[warn] paired plotting requested but group sizes differ ({n0} vs {n1}); "
                f"connecting lines disabled, falling back to independent jitter."
            )
            paired_mode = False
        else:
            # simple symmetric offsets around group centers
            x0_center, x1_center = x[0], x[1]
            x0_pts = np.full(n0, x0_center - jitter)
            x1_pts = np.full(n1, x1_center + jitter)
            # plot lines first (so points appear on top)
            for k in range(n0):
                ax.plot(
                    [x0_pts[k], x1_pts[k]],
                    [arr0[k], arr1[k]],
                    color="gray",
                    alpha=0.5,
                    linewidth=0.8,
                    zorder=0,
                )
            # scatter the two groups
            kwargs0 = dict(s=point_size, alpha=0.9)
            kwargs1 = dict(s=point_size, alpha=0.9)
            if point_colors is not None:
                kwargs0["color"] = point_colors[0]
                kwargs1["color"] = point_colors[1]
            ax.scatter(x0_pts, arr0, **kwargs0)
            ax.scatter(x1_pts, arr1, **kwargs1)
    if not paired_mode:
        # original jittered scatter for all groups
        for i, arr in enumerate(pts):
            if arr.size == 0:
                continue
            xi = x[i] + (np.random.rand(arr.size) - 0.5) * 2 * jitter
            kwargs = dict(s=point_size, alpha=0.8)
            if point_colors is not None:
                kwargs["color"] = point_colors[i]
            ax.scatter(xi, arr, **kwargs)
    # ---- labels & fonts ----
    ax.set_xticks(x)
    ax.set_xticklabels(gnames, rotation=20)
    ax.set_ylabel("SI")
    if label_fontsize is not None:
        ax.yaxis.label.set_size(label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
    if title:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    # ==== STATS & SIGNIFICANCE BARS ====
    pvals = {}
    if stat_test is not None and len(gnames) >= 2:
        # build comparison list
        if comparisons == "all_vs_first":
            comp_pairs = [(0, j) for j in range(1, len(gnames))]
        elif comparisons in ("all_pairs", "pairwise"):
            comp_pairs = [
                (i, j) for i in range(len(gnames)) for j in range(i + 1, len(gnames))
            ]
        else:
            # user-specified; normalize names -> indices if needed
            comp_pairs = []
            for a, b in comparisons:
                if isinstance(a, str):
                    i = gnames.index(a)
                else:
                    i = int(a)
                if isinstance(b, str):
                    j = gnames.index(b)
                else:
                    j = int(b)
                comp_pairs.append((i, j))
        def p_to_stars(p):
            if p < 1e-4:
                return "****"
            if p < 1e-3:
                return "***"
            if p < 1e-2:
                return "**"
            if p < 0.05:
                return "*"
            return "n.s." if show_ns else ""
        # get current y-limits to stack brackets above
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin if ymax > ymin else 1.0
        # starting height slightly above tallest bar/point
        y_start = ymax + 0.02 * yrange
        step = stat_text_height * yrange
        for k, (i, j) in enumerate(comp_pairs):
            v1 = vals_for_test[i]
            v2 = vals_for_test[j]
            # skip empty groups
            if v1.size == 0 or v2.size == 0:
                continue
            if stat_test in ("paired_ttest", "wilcoxon"):
                # paired: require equal length, then drop NaNs pairwise
                if v1.size != v2.size:
                    print(
                        f"[warn] paired test requested but {gnames[i]} vs {gnames[j]} "
                        f"have different sizes ({v1.size} vs {v2.size}); skipping."
                    )
                    continue
                mask = np.isfinite(v1) & np.isfinite(v2)
                v1p = v1[mask]
                v2p = v2[mask]
                if v1p.size < 2:
                    print(
                        f"[warn] not enough paired non-NaN samples for {gnames[i]} vs {gnames[j]}; skipping."
                    )
                    continue
                if stat_test == "paired_ttest":
                    stat, p = stats.ttest_rel(v1p, v2p)
                else:  # 'wilcoxon'
                    try:
                        stat, p = stats.wilcoxon(v1p, v2p, alternative="two-sided")
                    except ValueError:
                        # e.g., all differences zero
                        continue
            else:
                # independent-sample tests
                if stat_test == "ttest":
                    stat, p = stats.ttest_ind(
                        v1, v2, equal_var=False, nan_policy="omit"
                    )
                elif stat_test == "mannwhitney":
                    try:
                        stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    except ValueError:
                        continue
                else:
                    raise ValueError(
                        "stat_test must be one of: None, 'ttest', 'mannwhitney', 'paired_ttest', 'wilcoxon'"
                    )
            key = (gnames[i], gnames[j])
            pvals[key] = float(p)
            label = p_to_stars(p)
            if not label:
                continue
            # bracket position
            x1, x2 = x[i], x[j]
            y = y_start + k * step
            line_y = y - 0.2 * step
            # draw bracket
            ax.plot(
                [x1, x1, x2, x2],
                [line_y, y, y, line_y],
                color=stat_color,
                linewidth=stat_line_width,
            )
            # draw text
            ax.text(
                (x1 + x2) / 2.0,
                y,
                label,
                ha="center",
                va="bottom",
                color=stat_color,
                fontsize=(label_fontsize if label_fontsize is not None else None),
            )
        # expand ylim to fit brackets
        ax.set_ylim(ymin, y_start + (len(comp_pairs) + 1) * step)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return dict(
        means=np.array(means),
        sems=np.array(sems),
        points=pts,
        counts=counts,
        pvals=pvals,
    )


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: collect]
def collect_group_vector(summary, variant, group, field="data_mean"):
    """
    Functionality:
    - Collect group vector.
    Inputs:
    - `summary`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `variant`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `group`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `field`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    pairs = list(summary[variant][group].keys())
    vals = np.array([summary[variant][group][k][field] for k in pairs], float)
    return pairs, vals


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: corr]
def _holm_correction(pvals: List[float]) -> List[float]:
    """
    Functionality:
    - Holm correction.
    Inputs:
    - `pvals`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    _, p_adj = _holm_correction_canonical(pvals, alpha=0.05)
    return np.asarray(p_adj, dtype=float).tolist()


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _fdr_bh(pvals: List[float]) -> List[float]:
    """
    Functionality:
    - Fdr bh.
    Inputs:
    - `pvals`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    adj = np.empty(m, float)
    # BH step-up
    prev = 1.0
    for i in range(m - 1, -1, -1):
        adj_i = p_sorted[i] * m / (i + 1)
        prev = min(prev, adj_i)
        adj[i] = min(1.0, prev)
    out = np.empty(m, float)
    out[order] = adj
    return out.tolist()


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: paired]
def _paired_t(x, y, alternative="two-sided"):
    """Paired t-test with effect size dz and CI for mean difference."""
    d = y - x
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return dict(
            n=n,
            t=np.nan,
            p=np.nan,
            mean_diff=np.nan,
            sd_diff=np.nan,
            dz=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
        )
    mean_d = float(np.nanmean(d))
    sd_d = float(np.nanstd(d, ddof=1))
    # t and p
    res = stats.ttest_rel(x, y, alternative=alternative, nan_policy="omit")
    tval, pval = float(res.statistic), float(res.pvalue)
    # effect size (dz)
    dz = mean_d / sd_d if sd_d > 0 else np.nan
    # 95% CI for mean difference: mean Â± tcrit * sd_d/sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    se = sd_d / math.sqrt(n)
    ci_low, ci_high = mean_d - tcrit * se, mean_d + tcrit * se
    return dict(
        n=n,
        t=tval,
        p=pval,
        mean_diff=mean_d,
        sd_diff=sd_d,
        dz=dz,
        ci_low=ci_low,
        ci_high=ci_high,
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: paired]
def _paired_wilcoxon(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank; reports n and p. (Effect size not computed here.)"""
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    n = x2.size
    if n < 1:
        return dict(n=n, W=np.nan, p=np.nan, median_diff=np.nan)
    try:
        res = stats.wilcoxon(x2, y2, alternative=alternative, zero_method="wilcox")
        W, pval = float(res.statistic), float(res.pvalue)
    except ValueError:
        # all diffs equal to zero
        W, pval = np.nan, 1.0
    median_diff = float(np.nanmedian(y2 - x2)) if n else np.nan
    return dict(n=n, W=W, p=pval, median_diff=median_diff)


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: paired]
def pairwise_paired_tests(
    M: np.ndarray,
    labels: List[str],
    *,
    test: str = "ttest",  # 'ttest' or 'wilcoxon'
    correction: str = "holm",  # 'holm' or 'fdr_bh' or None
    alternative: str = "two-sided",
    min_n: int = 3,
):
    """
    Returns:
      results: list of dicts (one per pair)
      P_adj:   square matrix of adjusted p-values (np.nan on diagonal)
    """
    M = np.asarray(M, float)
    k = M.shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    raw_ps = []
    rows = []
    for (i, j) in pairs:
        xi, yj = M[:, i], M[:, j]
        mask = np.isfinite(xi) & np.isfinite(yj)
        x, y = xi[mask], yj[mask]
        if x.size < min_n:
            res = dict(a=labels[i], b=labels[j], n=int(x.size), p=np.nan)
        else:
            if test == "wilcoxon":
                stat = _paired_wilcoxon(x, y, alternative=alternative)
                res = dict(a=labels[i], b=labels[j], **stat)
                res["p_raw"] = res.pop("p")
            else:  # t-test
                stat = _paired_t(x, y, alternative=alternative)
                res = dict(a=labels[i], b=labels[j], **stat)
                res["p_raw"] = res.pop("p")
        raw_ps.append(res["p_raw"])
        rows.append(res)
    # multiple-comparisons correction
    if correction in (None, "none"):
        p_adj = raw_ps
    elif correction == "holm":
        p_adj = _holm_correction(raw_ps)
    elif correction == "fdr_bh":
        p_adj = _fdr_bh(raw_ps)
    else:
        raise ValueError("correction must be 'holm', 'fdr_bh', or None")
    for r, pa in zip(rows, p_adj):
        r["p_adj"] = pa
    # build symmetric matrix of adjusted p-values
    P = np.full((k, k), np.nan, float)
    for (idx, (i, j)) in enumerate(pairs):
        P[i, j] = P[j, i] = p_adj[idx]
    return rows, P


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _stars(p):
    """
    Functionality:
    - Stars results.
    Inputs:
    - `p`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if np.isnan(p):
        return ""
    return (
        "****"
        if p < 1e-4
        else (
            "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))
        )
    )


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_pmatrix(
    P: np.ndarray, labels: List[str], title: str = "Pairwise tests (Holm-adjusted)"
):
    """Heatmap of -log10(p_adj) with star annotations."""
    k = len(labels)
    with np.errstate(divide="ignore"):
        H = -np.log10(P)
    vmax = np.nanpercentile(H[~np.isinf(H)], 95) if np.isfinite(H).any() else 1.0
    fig, ax = plt.subplots(figsize=(3.0 + 0.35 * k, 3.0 + 0.35 * k))
    im = ax.imshow(H, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$-\log_{10}(p_{\mathrm{adj}})$")
    # annotate with stars
    for i in range(k):
        for j in range(k):
            if i == j or np.isnan(P[i, j]):
                continue
            txt = _stars(P[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white")
    # plt.tight_layout()
    plt.show()
    return ax


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: paired]
def plot_paired_lines(M: np.ndarray, labels: List[str], a: str, b: str, title=None):
    """Within-subject lines for two chosen variants."""
    i, j = labels.index(a), labels.index(b)
    xi, yj = M[:, i], M[:, j]
    mask = np.isfinite(xi) & np.isfinite(yj)
    x, y = xi[mask], yj[mask]
    n = x.size
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for r in range(n):
        ax.plot([0, 1], [x[r], y[r]], marker="o", lw=1.0, alpha=0.7)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([a, b])
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    # Means Â± SEM
    def mean_sem(v):
        v = v[np.isfinite(v)]
        m = float(np.mean(v)) if v.size else np.nan
        s = float(np.std(v, ddof=1)) / np.sqrt(v.size) if v.size > 1 else np.nan
        return m, s
    mx, sx = mean_sem(x)
    my, sy = mean_sem(y)
    ax.errorbar([-0.05, 1.05], [mx, my], yerr=[sx, sy], fmt="s", lw=1.5, capsize=3)
    # plt.tight_layout()
    plt.show()
    return ax


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: sem]
def _nansem(a: np.ndarray, axis=0) -> np.ndarray:
    """
    Functionality:
    - Nansem results.
    Inputs:
    - `a`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `axis`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `sem` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    n = np.sum(np.isfinite(a), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nanstd(a, axis=axis, ddof=1)
        sem = np.where(n > 1, s / np.sqrt(n), np.nan)
    return sem


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_variants_bars_points_lines(
    M: np.ndarray,
    labels: List[str],  # e.g. ["d1v1","d1v2","d2v1","d2v2"]
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = (4.2, 3.2),
    y_label: str = "Similarity",
    title: Optional[str] = None,
    bar_width: float = 0.55,
    bar_alpha: float = 0.85,
    capsize: float = 3.0,
    err: str = "sem",  # "sem" | "std" | "ci95"
    point_size: float = 18,
    point_alpha: float = 0.85,
    jitter: float = 0.06,  # horizontal jitter for points
    line_alpha: float = 0.45,
    line_width: float = 1.0,
    rng_seed: int = 0,
    rows_meta: Optional[
        List[Dict[str, Any]]
    ] = None,  # [{'group':..., 'animal':...}, ...] aligned to M rows
    group_colors: Optional[
        Dict[str, str]
    ] = None,  # color per group for points/lines if rows_meta provided
):
    """
    Plot all variants together:
      - bars = column means with error bars (SEM by default)
      - scatter = individual rows (jittered)
      - lines = connect the same row across variants (within-subject)
    Parameters
    ----------
    M : array (n_rows, n_variants)
    labels : names for variants (len == n_variants)
    rows_meta : optional metadata per row; if provided and group_colors is set,
                points/lines are colored by 'group' in rows_meta.
    """
    M = np.asarray(M, float)
    n, k = M.shape
    if len(labels) != k:
        raise ValueError("len(labels) must equal number of columns in M")
    # prepare axis
    if ax is None:
        try:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        except TypeError:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
    x = np.arange(k)
    # summary stats
    col_means = np.nanmean(M, axis=0)
    if err == "std":
        yerr = np.nanstd(M, axis=0, ddof=1)
    elif err == "ci95":
        sem = _nansem(M, axis=0)
        from scipy import stats
        counts = np.sum(np.isfinite(M), axis=0)
        tcrit = stats.t.ppf(0.975, np.maximum(counts - 1, 1))
        yerr = sem * tcrit
    else:
        yerr = _nansem(M, axis=0)  # default SEM
    # bars
    bars = ax.bar(x, col_means, width=bar_width, alpha=bar_alpha)
    # errorbars (draw on top)
    ax.errorbar(
        x,
        col_means,
        yerr=yerr,
        fmt="none",
        lw=1.3,
        capsize=capsize,
        color="k",
        zorder=5,
    )
    # colors for points/lines
    rng = np.random.default_rng(rng_seed)
    if rows_meta is not None and group_colors:
        # map groups to provided colors
        row_cols = [
            group_colors.get(r.get("group"), None) if isinstance(r, dict) else None
            for r in rows_meta
        ]
    else:
        row_cols = [None] * n  # default to auto color cycle
    # scatter points + within-row connecting lines
    for i in range(n):
        row = M[i, :]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        # jittered x for this row (only where finite)
        jitter_i = rng.uniform(-jitter, jitter, size=k)
        xj = x[finite] + jitter_i[finite]
        yv = row[finite]
        c = row_cols[i]
        # points
        ax.scatter(xj, yv, s=point_size, alpha=point_alpha, zorder=4, color=c)
        # lines across available variants for this row (connect in x order)
        if np.sum(finite) >= 2:
            ax.plot(
                x[finite],
                row[finite],
                lw=line_width,
                alpha=line_alpha,
                color=c,
                zorder=3,
            )
    # cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    # show sample size above bars
    ns = np.sum(np.isfinite(M), axis=0)
    ymax = np.nanmax(
        np.vstack(
            [
                col_means + (yerr if np.all(np.isfinite(yerr)) else 0),
                np.nanmax(M, axis=0),
            ]
        ),
        axis=0,
    )
    for xi, yi, ni in zip(x, ymax, ns):
        ax.text(
            xi,
            yi + 0.02 * (np.nanmax(M) - np.nanmin(M)),
            f"n={int(ni)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    return ax


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def prior_distance_diagnostic(si_empirical, si_uniform, eps=1e-12):
    """
    Quick diagnostic: compare empirical prior vs uniform prior indirectly using K_valid/coverage.
    If you also want actual L1/KL distances between priors per window, youâ€™d need to return priors
    from the SI function. This is a lightweight proxy.
    """
    t = si_empirical["pivot_times"] / 60.0
    K = si_empirical["meta"]["K_valid"]
    cov = si_empirical["meta"]["coverage"]
    print("K_valid range:", np.nanmin(K), "to", np.nanmax(K))
    print("coverage range:", np.nanmin(cov), "to", np.nanmax(cov))
    # Compare mean curves
    d = si_empirical["si_mean"] - si_uniform["si_mean"]
    print("Mean(empirical - uniform) across windows:", np.nanmean(d))


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def _compute_reference_prior_from_full_session(
    labels_all, n_bins, min_frames, eps=1e-12
):
    """Whole-session occupancy prior (with a min_frames mask like in the pivot code)."""
    bc_full = np.bincount(labels_all, minlength=n_bins).astype(float)
    full_valid = bc_full >= int(min_frames)
    rp = np.zeros(n_bins, float)
    rp[full_valid] = bc_full[full_valid]
    s = rp.sum()
    rp = (rp / s) if s > eps else rp
    return rp


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _si_one_window_from_Q_labels(
    Q_window,
    labels_window,
    n_bins,
    n_bins_x,
    n_bins_y,
    dt_eff_tuning,
    min_frames,
    eps,
    prior_mode="empirical",
    reference_prior=None,
    reference_scope="conditional",  # only used if prior_mode='reference'
    unvisited_lam_policy="global_mean",  # only used if reference_scope='unconditional'
    common_support=None,  # optional fixed support mask (n_bins,)
    filter_ratemap=False,
    sigma=1.0,
):
    """
    Compute SI vector (M,) for one window given Q_window (T,M) and labels_window (T,).
    Requires your _skaggs_info(lam, prior, eps) to exist.
    """
    labels_window = np.asarray(labels_window, int)
    lam, bc, _ = _compute_lam_visited(
        Q_window, labels_window, n_bins, dt_eff_tuning, eps
    )
    visited = bc >= int(min_frames)
    if common_support is not None:
        visited &= np.asarray(common_support, bool)
    if not np.any(visited):
        return np.full(Q_window.shape[1], np.nan, float)
    # optional smoothing (2D only)
    if filter_ratemap and n_bins_y > 1:
        for j in range(Q_window.shape[1]):
            lam[:, j] = gaussian_filter(
                lam[:, j].reshape(n_bins_y, n_bins_x),
                sigma=sigma,
                mode="constant",
                cval=0.0,
            ).ravel()
    if prior_mode != "reference":
        prior = _make_prior(bc, visited, prior_mode, eps=eps)
        lam_use = lam.copy()
        lam_use[~visited, :] = 0.0
        return _skaggs_info(lam_use, prior, eps)
    # prior_mode == 'reference'
    if reference_prior is None:
        raise ValueError("reference_prior is required for prior_mode='reference'")
    if reference_scope == "conditional":
        prior = _make_prior(
            bc, visited, "reference", eps=eps, reference_prior=reference_prior
        )
        lam_use = lam.copy()
        lam_use[~visited, :] = 0.0
        return _skaggs_info(lam_use, prior, eps)
    elif reference_scope == "unconditional":
        prior_support = np.asarray(reference_prior, float) > 0
        if common_support is not None:
            prior_support &= np.asarray(common_support, bool)
        p = np.zeros(n_bins, float)
        p[prior_support] = np.asarray(reference_prior, float)[prior_support]
        ps = p.sum()
        prior = (p / ps) if ps > eps else p
        lam_use = _impute_lam_for_unvisited(
            lam,
            visited,
            prior_support,
            Q_window,
            dt_eff_tuning,
            eps,
            policy=unvisited_lam_policy,
        )
        lam_use[~prior_support, :] = 0.0
        return _skaggs_info(lam_use, prior, eps)
    else:
        raise ValueError("reference_scope must be 'conditional'|'unconditional'")


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def compute_shuffle_z_for_variant(
    si_real_result,  # output dict from pivot_window_spatial_information_fast
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    prior_mode="empirical",
    reference_prior=None,  # None => computed from full session
    reference_scope="conditional",
    unvisited_lam_policy="global_mean",
    support_mode="window",  # if you used common_support in pivot meta, it will be picked up
    filter_ratemap=False,
    sigma=1.0,
    n_shuffles=200,
    shuffle_mode="circular",
    shift_min_sec=10.0,
    per_neuron=True,
    random_state=0,
    z_on="raw",  # 'raw' | 'norm'
):
    """
    Computes per-window z-scores vs occupancy-preserving shuffles.
    Uses the SAME window bounds and neuron set as si_real_result.
    """
    rng = np.random.default_rng(random_state)
    bounds = si_real_result["pivot_bounds"]
    kept = np.asarray(si_real_result["kept_neuron_ids"], int)
    M = len(kept)
    # discretize full positions once
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    labels_all, _, _ = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_all = np.asarray(labels_all, int)
    n_bins = int(n_bins_x * n_bins_y)
    dt_eff_tuning = float(dt) * float(time_window + 1)
    min_frames = int(
        np.ceil(float(si_real_result["meta"]["min_time_per_bin_sec"]) / dt_eff_tuning)
    )
    # reference prior if needed
    if prior_mode == "reference":
        if reference_prior is None:
            reference_prior = _compute_reference_prior_from_full_session(
                labels_all, n_bins, min_frames
            )
        else:
            reference_prior = np.asarray(reference_prior, float)
    # common support if your pivot used it (optional)
    common_support = (
        si_real_result["meta"].get("common_support", None)
        if support_mode != "window"
        else None
    )
    # pre-filtered counts
    spike_f = spike_counts[:, kept]
    # shift in frames
    min_shift_frames = int(np.ceil(float(shift_min_sec) / dt_eff_tuning))
    min_shift_frames = max(min_shift_frames, 1)
    # choose real matrix to z-score
    if z_on == "raw":
        real_mat = np.asarray(si_real_result["si_matrix"], float)
    elif z_on == "norm":
        real_mat = np.asarray(si_real_result["si_matrix_norm"], float)
    else:
        raise ValueError("z_on must be 'raw' or 'norm'")
    z = np.full_like(real_mat, np.nan)
    null_mean = np.full_like(real_mat, np.nan)
    null_std = np.full_like(real_mat, np.nan)
    for w_i, (s_win, e_win) in enumerate(bounds):
        if e_win <= s_win:
            continue
        # build Q_window consistent with pivot
        if time_window == 0:
            Q_window = spike_f[s_win:e_win, :]
        else:
            frames_in_window = np.arange(s_win, e_win, dtype=int)
            Q_window = build_count_matrix(
                spike_counts,
                raw_counts,
                selected_df,
                frames=frames_in_window,
                time_window=time_window,
                neuron_ids=kept,
            ).T
        labels_window = labels_all[s_win:e_win]
        # streaming mean/var (Welford) for null
        mu = np.zeros(M, float)
        m2 = np.zeros(M, float)
        n = 0
        for _ in range(int(n_shuffles)):
            if shuffle_mode == "circular":
                Qs = _circular_shift_Q(
                    Q_window,
                    rng,
                    min_shift_frames=min_shift_frames,
                    per_neuron=per_neuron,
                )
                labs = labels_window
            else:
                raise ValueError(
                    "Only shuffle_mode='circular' implemented here (recommended)."
                )
            si_shuf = _si_one_window_from_Q_labels(
                Qs,
                labs,
                n_bins=n_bins,
                n_bins_x=n_bins_x,
                n_bins_y=n_bins_y,
                dt_eff_tuning=dt_eff_tuning,
                min_frames=min_frames,
                eps=1e-12,
                prior_mode=prior_mode,
                reference_prior=reference_prior,
                reference_scope=reference_scope,
                unvisited_lam_policy=unvisited_lam_policy,
                common_support=common_support,
                filter_ratemap=filter_ratemap,
                sigma=sigma,
            )
            if z_on == "norm":
                K = si_real_result["meta"]["K_valid"][w_i]
                denom = np.log2(K) if (np.isfinite(K) and K > 1) else np.nan
                si_shuf = (
                    si_shuf / denom
                    if np.isfinite(denom)
                    else np.full_like(si_shuf, np.nan)
                )
            n += 1
            delta = si_shuf - mu
            mu += delta / n
            m2 += delta * (si_shuf - mu)
        sd = np.sqrt(m2 / max(n - 1, 1)) + 1e-12
        null_mean[w_i, :] = mu
        null_std[w_i, :] = sd
        z[w_i, :] = (real_mat[w_i, :] - mu) / sd
    # summary over neurons per window
    z_mean = np.nanmean(z, axis=1)
    z_std = np.nanstd(z, axis=1)
    z_sem = z_std / np.sqrt(max(M, 1))
    return dict(
        z_matrix=z,
        z_mean=z_mean,
        z_sem=z_sem,
        null_mean=null_mean,
        null_std=null_std,
        pivot_times=si_real_result["pivot_times"],
        pivot_times_maze=si_real_result.get("pivot_times_maze", None),
        kept_neuron_ids=kept,
        meta=dict(
            n_shuffles=n_shuffles,
            shuffle_mode=shuffle_mode,
            shift_min_sec=shift_min_sec,
            per_neuron=per_neuron,
            z_on=z_on,
        ),
    )


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: run]
def run_si_variants(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    win_len_sec=10 * 60,
    pivot_step_sec=60,
    min_time_per_bin_sec=1.0,
    min_spikes_per_min=0.1,
    random_state=0,
    filter_ratemap=False,
    sigma=1.0,
):
    """
    Functionality:
    - Run si variants.
    Inputs:
    - `spike_counts`: Neural activity values.
      Typical shape: Typically (n_frames, n_neurons).
    - `raw_counts`: Raw activity values before processing.
      Typical shape: Typically (n_frames, n_neurons).
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `positions`: Position coordinates over time.
      Typical shape: Typically (n_frames, 2) for x/y.
    - `n_bins_x`: Count/size parameter.
      Typical shape: Scalar integer.
    - `n_bins_y`: Count/size parameter.
      Typical shape: Scalar integer.
    - `dt`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `time_window`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `win_len_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `pivot_step_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_time_per_bin_sec`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_spikes_per_min`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `random_state`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `filter_ratemap`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `sigma`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Variable `out` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    variants = {
        "empirical": dict(prior_mode="empirical"),
        "uniform": dict(prior_mode="uniform"),
        "ref_cond": dict(
            prior_mode="reference", reference_prior=None, reference_scope="conditional"
        ),
        # (optional) include unconditional if you want:
        # "ref_uncond_mean": dict(prior_mode="reference", reference_prior=None, reference_scope="unconditional", unvisited_lam_policy="global_mean"),
        # "ref_uncond_zero": dict(prior_mode="reference", reference_prior=None, reference_scope="unconditional", unvisited_lam_policy="zero"),
    }
    out = {}
    for name, kw in variants.items():
        out[name] = pivot_window_spatial_information_fast(
            spike_counts,
            raw_counts,
            selected_df,
            positions,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            dt=dt,
            time_window=time_window,
            win_len_sec=win_len_sec,
            pivot_step_sec=pivot_step_sec,
            min_time_per_bin_sec=min_time_per_bin_sec,
            min_spikes_per_min=min_spikes_per_min,
            pivot_time_axis="selected",
            random_state=random_state,
            filter_ratemap=filter_ratemap,
            sigma=sigma,
            support_mode="window",
            subsample_mode="none",
            **kw,
        )
    return out


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_si_comparison(si_dict, title="SI under different priors", use_norm=False):
    """
    Functionality:
    - Plot si comparison.
    Inputs:
    - `si_dict`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `use_norm`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    any_key = next(iter(si_dict))
    t = si_dict[any_key]["pivot_times"] / 60.0
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in si_dict.items():
        if use_norm:
            mean = res["si_mean_norm"]
            std = res["si_std_norm"]
            ylab = "Normalized SI (SI / log2(K_valid))"
        else:
            mean = res["si_mean"]
            std = res["si_std"]
            ylab = "Spatial Information (bits/spike)"
        M = res["si_matrix"].shape[1]
        sem = std / np.sqrt(max(M, 1))
        ax.plot(t, mean, linewidth=2, label=f"{name} (n={M})")
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(ylab)
    ax.set_title(title + (" [normalized]" if use_norm else ""))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    # plot coverage + K_valid once
    meta = si_dict[any_key]["meta"]
    if "coverage" in meta and "K_valid" in meta:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(t, meta["coverage"], linewidth=2)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Coverage (fraction valid bins)")
        ax.set_title("Window coverage")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(t, meta["K_valid"], linewidth=2)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("K_valid (# valid bins)")
        ax.set_title("Window support size (K_valid)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_z_comparison(z_dict, title="Shuffle z-score vs time"):
    """
    Functionality:
    - Plot z comparison.
    Inputs:
    - `z_dict`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    any_key = next(iter(z_dict))
    t = z_dict[any_key]["pivot_times"] / 60.0
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in z_dict.items():
        z_mean = res["z_mean"]
        z_sem = res["z_sem"]
        ax.plot(t, z_mean, linewidth=2, label=name)
        ax.fill_between(t, z_mean - z_sem, z_mean + z_sem, alpha=0.2)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("z(SI) vs shuffle")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def compute_shuffle_mean_for_variant_fast(
    si_real_result,
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    prior_mode="uniform",
    reference_prior=None,
    reference_scope="conditional",
    unvisited_lam_policy="global_mean",
    support_mode="window",
    filter_ratemap=False,
    sigma=1.0,
    n_shuffles=30,
    shift_min_sec=30.0,
    per_neuron=False,  # <<< FAST: shift all neurons together
    neuron_subset=200,  # <<< FAST: None for all neurons
    random_state=0,
    use_norm=True,  # <<< recommended with changing K_valid
):
    """
    Fast shuffle control that returns one value per window:
      - real_mean(t): mean SI (or SI_norm) across neurons
      - null_mean(t), null_std(t): across shuffles of mean SI
      - delta(t) = real_mean - null_mean
      - z_mean(t) = delta / null_std  (only stable if n_shuffles is decent)
    Much faster than per-neuron z because we aggregate over neurons first.
    """
    rng = np.random.default_rng(random_state)
    bounds = si_real_result["pivot_bounds"]
    kept_all = np.asarray(si_real_result["kept_neuron_ids"], int)
    M_all = len(kept_all)
    # --- optionally subsample neurons for speed ---
    if (neuron_subset is not None) and (neuron_subset < M_all):
        kept = rng.choice(kept_all, size=int(neuron_subset), replace=False)
        kept.sort()
    else:
        kept = kept_all
    # discretize full positions once
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    labels_all, _, _ = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_all = np.asarray(labels_all, int)
    n_bins = int(n_bins_x * n_bins_y)
    dt_eff_tuning = float(dt) * float(time_window + 1)
    min_frames = int(
        np.ceil(float(si_real_result["meta"]["min_time_per_bin_sec"]) / dt_eff_tuning)
    )
    # reference prior if needed
    if prior_mode == "reference":
        if reference_prior is None:
            reference_prior = _compute_reference_prior_from_full_session(
                labels_all, n_bins, min_frames
            )
        else:
            reference_prior = np.asarray(reference_prior, float)
    # common support if your pivot used it (optional)
    common_support = (
        si_real_result["meta"].get("common_support", None)
        if support_mode != "window"
        else None
    )
    # pre-filtered counts
    spike_f = spike_counts[:, kept]
    # shift in frames
    min_shift_frames = max(int(np.ceil(float(shift_min_sec) / dt_eff_tuning)), 1)
    # real mean curve on the same neuron subset
    if use_norm:
        real_mat = np.asarray(si_real_result["si_matrix_norm"], float)
    else:
        real_mat = np.asarray(si_real_result["si_matrix"], float)
    # map subset columns
    # (kept is sorted; kept_all is sorted in si_real_result)
    col_idx = np.searchsorted(kept_all, kept)
    real_mean = np.nanmean(real_mat[:, col_idx], axis=1)
    null_mean = np.full(len(bounds), np.nan, float)
    null_std = np.full(len(bounds), np.nan, float)
    delta = np.full(len(bounds), np.nan, float)
    z_mean = np.full(len(bounds), np.nan, float)
    for w_i, (s_win, e_win) in enumerate(bounds):
        if e_win <= s_win:
            continue
        # window data
        if time_window == 0:
            Q_window = spike_f[s_win:e_win, :]
        else:
            frames_in_window = np.arange(s_win, e_win, dtype=int)
            Q_window = build_count_matrix(
                spike_counts,
                raw_counts,
                selected_df,
                frames=frames_in_window,
                time_window=time_window,
                neuron_ids=kept,
            ).T
        labels_window = labels_all[s_win:e_win]
        # compute null distribution over shuffles of the MEAN (fast)
        null_vals = np.zeros(int(n_shuffles), float)
        for k in range(int(n_shuffles)):
            Qs = _circular_shift_Q(
                Q_window, rng, min_shift_frames=min_shift_frames, per_neuron=per_neuron
            )
            si_vec = _si_one_window_from_Q_labels(
                Qs,
                labels_window,
                n_bins=n_bins,
                n_bins_x=n_bins_x,
                n_bins_y=n_bins_y,
                dt_eff_tuning=dt_eff_tuning,
                min_frames=min_frames,
                eps=1e-12,
                prior_mode=prior_mode,
                reference_prior=reference_prior,
                reference_scope=reference_scope,
                unvisited_lam_policy=unvisited_lam_policy,
                common_support=common_support,
                filter_ratemap=filter_ratemap,
                sigma=sigma,
            )
            if use_norm:
                K = si_real_result["meta"]["K_valid"][w_i]
                denom = np.log2(K) if (np.isfinite(K) and K > 1) else np.nan
                si_vec = (
                    si_vec / denom
                    if np.isfinite(denom)
                    else np.full_like(si_vec, np.nan)
                )
            null_vals[k] = np.nanmean(si_vec)
        mu = np.nanmean(null_vals)
        sd = np.nanstd(null_vals) + 1e-12
        null_mean[w_i] = mu
        null_std[w_i] = sd
        delta[w_i] = real_mean[w_i] - mu
        z_mean[w_i] = delta[w_i] / sd
    return dict(
        pivot_times=si_real_result["pivot_times"],
        real_mean=real_mean,
        null_mean=null_mean,
        null_std=null_std,
        delta=delta,
        z_mean=z_mean,
        meta=dict(
            n_shuffles=n_shuffles,
            per_neuron=per_neuron,
            neuron_subset=len(kept),
            shift_min_sec=shift_min_sec,
            use_norm=use_norm,
            prior_mode=prior_mode,
        ),
    )


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def plot_shuffle_delta(shuf_dict, title="Î”SI vs shuffle"):
    """
    Functionality:
    - Plot shuffle delta.
    Inputs:
    - `shuf_dict`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    t = shuf_dict["pivot_times"] / 60.0
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, shuf_dict["delta"], linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (min)")
    plt.ylabel("Î”SI  (real âˆ’ mean(shuffle))")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: shuffle]
def plot_shuffle_zmean(shuf_dict, title="z(mean SI) vs shuffle"):
    """
    Functionality:
    - Plot shuffle zmean.
    Inputs:
    - `shuf_dict`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `title`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None (no explicit return).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    t = shuf_dict["pivot_times"] / 60.0
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, shuf_dict["z_mean"], linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time (min)")
    plt.ylabel("z of mean SI")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: process]
def process_session_si(
    session_name,
    save_dir,
    params,
    *,
    session_to_group=None,
    maze_factory=None,
    ma_obj=None,
    behav_wd=None,
    neural_wd=None,
    video=None,
):
    """
    Process one session using the original pivot_window_spatial_information function.
    This is slower but produces accurate results.
    """
    try:
        start_time = time.time()
        # Load data
        if ma_obj is not None:
            ma = ma_obj
        else:
            if maze_factory is None:
                maze_factory = globals().get("NewMaze")
            if maze_factory is None:
                raise ValueError(
                    "maze_factory/NewMaze is required to build maze object."
                )
            ma = maze_factory(6)
        tr, selected_df, nose_interp_df, C_raw, C, S, Cn = load_behav_neural(
            session_name,
            behav_wd=behav_wd,
            neural_wd=neural_wd,
            video=video,
            ma=ma,
        )
        # Remove NaN frames
        nan_idx = np.where(np.isnan(C.T))[0]
        selected_df = selected_df[~selected_df["frame"].isin(nan_idx)]
        # Add levels
        level_list = generate_level_list(ma.ce, grid_size=15)
        levels = assign_levels(selected_df, ma.ce, level_list)
        selected_df["level"] = levels
        selected_df = selected_df[~selected_df["level"].isna()]
        selected_df = selected_df.reset_index(drop=True)
        # Extract positions and spike counts
        positions = np.array(
            [selected_df["mass_x"].values, selected_df["mass_y"].values]
        ).T
        frame = selected_df["frame"].values.astype(int)
        spike_counts = S.T[frame]
        raw_counts = C.T
        # Run SI analysis with original (accurate) function
        si_result = pivot_window_spatial_information(
            spike_counts, raw_counts, selected_df, positions, **params
        )
        # Save results
        output_file = os.path.join(save_dir, f"{session_name}_si_result.pkl")
        group_map = (
            session_to_group
            if session_to_group is not None
            else globals().get("session_to_group", {})
        )
        with open(output_file, "wb") as f:
            pickle.dump(
                {
                    "session_name": session_name,
                    "si_result": si_result,
                    "params": params,
                    "group": group_map.get(session_name, "unknown"),
                },
                f,
            )
        elapsed = time.time() - start_time
        print(
            f"âœ“ {session_name}: {elapsed:.1f}s - SI shape: {si_result['si_matrix'].shape}"
        )
        return True, session_name, elapsed
    except Exception as e:
        print(f"âœ— {session_name}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False, session_name, str(e)


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_group_si_curves(save_dir, groups_to_plot=None):
    """
    Load all results and plot averaged SI curves by group
    """
    import glob
    # Load all results
    pkl_files = glob.glob(os.path.join(save_dir, "*_si_result.pkl"))
    group_data = {}
    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        group = data.get("group", "unknown")
        si_result = data["si_result"]
        if group not in group_data:
            group_data[group] = []
        # Store mean SI curve for this session
        group_data[group].append(
            {
                "session": data["session_name"],
                "time": si_result["pivot_times"] / 60,
                "mean_si": si_result["si_mean"],
                "si_matrix": si_result["si_matrix"],
            }
        )
    # Plot
    if groups_to_plot is None:
        groups_to_plot = sorted(group_data.keys())
    n_groups = len(groups_to_plot)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups))
    if n_groups == 1:
        axes = [axes]
    for ax, group_name in zip(axes, groups_to_plot):
        if group_name not in group_data:
            continue
        sessions = group_data[group_name]
        # Find common time axis (use shortest)
        min_len = min(len(s["time"]) for s in sessions)
        # Collect all curves
        all_curves = []
        for sess in sessions:
            curve = sess["mean_si"][:min_len]
            all_curves.append(curve)
            # Plot individual sessions faintly
            ax.plot(sess["time"][:min_len], curve, "gray", alpha=0.3, linewidth=0.5)
        # Compute group average
        all_curves = np.array(all_curves)
        group_mean = np.nanmean(all_curves, axis=0)
        group_sem = np.nanstd(all_curves, axis=0) / np.sqrt(len(all_curves))
        time_axis = sessions[0]["time"][:min_len]
        # Plot group average
        ax.plot(
            time_axis,
            group_mean,
            "b-",
            linewidth=3,
            label=f"Group Mean (n={len(sessions)})",
        )
        ax.fill_between(
            time_axis,
            group_mean - group_sem,
            group_mean + group_sem,
            alpha=0.3,
            color="b",
        )
        ax.set_xlabel("Time (minutes)", fontsize=11)
        ax.set_ylabel("SI (bits/spike)", fontsize=11)
        ax.set_title(f"{group_name}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, group_data


# [FUNC_CATEGORY: plotting] [SEARCH_TAGS: plot]
def plot_si_curve(session_name, save_dir):
    """Load and plot SI curve for a session"""
    file_path = os.path.join(save_dir, f"{session_name}_si_result.pkl")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    si_result = data["si_result"]
    group = data.get("group", "unknown")
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    time_axis = si_result["pivot_times"] / 60  # Convert to minutes
    mean_si = si_result["si_mean"]
    sem_si = si_result["si_std"] / np.sqrt(si_result["si_matrix"].shape[1])
    # Plot
    ax.plot(time_axis, mean_si, "b-", linewidth=2, label="Mean SI")
    ax.fill_between(
        time_axis,
        mean_si - sem_si,
        mean_si + sem_si,
        alpha=0.3,
        color="b",
        label="Â±SEM",
    )
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Spatial Information (bits/spike)", fontsize=12)
    ax.set_title(f"{session_name} ({group}) - SI Dynamics", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Add stats
    stats_text = (
        f"Neurons: {si_result['si_matrix'].shape[1]}\n"
        f"Mean SI: {np.nanmean(mean_si):.4f}\n"
        f"Windows: {len(time_axis)}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )
    plt.tight_layout()
    return fig, ax


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: sparsity]
def compute_lifetime_sparsity(rates):
    """
    Compute lifetime sparsity (Treves & Rolls, 1995) for a population of neurons.
    Measures how evenly activity is distributed across neurons.
    Higher values (closer to 1) mean more evenly distributed activity.
    Lower values (closer to 0) mean activity concentrated in few neurons.
    Parameters:
    -----------
    rates : array-like, shape (n_neurons,) or (n_neurons, n_bins)
        Firing rates for each neuron. Can be mean rates or rate maps.
    Returns:
    --------
    sparsity : float or array
        Lifetime sparsity value(s). Range [0, 1].
        If rates is 2D, returns sparsity per spatial bin.
    """
    rates = np.array(rates)
    # Handle 2D rate maps (neurons x bins)
    if rates.ndim == 2:
        # Compute sparsity per spatial bin (across neurons)
        mean_rate = np.nanmean(rates, axis=0)  # Mean per bin
        mean_sq_rate = np.nanmean(rates**2, axis=0)  # Mean of squares per bin
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            sparsity = (mean_rate**2) / mean_sq_rate
            sparsity = np.where(mean_sq_rate > 0, sparsity, np.nan)
        n_neurons = rates.shape[0]
        # Normalize to [0, 1] range
        sparsity = (1 - sparsity) / (1 - 1 / n_neurons)
    else:
        # 1D case: single set of rates across neurons
        mean_rate = np.nanmean(rates)
        mean_sq_rate = np.nanmean(rates**2)
        if mean_sq_rate > 0:
            sparsity = (mean_rate**2) / mean_sq_rate
            n_neurons = len(rates)
            sparsity = (1 - sparsity) / (1 - 1 / n_neurons)
        else:
            sparsity = np.nan
    return sparsity


# [FUNC_CATEGORY: similarity_stats] [SEARCH_TAGS: gini]
def compute_gini_coefficient(rates):
    """
    Compute Gini coefficient for population firing rates.
    Measures inequality in firing rate distribution.
    0 = perfectly equal (all neurons same rate)
    1 = perfectly unequal (one neuron has all activity)
    Parameters:
    -----------
    rates : array-like, shape (n_neurons,) or (n_neurons, n_bins)
        Firing rates for each neuron.
    Returns:
    --------
    gini : float or array
        Gini coefficient(s). Range [0, 1].
    """
    rates = np.array(rates)
    # Handle 2D case (compute per bin)
    if rates.ndim == 2:
        gini_vals = np.zeros(rates.shape[1])
        for i in range(rates.shape[1]):
            gini_vals[i] = compute_gini_coefficient(rates[:, i])
        return gini_vals
    # Remove NaNs and zeros
    rates = rates[~np.isnan(rates)]
    rates = rates[rates >= 0]
    if len(rates) == 0:
        return np.nan
    # Sort rates
    sorted_rates = np.sort(rates)
    n = len(sorted_rates)
    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_rates)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_rates)) / (n * cumsum[-1]) - (
        n + 1
    ) / n
    return gini


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_activity_fraction(rates, threshold=0.1):
    """
    Compute fraction of active neurons.
    Parameters:
    -----------
    rates : array-like, shape (n_neurons,) or (n_neurons, n_bins)
        Firing rates for each neuron.
    threshold : float
        Minimum rate to be considered "active" (Hz)
    Returns:
    --------
    active_frac : float or array
        Fraction of neurons exceeding threshold.
    """
    rates = np.array(rates)
    if rates.ndim == 2:
        # Per bin activity fraction
        active = rates > threshold
        active_frac = np.nanmean(active, axis=0)
    else:
        # Single value
        active = rates > threshold
        active_frac = np.nanmean(active)
    return active_frac


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: spatial, bin]
def find_matched_spatial_bins(occupancy1, occupancy2, min_time=1.0):
    """
    Find spatial bins visited in both time windows.
    Parameters:
    -----------
    occupancy1 : array, shape (n_bins,)
        Occupancy time in each bin for window 1 (seconds)
    occupancy2 : array, shape (n_bins,)
        Occupancy time in each bin for window 2 (seconds)
    min_time : float
        Minimum occupancy time to consider bin as "visited"
    Returns:
    --------
    matched_bins : array of bool, shape (n_bins,)
        True for bins visited in both windows
    """
    visited1 = occupancy1 >= min_time
    visited2 = occupancy2 >= min_time
    matched_bins = visited1 & visited2
    return matched_bins


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_effective_dimensionality(rate_matrix):
    """
    Compute effective dimensionality using PCA participation ratio.
    Robust to NaNs/Inf and ill-conditioned covariance.
    """
    rate_matrix = np.asarray(rate_matrix)
    # Remove neurons with no variance
    valid_idx = np.nanstd(rate_matrix, axis=1) > 0
    if np.sum(valid_idx) < 2:
        return np.nan
    clean_rates = rate_matrix[valid_idx, :].astype(float, copy=True)
    if not np.isfinite(clean_rates).any():
        return np.nan
    if np.any(~np.isfinite(clean_rates)):
        col_means = np.nanmean(clean_rates, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        for i in range(clean_rates.shape[1]):
            bad = ~np.isfinite(clean_rates[:, i])
            if np.any(bad):
                clean_rates[bad, i] = col_means[i]
    # Final safety
    if not np.isfinite(clean_rates).all():
        clean_rates = np.nan_to_num(clean_rates, nan=0.0, posinf=0.0, neginf=0.0)
    # Center data
    clean_rates = clean_rates - np.mean(clean_rates, axis=1, keepdims=True)
    # Compute covariance and eigenvalues
    try:
        cov = np.cov(clean_rates)
        if not np.isfinite(cov).all():
            return np.nan
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return np.nan
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return np.nan
    eff_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)
    return eff_dim


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_windowed_rate_maps(
    spike_counts,
    positions,
    time_seconds,
    n_bins_x=15,
    n_bins_y=15,
    win_len_sec=600,
    pivot_step_sec=60,
    sigma=0.0,
    min_occupancy=0.5,
    smooth_separately=False,
    dtype=np.float32,
):
    """
    Optimized version of compute_windowed_rate_maps.
    Preserves the original output structure and axis order:
      rate_maps: (n_windows, n_neurons, n_bins_x, n_bins_y)
      occupancy_maps: (n_windows, n_bins_x, n_bins_y)
      valid_masks: (n_windows, n_bins_x, n_bins_y)
    Notes:
    - Assumes time_seconds is sorted ascending (typical).
    - Uses constant dt = median(diff(time_seconds)) like your original.
    - By default, matches your original smoothing semantics (smooth after divide).
      If smooth_separately=True, it smooths spike_map and occ_map separately and then divides
      (often more â€œprincipledâ€ near unvisited bins, but slightly different numerically).
    """
    spike_counts = np.asarray(spike_counts)
    positions = np.asarray(positions)
    time_seconds = np.asarray(time_seconds)
    n_frames, n_neurons = spike_counts.shape
    K = n_bins_x * n_bins_y
    # Frame duration (kept consistent with original)
    if n_frames >= 2:
        dt = float(np.median(np.diff(time_seconds)))
    else:
        dt = 0.0
    # Window centers
    t_start = float(time_seconds[0])
    t_end = float(time_seconds[-1])
    pivot_times = np.arange(
        t_start + win_len_sec / 2,
        t_end - win_len_sec / 2 + pivot_step_sec,
        pivot_step_sec,
        dtype=float,
    )
    n_windows = len(pivot_times)
    # Discretize positions once: labels_pos should be 0..K-1 with mapping flat = y*n_bins_x + x
    labels_pos, x_edges, y_edges = discretize_positions(
        positions, n_bins_x=n_bins_x, n_bins_y=n_bins_y
    )
    labels_pos = np.asarray(labels_pos, dtype=np.int64)
    # Allocate outputs
    rate_maps = np.zeros((n_windows, n_neurons, n_bins_x, n_bins_y), dtype=dtype)
    occupancy_maps = np.zeros((n_windows, n_bins_x, n_bins_y), dtype=dtype)
    valid_masks = np.zeros((n_windows, n_bins_x, n_bins_y), dtype=bool)
    # Precompute window bounds and frame indices via searchsorted (fast)
    half = win_len_sec / 2
    t_lows = pivot_times - half
    t_highs = pivot_times + half
    # time_seconds must be sorted for this to be correct/fast
    starts = np.searchsorted(time_seconds, t_lows, side="left")
    ends = np.searchsorted(time_seconds, t_highs, side="left")
    for i_win in range(n_windows):
        s = starts[i_win]
        e = ends[i_win]
        if e <= s:
            continue
        win_bins = labels_pos[s:e]  # (T_w,)
        win_spikes = spike_counts[s:e, :]  # (T_w, n_neurons)
        # --- Occupancy: bincount over bins (counts * dt) ---
        occ_counts = np.bincount(win_bins, minlength=K).astype(np.float64, copy=False)
        # reshape to (y, x) then transpose -> (x, y) to match your original
        occ_map = (occ_counts * dt).reshape(n_bins_y, n_bins_x).T
        occupancy_maps[i_win] = occ_map.astype(dtype, copy=False)
        valid_mask = occ_map >= float(min_occupancy)
        valid_masks[i_win] = valid_mask
        # --- Spike sums per bin, for all neurons at once ---
        # spike_sums[k, n] = sum spikes of neuron n when animal was in flat bin k
        spike_sums = np.zeros((K, n_neurons), dtype=np.float64)
        np.add.at(spike_sums, win_bins, win_spikes)
        # reshape spikes to (neurons, x, y) to match your original
        spike_map = spike_sums.T.reshape(n_neurons, n_bins_y, n_bins_x).transpose(
            0, 2, 1
        )
        if smooth_separately and sigma > 0:
            # Smooth spike and occupancy separately, then divide (reduces â€œzero-occupancy bleedâ€)
            spike_sm = gaussian_filter(spike_map, sigma=(0.0, sigma, sigma))
            occ_sm = gaussian_filter(occ_map, sigma=sigma)
            rate_map = np.divide(
                spike_sm,
                occ_sm[None, :, :],
                out=np.zeros_like(spike_sm, dtype=np.float64),
                where=occ_sm[None, :, :] > 0,
            )
        else:
            # Match original semantics: divide first (occ==0 -> 0), then smooth
            rate_map = np.divide(
                spike_map,
                occ_map[None, :, :],
                out=np.zeros_like(spike_map, dtype=np.float64),
                where=occ_map[None, :, :] > 0,
            )
            if sigma > 0:
                rate_map = gaussian_filter(rate_map, sigma=(0.0, sigma, sigma))
        # Set invalid bins to NaN (same as your original)
        rate_map = rate_map.astype(
            np.float32 if dtype == np.float32 else np.float64, copy=False
        )
        rate_map[:, ~valid_mask] = np.nan
        rate_maps[i_win] = rate_map.astype(dtype, copy=False)
    return {
        "rate_maps": rate_maps,
        "occupancy_maps": occupancy_maps,
        "pivot_times": pivot_times,
        "valid_masks": valid_masks,
        "n_spatial_bins": K,
        "n_bins_x": n_bins_x,
        "n_bins_y": n_bins_y,
    }


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def compute_neuron_quality_metrics(spike_counts, positions, rate_maps=None):
    """
    Compute quality metrics for each neuron to identify outliers.
    Parameters:
    -----------
    spike_counts : array, shape (n_timepoints, n_neurons)
        Deconvolved spike counts
    positions : array, shape (n_timepoints, 2)
        x, y positions
    rate_maps : array, optional, shape (n_neurons, n_bins_x, n_bins_y)
        Spatial rate maps (if already computed)
    Returns:
    --------
    metrics : dict
        Dictionary with quality metrics for each neuron
    """
    n_neurons = spike_counts.shape[1]
    # 1. Mean firing rate (Hz)
    dt = 0.04  # 40ms per frame
    mean_rate = np.mean(spike_counts, axis=0) / dt
    # 2. Peak firing rate (Hz)
    max_rate = np.max(spike_counts, axis=0) / dt
    # 3. Coefficient of variation (regularity of firing)
    # Higher CV = more irregular/bursty
    cv = np.std(spike_counts, axis=0) / (np.mean(spike_counts, axis=0) + 1e-10)
    # 4. Activity fraction (% of time bins active)
    activity_frac = np.mean(spike_counts > 0, axis=0)
    # 5. Spatial coverage (if rate maps provided)
    if rate_maps is not None:
        spatial_coverage = np.zeros(n_neurons)
        spatial_info = np.zeros(n_neurons)
        for i_neuron in range(n_neurons):
            rate_map = rate_maps[i_neuron]
            valid_bins = ~np.isnan(rate_map)
            # Coverage: fraction of spatial bins with activity
            active_bins = rate_map > 0.1  # 0.1 Hz threshold
            spatial_coverage[i_neuron] = (
                np.sum(active_bins) / np.sum(valid_bins)
                if np.sum(valid_bins) > 0
                else 0
            )
            # Spatial information (bits/spike)
            if np.sum(valid_bins) > 0:
                rate_flat = rate_map[valid_bins]
                mean_r = np.nanmean(rate_flat)
                if mean_r > 0:
                    # Simplified spatial info (assuming uniform occupancy)
                    p_i = 1.0 / np.sum(valid_bins)
                    si = 0
                    for r in rate_flat:
                        if r > 0:
                            si += p_i * (r / mean_r) * np.log2(r / mean_r)
                    spatial_info[i_neuron] = si
    else:
        spatial_coverage = None
        spatial_info = None
    metrics = {
        "mean_rate": mean_rate,
        "max_rate": max_rate,
        "cv": cv,
        "activity_frac": activity_frac,
        "spatial_coverage": spatial_coverage,
        "spatial_info": spatial_info,
    }
    return metrics


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: filter]
def filter_neurons(quality_metrics, criteria):
    """
    Filter neurons based on quality criteria.
    Parameters:
    -----------
    quality_metrics : dict
        Output from compute_neuron_quality_metrics
    criteria : dict
        Filtering thresholds, e.g.:
        {
            'min_mean_rate': 0.05,
            'max_mean_rate': 5.0,
            'max_peak_rate': 10.0,
            'max_cv': 20.0,
            'min_activity_frac': 0.01,
            'min_spatial_coverage': 0.05,
            'max_spatial_coverage': 0.95,
        }
    Returns:
    --------
    keep_mask : array of bool
        Mask indicating which neurons to keep
    """
    n_neurons = len(quality_metrics["mean_rate"])
    keep_mask = np.ones(n_neurons, dtype=bool)
    # Apply each criterion
    if "min_mean_rate" in criteria:
        keep_mask &= quality_metrics["mean_rate"] >= criteria["min_mean_rate"]
    if "max_mean_rate" in criteria:
        keep_mask &= quality_metrics["mean_rate"] <= criteria["max_mean_rate"]
    if "max_peak_rate" in criteria:
        keep_mask &= quality_metrics["max_rate"] <= criteria["max_peak_rate"]
    if "max_cv" in criteria:
        keep_mask &= quality_metrics["cv"] <= criteria["max_cv"]
    if "min_activity_frac" in criteria:
        keep_mask &= quality_metrics["activity_frac"] >= criteria["min_activity_frac"]
    if quality_metrics["spatial_coverage"] is not None:
        if "min_spatial_coverage" in criteria:
            keep_mask &= (
                quality_metrics["spatial_coverage"] >= criteria["min_spatial_coverage"]
            )
        if "max_spatial_coverage" in criteria:
            keep_mask &= (
                quality_metrics["spatial_coverage"] <= criteria["max_spatial_coverage"]
            )
    return keep_mask


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: compute]
def _compute_combined(
    spike_counts_use,
    visit_spans_per_cluster=None,
    clusters=None,
    min_visits_per_cluster=5,
):
    """
    Functionality:
    - Compute combined.
    Inputs:
    - `spike_counts_use`: Neural activity values.
      Typical shape: Typically (n_frames, n_neurons).
    - `visit_spans_per_cluster`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `clusters`: Cluster definitions.
      Typical shape: Usually list length n_clusters.
    - `min_visits_per_cluster`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 4 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if visit_spans_per_cluster is None:
        visit_spans_per_cluster = globals().get("visit_spans_per_cluster", [])
    if clusters is None:
        clusters = globals().get("clusters", [])
    if min_visits_per_cluster is None:
        min_visits_per_cluster = globals().get("min_visits_per_cluster", 5)
    # Per-cluster visit metrics
    cluster_visit_sparsity = []
    cluster_visit_gini = []
    cluster_visit_total = []
    cluster_visit_weights = []
    cluster_visit_counts = []
    for ci, visit_spans in enumerate(visit_spans_per_cluster):
        if len(visit_spans) == 0:
            cluster_visit_sparsity.append(np.array([]))
            cluster_visit_gini.append(np.array([]))
            cluster_visit_total.append(np.array([]))
            cluster_visit_weights.append(np.array([]))
            cluster_visit_counts.append(0)
            continue
        visit_spars = []
        visit_gini = []
        visit_total = []
        visit_w = []
        for s, e in visit_spans:
            pv = np.nanmean(spike_counts_use[s : e + 1], axis=0)
            visit_spars.append(compute_lifetime_sparsity(pv))
            visit_gini.append(compute_gini_coefficient(pv))
            visit_total.append(np.nansum(pv))
            visit_w.append(e - s + 1)
        visit_spars = np.array(visit_spars, dtype=float)
        visit_gini = np.array(visit_gini, dtype=float)
        visit_total = np.array(visit_total, dtype=float)
        visit_w = np.array(visit_w, dtype=float)
        cluster_visit_sparsity.append(visit_spars)
        cluster_visit_gini.append(visit_gini)
        cluster_visit_total.append(visit_total)
        cluster_visit_weights.append(visit_w)
        cluster_visit_counts.append(len(visit_spars))
    # Combine by visit index
    max_visits = max(cluster_visit_counts) if len(cluster_visit_counts) > 0 else 0
    combined = {"sparsity": [], "gini": [], "total": [], "nclusters": []}
    for k in range(max_visits):
        vals_s = []
        vals_g = []
        vals_t = []
        weights = []
        for ci in range(len(clusters)):
            vs = cluster_visit_sparsity[ci]
            vg = cluster_visit_gini[ci]
            vt = cluster_visit_total[ci]
            vw = cluster_visit_weights[ci]
            if len(vs) <= k:
                continue
            if len(vs) < min_visits_per_cluster:
                continue
            vals_s.append(vs[k])
            vals_g.append(vg[k])
            vals_t.append(vt[k])
            weights.append(vw[k])
        if len(vals_s) == 0:
            combined["sparsity"].append(np.nan)
            combined["gini"].append(np.nan)
            combined["total"].append(np.nan)
            combined["nclusters"].append(0)
            continue
        weights = np.array(weights, dtype=float)
        combined["sparsity"].append(np.average(vals_s, weights=weights))
        combined["gini"].append(np.average(vals_g, weights=weights))
        combined["total"].append(np.average(vals_t, weights=weights))
        combined["nclusters"].append(len(vals_s))
    combined_s = np.array(combined["sparsity"], dtype=float)
    combined_g = np.array(combined["gini"], dtype=float)
    combined_t = np.array(combined["total"], dtype=float)
    combined_n = np.array(combined["nclusters"], dtype=int)
    return combined_s, combined_g, combined_t, combined_n


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _pad_stats(curves_list):
    """
    Functionality:
    - Pad stats.
    Inputs:
    - `curves_list`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 3 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if len(curves_list) == 0:
        return None, None, None
    max_len = max(len(c) for c in curves_list)
    mat = np.full((len(curves_list), max_len), np.nan)
    for i, c in enumerate(curves_list):
        mat[i, : len(c)] = c
    mean = np.nanmean(mat, axis=0)
    lo = np.nanpercentile(mat, 2.5, axis=0)
    hi = np.nanpercentile(mat, 97.5, axis=0)
    return mean, lo, hi


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _to_day2(name):
    """
    Functionality:
    - To day2.
    Inputs:
    - `name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    - Variable `name` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if isinstance(name, str):
        return name.replace("day1", "day2")
    return name


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def smooth_nan(x, window=9):
    """
    Functionality:
    - Smooth nan.
    Inputs:
    - `x`: Numeric input vector/array.
      Typical shape: Scalar or 1D array.
    - `window`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    - Variable `y` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    half = window // 2
    y = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            y[i] = np.mean(seg)
    return y


# [FUNC_CATEGORY: pipeline_analysis] [SEARCH_TAGS: filter, apply]
def apply_quality_filter_with_ratemap(spike_counts, positions, rate_map_results):
    """
    Functionality:
    - Apply quality filter with ratemap.
    Inputs:
    - `spike_counts`: Neural activity values.
      Typical shape: Typically (n_frames, n_neurons).
    - `positions`: Position coordinates over time.
      Typical shape: Typically (n_frames, 2) for x/y.
    - `rate_map_results`: Rate/tuning map.
      Typical shape: Commonly (n_neurons, n_bins) or (n_bins_x, n_bins_y).
    Outputs:
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    keep = np.ones(spike_counts.shape[1], dtype=bool)
    quality_metrics = None
    qm_fn = globals().get("compute_neuron_quality_metrics")
    fc = globals().get("filter_criteria")
    if qm_fn is None or fc is None:
        return keep, quality_metrics
    try:
        mean_rate_map = np.nanmean(rate_map_results["rate_maps"], axis=0)
        quality_metrics = qm_fn(spike_counts, positions, rate_maps=mean_rate_map)
        filter_fn = globals().get("filter_neurons")
        if callable(filter_fn):
            keep = filter_fn(quality_metrics, fc)
        else:
            # Fallback to range-based filtering if filter_neurons is missing
            def _apply_range(key, vals):
                nonlocal keep
                if key in fc and fc[key] is not None:
                    lo, hi = fc[key]
                    if lo is not None:
                        keep &= vals >= lo
                    if hi is not None:
                        keep &= vals <= hi
            _apply_range("mean_rate", quality_metrics["mean_rate"])
            _apply_range("max_rate", quality_metrics["max_rate"])
            _apply_range("cv", quality_metrics["cv"])
            _apply_range("activity_frac", quality_metrics["activity_frac"])
            if quality_metrics.get("spatial_coverage") is not None:
                _apply_range("spatial_coverage", quality_metrics["spatial_coverage"])
    except Exception as e:
        print(f"[filter] skipped: {e}")
    return keep, quality_metrics


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: bin, level]
def compute_level_combined(
    level_idx,
    spike_counts_use,
    selected_df,
    level_list,
    ma,
    gap_frames=None,
    min_frames_per_visit=None,
    min_visits_per_cluster=None,
):
    """
    Functionality:
    - Compute level combined.
    Inputs:
    - `level_idx`: Index of selected level.
      Typical shape: Scalar integer.
    - `spike_counts_use`: Neural activity values.
      Typical shape: Typically (n_frames, n_neurons).
    - `selected_df`: Behavior dataframe aligned to frames.
      Typical shape: Rows correspond to frames/time samples.
    - `level_list`: Level masks/definitions for maze nodes.
      Typical shape: Usually length n_levels.
    - `ma`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `gap_frames`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_frames_per_visit`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `min_visits_per_cluster`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Tuple with 3 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if gap_frames is None:
        gap_frames = globals().get("gap_frames", 10)
    if min_frames_per_visit is None:
        min_frames_per_visit = globals().get("min_frames_per_visit", 5)
    if min_visits_per_cluster is None:
        min_visits_per_cluster = globals().get("min_visits_per_cluster", 5)
    if spike_counts_use.shape[1] == 0:
        return np.full(0, np.nan), np.full(0, np.nan), np.full(0, np.nan)
    level_mask = level_list[level_idx].astype(bool)
    idx_to_coord = {idx: coord for coord, idx in ma.ce.items()}
    level_coords = []
    for idx in range(len(level_mask)):
        if level_mask[idx]:
            coord = idx_to_coord.get(idx)
            if coord is not None:
                level_coords.append(coord)
    level_coord_set = set(level_coords)
    # Connected components (4-neighborhood)
    visited = set()
    clusters = []
    for coord in level_coords:
        if coord in visited:
            continue
        stack = [coord]
        visited.add(coord)
        comp = []
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                ncoord = (ni, nj)
                if ncoord in level_coord_set and ncoord not in visited:
                    visited.add(ncoord)
                    stack.append(ncoord)
        clusters.append(comp)
    # Map each frame to cluster id
    n_frames = len(selected_df)
    frame_cluster = np.full(n_frames, -1, dtype=int)
    bx = selected_df["mass_bx"].to_numpy()
    by = selected_df["mass_by"].to_numpy()
    coord_to_cluster = {}
    for ci, comp in enumerate(clusters):
        for coord in comp:
            coord_to_cluster[coord] = ci
    for t in range(n_frames):
        coord = (int(by[t]), int(bx[t]))
        if coord in coord_to_cluster:
            frame_cluster[t] = coord_to_cluster[coord]
    cluster_visit_sparsity = []
    cluster_visit_gini = []
    cluster_visit_total = []
    cluster_visit_weights = []
    cluster_visit_counts = []
    for ci, comp in enumerate(clusters):
        frames_in_cluster = np.where(frame_cluster == ci)[0]
        if len(frames_in_cluster) == 0:
            cluster_visit_sparsity.append(np.array([]))
            cluster_visit_gini.append(np.array([]))
            cluster_visit_total.append(np.array([]))
            cluster_visit_weights.append(np.array([]))
            cluster_visit_counts.append(0)
            continue
        # Build visits
        visit_spans = []
        start = frames_in_cluster[0]
        prev = frames_in_cluster[0]
        for f in frames_in_cluster[1:]:
            if f - prev > gap_frames:
                visit_spans.append((start, prev))
                start = f
            prev = f
        visit_spans.append((start, prev))
        visit_spans = [
            v for v in visit_spans if (v[1] - v[0] + 1) >= min_frames_per_visit
        ]
        visit_spars = []
        visit_gini = []
        visit_total = []
        visit_w = []
        for s, e in visit_spans:
            if e < s:
                continue
            s = max(0, s)
            e = min(e, spike_counts_use.shape[0] - 1)
            if e < s:
                continue
            seg = spike_counts_use[s : e + 1]
            if seg.size == 0 or not np.isfinite(seg).any():
                visit_spars.append(np.nan)
                visit_gini.append(np.nan)
                visit_total.append(0.0)
                visit_w.append(e - s + 1)
                continue
            pv = np.nanmean(seg, axis=0)
            if pv.size == 0 or np.nansum(pv) == 0:
                visit_spars.append(np.nan)
                visit_gini.append(np.nan)
                visit_total.append(0.0)
                visit_w.append(e - s + 1)
                continue
            visit_spars.append(compute_lifetime_sparsity(pv))
            visit_gini.append(compute_gini_coefficient(pv))
            visit_total.append(np.nansum(pv))
            visit_w.append(e - s + 1)
        visit_spars = np.array(visit_spars, dtype=float)
        visit_gini = np.array(visit_gini, dtype=float)
        visit_total = np.array(visit_total, dtype=float)
        visit_w = np.array(visit_w, dtype=float)
        cluster_visit_sparsity.append(visit_spars)
        cluster_visit_gini.append(visit_gini)
        cluster_visit_total.append(visit_total)
        cluster_visit_weights.append(visit_w)
        cluster_visit_counts.append(len(visit_spars))
    max_visits = max(cluster_visit_counts) if len(cluster_visit_counts) > 0 else 0
    combined_s = []
    combined_g = []
    combined_t = []
    for k in range(max_visits):
        vals_s = []
        vals_g = []
        vals_t = []
        weights = []
        for ci in range(len(clusters)):
            vs = cluster_visit_sparsity[ci]
            vg = cluster_visit_gini[ci]
            vt = cluster_visit_total[ci]
            vw = cluster_visit_weights[ci]
            if len(vs) <= k:
                continue
            if len(vs) < min_visits_per_cluster:
                continue
            vals_s.append(vs[k])
            vals_g.append(vg[k])
            vals_t.append(vt[k])
            weights.append(vw[k])
        if len(vals_s) == 0:
            combined_s.append(np.nan)
            combined_g.append(np.nan)
            combined_t.append(np.nan)
            continue
        weights = np.array(weights, dtype=float)
        combined_s.append(np.average(vals_s, weights=weights))
        combined_g.append(np.average(vals_g, weights=weights))
        combined_t.append(np.average(vals_t, weights=weights))
    return (
        np.array(combined_s, dtype=float),
        np.array(combined_g, dtype=float),
        np.array(combined_t, dtype=float),
    )


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: session]
def _parse_session_name(name):
    # Accept IDs with optional letters (e.g., 489Lday1) and optional -month (e.g., 418-3day1)
    """
    Functionality:
    - Parse session name.
    Inputs:
    - `name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None in some branches.
    - Tuple with 2 element(s); shapes depend on branch/data.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    m = re.match(r"^(?P<animal>\d+[A-Za-z]*)(?:-(?P<month>\d+))?day(?P<day>\d+)$", name)
    if not m:
        return None
    month = int(m.group("month")) if m.group("month") else 0
    day = int(m.group("day"))
    return month, day


# [FUNC_CATEGORY: io_loading] [SEARCH_TAGS: load]
def _load_time_metrics(sess_dir, sess_name):
    """
    Functionality:
    - Load time metrics.
    Inputs:
    - `sess_dir`: Directory path.
      Typical shape: Scalar string/path.
    - `sess_name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None in some branches.
    - Dictionary of computed fields/metrics.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    npz_path = os.path.join(
        sess_dir, f"time_window_ratemap_metrics_raw_{sess_name}.npz"
    )
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "activity_frac": data.get("activity_frac"),
        "lifetime_spars": data.get("lifetime_spars"),
        "gini": data.get("gini"),
        "eff_dim": data.get("eff_dim"),
        "spatial_coverage": data.get("spatial_coverage"),
        "total_rate": data.get("total_rate"),
        "pivot_times": data.get("pivot_times"),
        "pivot_step_sec": data.get("pivot_step_sec"),
        "win_len_sec": data.get("win_len_sec"),
    }


# [FUNC_CATEGORY: mapping_spatial] [SEARCH_TAGS: level]
def _load_level_metrics(sess_dir, sess_name):
    """
    Functionality:
    - Load level metrics.
    Inputs:
    - `sess_dir`: Directory path.
      Typical shape: Scalar string/path.
    - `sess_name`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None in some branches.
    - Dictionary of computed fields/metrics.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    npz_path = os.path.join(sess_dir, f"level_visit_metrics_raw_{sess_name}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    sparsity = data.get("sparsity")
    gini = data.get("gini")
    total = data.get("total")
    if sparsity is None or gini is None or total is None:
        return None
    # stored as object array of per-level arrays
    return {
        "sparsity": [np.asarray(s, dtype=float) for s in sparsity],
        "gini": [np.asarray(g, dtype=float) for g in gini],
        "total": [np.asarray(t, dtype=float) for t in total],
    }


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _pad_and_mean(curves):
    """
    Functionality:
    - Pad and mean.
    Inputs:
    - `curves`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - Computed object/array/scalar depending on input.
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if len(curves) == 0:
        return np.array([])
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
    return np.nanmean(padded, axis=0)


# [FUNC_CATEGORY: utility] [SEARCH_TAGS: helper]
def _first_hour_curve(curve, metrics):
    """
    Functionality:
    - First hour curve.
    Inputs:
    - `curve`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    - `metrics`: Function input parameter.
      Typical shape: Dimension depends on caller context.
    Outputs:
    - None in some branches.
    - Computed object/array/scalar depending on input.
    - Variable `curve` (type/shape from runtime context).
    Dimension notes:
    - `n_frames`: number of time samples/rows.
    - `n_neurons`: number of recorded units/cells.
    - `n_bins`/`n_levels`/`n_clusters`: analysis-space sizes used by the caller.
    """
    if curve is None:
        return None
    curve = np.asarray(curve, dtype=float)
    pivot_times = metrics.get("pivot_times")
    if pivot_times is not None and len(pivot_times) == len(curve):
        mask = np.asarray(pivot_times, dtype=float) <= 3600
        if np.any(mask):
            return curve[mask]
        return curve[:0]
    # Fallback: use pivot_step_sec if available
    step = metrics.get("pivot_step_sec")
    if step is None or not np.isfinite(step):
        return curve
    n = int(np.floor(3600 / float(step)))
    return curve[: max(0, n)]


from .similarity import (
    build_count_matrix as _sim_build_count_matrix,
    compute_spatial_info_session_rategated as _sim_compute_spatial_info_session_rategated,
    cross_session_level_shift_similarity as _sim_cross_session_level_shift_similarity,
    cross_session_pv_correlation as _sim_cross_session_pv_correlation,
    cross_session_pv_cross_similarity_matrix as _sim_cross_session_pv_cross_similarity_matrix,
    cross_session_spatial_information_rategated as _sim_cross_session_spatial_information_rategated,
    cross_session_tuning_similarity as _sim_cross_session_tuning_similarity,
    discretize_positions as _sim_discretize_positions,
    within_session_pv_temporal_similarity as _sim_within_session_pv_temporal_similarity,
)
from .spatial import assign_levels as _spatial_assign_levels
from .stats import (
    compute_gini_coefficient as _stats_compute_gini_coefficient,
    compute_lifetime_sparsity as _stats_compute_lifetime_sparsity,
)


cross_session_tuning_similarity = _sim_cross_session_tuning_similarity
cross_session_level_shift_similarity = _sim_cross_session_level_shift_similarity
cross_session_pv_correlation = _sim_cross_session_pv_correlation
cross_session_pv_cross_similarity_matrix = _sim_cross_session_pv_cross_similarity_matrix
within_session_pv_temporal_similarity = _sim_within_session_pv_temporal_similarity
compute_spatial_info_session_rategated = _sim_compute_spatial_info_session_rategated
cross_session_spatial_information_rategated = (
    _sim_cross_session_spatial_information_rategated
)
discretize_positions = _sim_discretize_positions
build_count_matrix = _sim_build_count_matrix
assign_levels = _spatial_assign_levels
compute_lifetime_sparsity = _stats_compute_lifetime_sparsity
compute_gini_coefficient = _stats_compute_gini_coefficient


# Public export surface (top-level callables/classes).
__all__ = [
    n for n, v in globals().items()
    if not n.startswith("_") and callable(v)
]
