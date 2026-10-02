"""Level-map ordering used by the heatmap display helpers."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


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
