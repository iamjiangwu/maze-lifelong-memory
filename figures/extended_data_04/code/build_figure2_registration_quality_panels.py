"""Build CellReg registration-QC source tables and separate Figure 2 panels.

The output scope is the final sorted 36-pair Figure 2 scaffold. Candidate
features and registered-pair diagnostics are read directly from the matching
CellReg outputs under ``F:\\footprint-new``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPE_CSV = (
    REPO_ROOT
    / "docs"
    / "main_figs_v3_source_render_2026-06-24"
    / "figure_scope_exclusions_2026-06-29"
    / "figure2_distribution_shuffle_36_no_1616_short"
    / "current_panel_base_pairs_36_no_1616_short.csv"
)
DEFAULT_PAIR_SUMMARY_CSV = DEFAULT_SCOPE_CSV.with_name(
    "figure2_c_h_pair_summary_sorted_36_no_1616_short.csv"
)
DEFAULT_FOOTPRINT_ROOT = Path(r"F:\footprint-new")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "main_figs_v3_source_render_2026-06-24"
    / "registration_quality_exact_36_2026-07-09"
)

NEAREST_COLOR = "#0072B2"
OTHER_COLOR = "#D55E00"
REGISTERED_COLOR = "#4D5963"
GRID_COLOR = "#D8D8D8"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument(
        "--pair-summary-csv", type=Path, default=DEFAULT_PAIR_SUMMARY_CSV
    )
    parser.add_argument("--footprint-root", type=Path, default=DEFAULT_FOOTPRINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _matlab_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(value)) for value in values)


def _read_ref_vector(mat_file: h5py.File, ref) -> np.ndarray:
    dataset = mat_file[ref]
    if dataset.attrs.get("MATLAB_class") == b"canonical empty":
        return np.asarray([], dtype=float)
    return np.asarray(dataset, dtype=float).ravel(order="F")


def _pair_code(row: pd.Series) -> str:
    animal = str(row["animal"])
    src = str(row["src_session"])
    dst = str(row["dst_session"])
    if animal == "1616" and src == "1616bday1" and dst == "1616b-3day1":
        return "13"
    transition = str(row["transition"])
    return transition.replace("-", "")


def _pair_code_session_labels(code: str) -> tuple[str, str]:
    if code.endswith("12") and len(code) > 2:
        return code[:-2], "12"
    if len(code) == 2 and code.isdigit():
        return code[0], code[1]
    raise ValueError(f"Unsupported CellReg pair code: {code}")


def _load_exported_cimap(path: Path) -> np.ndarray:
    mat = sio.loadmat(_require_file(path), squeeze_me=True)
    if "cell_to_index_map" not in mat:
        raise KeyError(f"cell_to_index_map is missing from {path}")
    cimap = np.asarray(mat["cell_to_index_map"], dtype=int)
    if cimap.ndim != 2 or cimap.shape[1] != 2:
        raise ValueError(f"Expected a two-session CIMap in {path}; got {cimap.shape}")
    return cimap


def _expand_cleaned_cimap(
    cleaned_cimap: np.ndarray,
    *,
    animal_dir: Path,
    pair_code: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    expanded = np.asarray(cleaned_cimap, dtype=int).copy()
    removed_counts: list[int] = []
    for column, session_label in enumerate(_pair_code_session_labels(pair_code)):
        mask_path = animal_dir / f"allZeroFootprints_{pair_code}_{session_label}.mat"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask_mat = sio.loadmat(mask_path, squeeze_me=True)
        if "allZeroFootprints" not in mask_mat:
            raise KeyError(f"allZeroFootprints is missing from {mask_path}")
        all_zero = np.asarray(mask_mat["allZeroFootprints"], dtype=bool).ravel()
        kept_original_ids = np.flatnonzero(~all_zero) + 1
        positive = expanded[:, column] > 0
        if np.any(expanded[positive, column] > kept_original_ids.size):
            raise ValueError(
                f"Cleaned CIMap index exceeds retained-footprint count in {mask_path}"
            )
        expanded[positive, column] = kept_original_ids[expanded[positive, column] - 1]
        removed_counts.append(int(np.sum(all_zero)))
    return expanded, (removed_counts[0], removed_counts[1])


def _validate_cellreg_to_exported_cimap(
    cleaned_cimap: np.ndarray,
    *,
    exported_cimap: np.ndarray,
    animal_dir: Path,
    pair_code: str,
) -> tuple[np.ndarray, str, tuple[int, int]]:
    mask_paths = [
        animal_dir / f"allZeroFootprints_{pair_code}_{session_label}.mat"
        for session_label in _pair_code_session_labels(pair_code)
    ]
    mask_exists = [path.exists() for path in mask_paths]
    if any(mask_exists) and not all(mask_exists):
        raise FileNotFoundError(
            f"Incomplete all-zero-footprint masks for {animal_dir.name} {pair_code}: "
            f"{mask_paths}"
        )

    if all(mask_exists):
        expanded, removed_counts = _expand_cleaned_cimap(
            cleaned_cimap,
            animal_dir=animal_dir,
            pair_code=pair_code,
        )
        if sum(removed_counts) > 0:
            if np.array_equal(expanded, exported_cimap):
                return expanded, "exact_after_zero_reinsertion", removed_counts
            if np.array_equal(cleaned_cimap, exported_cimap):
                return (
                    expanded,
                    "source_exported_unexpanded_cleaned_coordinates",
                    removed_counts,
                )
            raise ValueError(
                f"CellReg CIMap does not match exported CIMap for {animal_dir.name} "
                f"{pair_code} in either cleaned or zero-reinserted coordinates."
            )

    if np.array_equal(cleaned_cimap, exported_cimap):
        return cleaned_cimap.copy(), "exact_raw", (0, 0)
    raise ValueError(
        f"CellReg CIMap does not match exported CIMap for {animal_dir.name} {pair_code}, "
        "and no non-empty all-zero-footprint remapping explains the difference."
    )


def _latest_cellregistered(cellreg_dir: Path) -> tuple[Path, int]:
    candidates = sorted(
        cellreg_dir.glob("cellRegistered*.mat"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No cellRegistered MAT found in {cellreg_dir}")
    return candidates[-1], len(candidates)


def _extract_candidate_features(
    modeled_path: Path,
    *,
    animal: str,
    pair_label: str,
    pair_code: str,
    src_session: str,
    dst_session: str,
) -> tuple[
    pd.DataFrame, dict[tuple[int, int, int], tuple[float, float]], dict[str, object]
]:
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[int, int, int], tuple[float, float]] = {}
    with h5py.File(_require_file(modeled_path), "r") as mat_file:
        group = mat_file["modeled_data_struct"]
        microns_per_pixel = float(np.asarray(group["microns_per_pixel"]).ravel()[0])
        maximal_distance_um = float(np.asarray(group["maximal_distance"]).ravel()[0])
        indexes_top = np.asarray(group["all_to_all_indexes"]).ravel(order="F")
        distances_top = np.asarray(group["all_to_all_centroid_distances"]).ravel(
            order="F"
        )
        correlations_top = np.asarray(group["all_to_all_spatial_correlations"]).ravel(
            order="F"
        )
        if not (len(indexes_top) == len(distances_top) == len(correlations_top) == 2):
            raise ValueError(f"Expected two sessions in {modeled_path}")

        session_names = (src_session, dst_session)
        for query_session_index in range(2):
            candidate_session_index = 1 - query_session_index
            index_cells = mat_file[indexes_top[query_session_index]]
            distance_cells = mat_file[distances_top[query_session_index]]
            correlation_cells = mat_file[correlations_top[query_session_index]]
            n_query_cells = int(index_cells.shape[1])
            for query_cell_zero_based in range(n_query_cells):
                candidate_ids = _read_ref_vector(
                    mat_file,
                    index_cells[candidate_session_index, query_cell_zero_based],
                ).astype(int, copy=False)
                distances_pixels = _read_ref_vector(
                    mat_file,
                    distance_cells[candidate_session_index, query_cell_zero_based],
                )
                spatial_correlations = _read_ref_vector(
                    mat_file,
                    correlation_cells[candidate_session_index, query_cell_zero_based],
                )
                if not (
                    candidate_ids.size
                    == distances_pixels.size
                    == spatial_correlations.size
                ):
                    raise ValueError(
                        f"Candidate feature lengths disagree in {modeled_path} for "
                        f"session {query_session_index + 1}, cell {query_cell_zero_based + 1}."
                    )
                if candidate_ids.size == 0:
                    continue
                finite_distance = np.isfinite(distances_pixels)
                if not np.any(finite_distance):
                    continue
                closest_index = int(np.nanargmin(distances_pixels))
                distance_order = np.argsort(distances_pixels, kind="stable")
                ranks = np.empty_like(distance_order)
                ranks[distance_order] = np.arange(1, candidate_ids.size + 1)
                for candidate_offset, candidate_id in enumerate(candidate_ids):
                    distance_pixels = float(distances_pixels[candidate_offset])
                    distance_um = distance_pixels * microns_per_pixel
                    spatial_correlation = float(spatial_correlations[candidate_offset])
                    is_closest = candidate_offset == closest_index
                    query_cell_id = int(query_cell_zero_based + 1)
                    candidate_cell_id = int(candidate_id)
                    lookup[
                        (query_session_index + 1, query_cell_id, candidate_cell_id)
                    ] = (distance_pixels, spatial_correlation)
                    rows.append(
                        {
                            "animal": animal,
                            "pair_label": pair_label,
                            "pair_code": pair_code,
                            "src_session": src_session,
                            "dst_session": dst_session,
                            "query_session_index": int(query_session_index + 1),
                            "query_session": session_names[query_session_index],
                            "candidate_session_index": int(candidate_session_index + 1),
                            "candidate_session": session_names[candidate_session_index],
                            "query_cell_id_cleaned": query_cell_id,
                            "candidate_cell_id_cleaned": candidate_cell_id,
                            "candidate_rank_by_centroid": int(ranks[candidate_offset]),
                            "candidate_class": (
                                "closest_by_centroid"
                                if is_closest
                                else "other_within_radius"
                            ),
                            "is_closest_by_centroid": bool(is_closest),
                            "centroid_distance_pixels": distance_pixels,
                            "centroid_distance_um": distance_um,
                            "spatial_footprint_correlation": spatial_correlation,
                            "correlation_in_0_to_1_plot": bool(
                                np.isfinite(spatial_correlation)
                                and spatial_correlation >= 0.0
                            ),
                        }
                    )

        candidate_df = pd.DataFrame(rows)
        stock_nn_distance = np.asarray(
            group["NN_centroid_distances"], dtype=float
        ).ravel()
        stock_nnn_distance = np.asarray(
            group["NNN_centroid_distances"], dtype=float
        ).ravel()
        closest_distance = candidate_df.loc[
            candidate_df["is_closest_by_centroid"], "centroid_distance_pixels"
        ].to_numpy(dtype=float)
        other_distance = candidate_df.loc[
            ~candidate_df["is_closest_by_centroid"], "centroid_distance_pixels"
        ].to_numpy(dtype=float)
        if not np.allclose(
            np.sort(closest_distance), np.sort(stock_nn_distance), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"Closest-distance reconstruction disagrees with CellReg in {modeled_path}"
            )
        if not np.allclose(
            np.sort(other_distance), np.sort(stock_nnn_distance), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"Other-distance reconstruction disagrees with CellReg in {modeled_path}"
            )

        audit = {
            "microns_per_pixel": microns_per_pixel,
            "maximal_distance_um": maximal_distance_um,
            "alignment_type": _matlab_char(group["alignment_type"]),
            "best_model": _matlab_char(group["best_model_string"]),
            "mse_centroid_distance_model": float(
                np.asarray(group["MSE_centroid_distances_model"]).ravel()[0]
            ),
            "mse_spatial_correlation_model": float(
                np.asarray(group["MSE_spatial_correlations_model"]).ravel()[0]
            ),
            "n_candidate_directed": int(candidate_df.shape[0]),
            "n_closest_directed": int(candidate_df["is_closest_by_centroid"].sum()),
            "n_other_directed": int((~candidate_df["is_closest_by_centroid"]).sum()),
            "n_candidate_negative_correlation": int(
                (candidate_df["spatial_footprint_correlation"] < 0).sum()
            ),
            "cellreg_stock_nn_correlation_definition": "maximum_correlation_per_query",
            "displayed_correlation_definition": "correlation_of_centroid_closest_candidate",
        }
    return candidate_df, lookup, audit


def _extract_registered_features(
    cellregistered_path: Path,
    *,
    corrected_cimap: np.ndarray,
    source_exported_cimap: np.ndarray,
    candidate_lookup: dict[tuple[int, int, int], tuple[float, float]],
    microns_per_pixel: float,
    animal: str,
    pair_label: str,
    pair_code: str,
    src_session: str,
    dst_session: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    with h5py.File(_require_file(cellregistered_path), "r") as mat_file:
        group = mat_file["cell_registered_struct"]
        cleaned_cimap = np.asarray(group["cell_to_index_map"], dtype=float).T.astype(
            int
        )
        cell_scores = np.asarray(group["cell_scores"], dtype=float).ravel()
        p_same_refs = np.asarray(group["p_same_registered_pairs"]).ravel()
        if not (
            cleaned_cimap.shape[0]
            == corrected_cimap.shape[0]
            == source_exported_cimap.shape[0]
            == cell_scores.size
            == p_same_refs.size
        ):
            raise ValueError(
                f"CellReg register lengths disagree in {cellregistered_path}"
            )
        active_clusters = np.flatnonzero(np.all(cleaned_cimap > 0, axis=1))
        symmetry_deltas: list[float] = []
        for cluster_zero_based in active_clusters:
            session1_cleaned, session2_cleaned = cleaned_cimap[cluster_zero_based]
            key_forward = (1, int(session1_cleaned), int(session2_cleaned))
            key_reverse = (2, int(session2_cleaned), int(session1_cleaned))
            if (
                key_forward not in candidate_lookup
                or key_reverse not in candidate_lookup
            ):
                raise KeyError(
                    f"Registered pair is absent from candidate features in {cellregistered_path}: "
                    f"{key_forward} / {key_reverse}"
                )
            distance_pixels, spatial_correlation = candidate_lookup[key_forward]
            reverse_distance, reverse_correlation = candidate_lookup[key_reverse]
            symmetry_deltas.append(abs(distance_pixels - reverse_distance))
            symmetry_deltas.append(abs(spatial_correlation - reverse_correlation))
            p_same_array = np.asarray(
                mat_file[p_same_refs[cluster_zero_based]], dtype=float
            )
            finite_p_same = p_same_array[np.isfinite(p_same_array)]
            if finite_p_same.size == 0:
                raise ValueError(
                    f"Registered pair has no finite P_same in {cellregistered_path}, "
                    f"cluster {cluster_zero_based + 1}."
                )
            session1_source, session2_source = source_exported_cimap[cluster_zero_based]
            session1_corrected, session2_corrected = corrected_cimap[cluster_zero_based]
            rows.append(
                {
                    "animal": animal,
                    "pair_label": pair_label,
                    "pair_code": pair_code,
                    "src_session": src_session,
                    "dst_session": dst_session,
                    "cluster_index": int(cluster_zero_based + 1),
                    "session1_cell_id_cleaned": int(session1_cleaned),
                    "session2_cell_id_cleaned": int(session2_cleaned),
                    "session1_cell_id_source_exported": int(session1_source),
                    "session2_cell_id_source_exported": int(session2_source),
                    "session1_cell_id_corrected": int(session1_corrected),
                    "session2_cell_id_corrected": int(session2_corrected),
                    "centroid_distance_pixels": float(distance_pixels),
                    "centroid_distance_um": float(distance_pixels * microns_per_pixel),
                    "spatial_footprint_correlation": float(spatial_correlation),
                    "p_same": float(np.nanmax(finite_p_same)),
                    "cellreg_registration_score": float(
                        cell_scores[cluster_zero_based]
                    ),
                }
            )
    return pd.DataFrame(rows), np.asarray(symmetry_deltas, dtype=float)


def _load_analysis_metadata_and_activity_gate(
    pair_summary: pd.DataFrame,
    row: pd.Series,
    registered_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    match = pair_summary.loc[
        (pair_summary["animal_id"] == str(row["animal"]))
        & (pair_summary["src_session"] == str(row["src_session"]))
        & (pair_summary["dst_session"] == str(row["dst_session"]))
    ]
    if match.shape[0] != 1:
        raise ValueError(
            f"Expected one Figure 2 pair-summary row for {row['animal']} "
            f"{row['src_session']}->{row['dst_session']}; found {match.shape[0]}."
        )
    pair_row = match.iloc[0]
    data_path = _require_file(Path(str(pair_row["data_path"])))
    with data_path.open("rb") as handle:
        result = pickle.load(handle)
    dropped = result.get("dropped", {})
    required = {
        "n_pairs_initial",
        "n_pairs_dropped_by_user",
        "n_pairs_dropped_by_rate",
        "n_pairs_final",
    }
    missing = required.difference(dropped)
    if missing:
        raise KeyError(f"Missing drop metadata {sorted(missing)} in {data_path}")

    train_ids = np.asarray(result["train_ids"], dtype=int).ravel() + 1
    test_ids = np.asarray(result["test_ids"], dtype=int).ravel() + 1
    if train_ids.size != test_ids.size:
        raise ValueError(f"Saved post-gate pair IDs disagree in {data_path}")
    saved_pairs = list(zip(train_ids.tolist(), test_ids.tolist()))
    saved_pair_set = set(saved_pairs)
    if len(saved_pair_set) != len(saved_pairs):
        raise ValueError(f"Saved post-gate pair IDs are duplicated in {data_path}")

    meta_train = result["meta_train"]
    meta_test = result["meta_test"]
    train_rate = np.asarray(meta_train["rate_per_min"], dtype=float).ravel()
    test_rate = np.asarray(meta_test["rate_per_min"], dtype=float).ravel()
    train_threshold = float(meta_train["threshold"])
    test_threshold = float(meta_test["threshold"])
    if not (
        np.isclose(train_threshold, 0.1)
        and np.isclose(test_threshold, 0.1)
        and str(dropped["policy"]) == "either"
    ):
        raise ValueError(
            f"Unexpected activity-gate definition in {data_path}: "
            f"thresholds=({train_threshold}, {test_threshold}), "
            f"policy={dropped['policy']}"
        )

    source_train_ids = registered_df["session1_cell_id_source_exported"].to_numpy(
        dtype=int
    )
    source_test_ids = registered_df["session2_cell_id_source_exported"].to_numpy(
        dtype=int
    )
    corrected_train_ids = registered_df["session1_cell_id_corrected"].to_numpy(
        dtype=int
    )
    corrected_test_ids = registered_df["session2_cell_id_corrected"].to_numpy(dtype=int)
    for ids, rates, label in (
        (source_train_ids, train_rate, "source train"),
        (source_test_ids, test_rate, "source test"),
        (corrected_train_ids, train_rate, "corrected train"),
        (corrected_test_ids, test_rate, "corrected test"),
    ):
        if np.any(ids <= 0) or np.any(ids > rates.size):
            raise IndexError(f"{label} IDs exceed rate-vector bounds in {data_path}")

    source_train_rate = train_rate[source_train_ids - 1]
    source_test_rate = test_rate[source_test_ids - 1]
    corrected_train_rate = train_rate[corrected_train_ids - 1]
    corrected_test_rate = test_rate[corrected_test_ids - 1]
    source_pass = (source_train_rate >= train_threshold) & (
        source_test_rate >= test_threshold
    )
    corrected_pass = (corrected_train_rate >= train_threshold) & (
        corrected_test_rate >= test_threshold
    )
    source_pairs = list(zip(source_train_ids.tolist(), source_test_ids.tolist()))
    corrected_pairs = list(
        zip(corrected_train_ids.tolist(), corrected_test_ids.tolist())
    )
    if len(set(source_pairs)) != len(source_pairs):
        raise ValueError(
            f"Source-exported registered pairs are duplicated in {data_path}"
        )
    if len(set(corrected_pairs)) != len(corrected_pairs):
        raise ValueError(f"Corrected registered pairs are duplicated in {data_path}")
    source_pass_set = {
        pair for pair, passes in zip(source_pairs, source_pass) if bool(passes)
    }
    corrected_pass_set = {
        pair for pair, passes in zip(corrected_pairs, corrected_pass) if bool(passes)
    }
    n_user_drops = int(dropped["n_pairs_dropped_by_user"])
    if not saved_pair_set.issubset(source_pass_set):
        raise ValueError(
            f"Saved post-gate IDs are not a subset of the source-exported-map gate in "
            f"{data_path}"
        )
    if len(source_pass_set - saved_pair_set) != n_user_drops:
        raise ValueError(
            f"Source-map gate differs from saved IDs by "
            f"{len(source_pass_set - saved_pair_set)} pairs, not the recorded "
            f"{n_user_drops} manual drops, in {data_path}"
        )
    if len(source_pass_set) != int(dropped["n_pairs_initial"]) - int(
        dropped["n_pairs_dropped_by_rate"]
    ):
        raise ValueError(f"Source-map activity-drop count disagrees in {data_path}")
    if len(saved_pair_set) != int(dropped["n_pairs_final"]):
        raise ValueError(f"Saved post-gate ID count disagrees in {data_path}")

    activity_gate = pd.DataFrame(
        {
            "animal": registered_df["animal"].to_numpy(),
            "pair_label": registered_df["pair_label"].to_numpy(),
            "cluster_index": registered_df["cluster_index"].to_numpy(dtype=int),
            "session1_cell_id_source_exported": source_train_ids,
            "session2_cell_id_source_exported": source_test_ids,
            "session1_cell_id_corrected": corrected_train_ids,
            "session2_cell_id_corrected": corrected_test_ids,
            "session1_rate_per_min_source_exported": source_train_rate,
            "session2_rate_per_min_source_exported": source_test_rate,
            "session1_rate_per_min_corrected": corrected_train_rate,
            "session2_rate_per_min_corrected": corrected_test_rate,
            "passes_activity_gate_source_exported": source_pass,
            "passes_activity_gate_corrected": corrected_pass,
            "source_pair_in_saved_post_gate_result": [
                pair in saved_pair_set for pair in source_pairs
            ],
            "corrected_pair_in_saved_post_gate_result": [
                pair in saved_pair_set for pair in corrected_pairs
            ],
        }
    )
    activity_gate["source_coordinate_pair_changed_by_correction"] = (
        source_train_ids != corrected_train_ids
    ) | (source_test_ids != corrected_test_ids)
    activity_gate["stale_manual_exclusion_recovered"] = (
        activity_gate["passes_activity_gate_source_exported"]
        & ~activity_gate["source_pair_in_saved_post_gate_result"]
    )

    metadata = {
        "figure2_data_path": str(data_path),
        "analysis_n_pairs_initial": int(dropped["n_pairs_initial"]),
        "analysis_n_pairs_dropped_by_user_stale": int(
            dropped["n_pairs_dropped_by_user"]
        ),
        "analysis_n_pairs_dropped_by_rate": int(dropped["n_pairs_dropped_by_rate"]),
        "analysis_n_pairs_final_saved_stale": int(dropped["n_pairs_final"]),
        "analysis_n_pairs_rate_pass_source_exported": int(np.sum(source_pass)),
        "corrected_n_pairs_dropped_by_rate": int(
            registered_df.shape[0] - np.sum(corrected_pass)
        ),
        "corrected_n_pairs_rate_pass": int(np.sum(corrected_pass)),
        "n_corrected_gate_pairs_absent_from_saved_result": int(
            len(corrected_pass_set - saved_pair_set)
        ),
        "n_saved_pairs_absent_from_corrected_gate": int(
            len(saved_pair_set - corrected_pass_set)
        ),
        "saved_pair_ids_match_source_gate_after_recorded_manual_drops": True,
    }
    return metadata, activity_gate


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 4.9,
        }
    )


def _save_figure(fig: plt.Figure, output_stem: Path, *, dpi: int) -> list[str]:
    paths: list[str] = []
    png_path: Path | None = None
    for suffix in (".png", ".svg", ".pdf"):
        path = output_stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
            png_path = path
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    if png_path is not None:
        tiff_path = output_stem.with_suffix(".tiff")
        with Image.open(png_path) as image:
            image.save(tiff_path, dpi=(dpi, dpi), compression="tiff_lzw")
        paths.append(str(tiff_path))
    return paths


def _histogram_table(
    values: np.ndarray,
    *,
    edges: np.ndarray,
    group: str,
    value_name: str,
) -> pd.DataFrame:
    counts, resolved_edges = np.histogram(values, bins=edges)
    return pd.DataFrame(
        {
            "group": group,
            "value": value_name,
            "bin_left": resolved_edges[:-1],
            "bin_right": resolved_edges[1:],
            "bin_center": (resolved_edges[:-1] + resolved_edges[1:]) / 2.0,
            "count": counts.astype(int),
        }
    )


def _render_candidate_histogram(
    candidate_df: pd.DataFrame,
    *,
    value_column: str,
    edges: np.ndarray,
    xlabel: str,
    output_stem: Path,
    dpi: int,
) -> tuple[list[str], pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(2.75, 2.35), facecolor="white")
    tables: list[pd.DataFrame] = []
    for group, label, color, zorder in [
        ("other_within_radius", "Other candidates", OTHER_COLOR, 1),
        ("closest_by_centroid", "Centroid-closest candidate", NEAREST_COLOR, 2),
    ]:
        values = pd.to_numeric(
            candidate_df.loc[candidate_df["candidate_class"] == group, value_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if value_column == "spatial_footprint_correlation":
            values = values[values >= 0.0]
        ax.hist(
            values,
            bins=edges,
            color=color,
            histtype="step",
            linewidth=1.25,
            alpha=1.0,
            label=label,
            zorder=zorder,
        )
        tables.append(
            _histogram_table(values, edges=edges, group=group, value_name=value_column)
        )
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Directed candidate count")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.35, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper center")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.97)
    return _save_figure(fig, output_stem, dpi=dpi), pd.concat(tables, ignore_index=True)


def _render_registered_joint(
    registered_df: pd.DataFrame,
    *,
    distance_edges: np.ndarray,
    correlation_edges: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> tuple[list[str], pd.DataFrame]:
    distance = pd.to_numeric(
        registered_df["centroid_distance_um"], errors="coerce"
    ).to_numpy(dtype=float)
    correlation = pd.to_numeric(
        registered_df["spatial_footprint_correlation"], errors="coerce"
    ).to_numpy(dtype=float)
    finite = np.isfinite(distance) & np.isfinite(correlation)
    distance = distance[finite]
    correlation = correlation[finite]
    joint_counts, resolved_distance_edges, resolved_correlation_edges = np.histogram2d(
        distance,
        correlation,
        bins=(distance_edges, correlation_edges),
    )
    if int(joint_counts.sum()) != int(registered_df.shape[0]):
        raise ValueError(
            "Registered-pair joint histogram does not preserve its denominator."
        )

    fig = plt.figure(figsize=(3.25, 2.9), facecolor="white")
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=(4.0, 1.0),
        height_ratios=(1.0, 4.0),
        hspace=0.06,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_main)

    masked_counts = np.ma.masked_less_equal(joint_counts.T, 0)
    vmax = max(1.0, float(np.nanmax(joint_counts)))
    mesh = ax_main.pcolormesh(
        resolved_distance_edges,
        resolved_correlation_edges,
        masked_counts,
        cmap="viridis",
        norm=LogNorm(vmin=1.0, vmax=vmax),
        shading="flat",
        rasterized=True,
    )
    distance_counts = joint_counts.sum(axis=1)
    correlation_counts = joint_counts.sum(axis=0)
    distance_widths = np.diff(resolved_distance_edges)
    correlation_widths = np.diff(resolved_correlation_edges)
    ax_top.bar(
        resolved_distance_edges[:-1],
        distance_counts,
        width=distance_widths,
        align="edge",
        color=REGISTERED_COLOR,
        edgecolor="white",
        linewidth=0.2,
    )
    ax_right.barh(
        resolved_correlation_edges[:-1],
        correlation_counts,
        height=correlation_widths,
        align="edge",
        color=REGISTERED_COLOR,
        edgecolor="white",
        linewidth=0.2,
    )
    ax_main.set_xlim(float(distance_edges[0]), float(distance_edges[-1]))
    ax_main.set_ylim(float(correlation_edges[0]), float(correlation_edges[-1]))
    ax_main.set_xlabel(r"Centroid distance ($\mu$m)")
    ax_main.set_ylabel("Spatial footprint correlation (r)")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_top.set_ylabel("Count", labelpad=2)
    ax_right.set_xlabel("Count", labelpad=2)
    ax_top.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    # A fixed figure-relative inset avoids backend-specific failures when the
    # same source figure is exported to several vector/raster formats.
    color_axis = fig.add_axes([0.61, 0.245, 0.025, 0.22])
    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.ax.set_title("Bin count\n(log scale)", fontsize=6.2, pad=2)
    colorbar.set_ticks([tick for tick in (1, 10, 100, 1000) if tick <= vmax])
    colorbar.ax.tick_params(labelsize=6.2, length=2)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.96)

    distance_left, correlation_left = np.meshgrid(
        resolved_distance_edges[:-1], resolved_correlation_edges[:-1], indexing="ij"
    )
    distance_right, correlation_right = np.meshgrid(
        resolved_distance_edges[1:], resolved_correlation_edges[1:], indexing="ij"
    )
    joint_table = pd.DataFrame(
        {
            "distance_bin_left_um": distance_left.ravel(),
            "distance_bin_right_um": distance_right.ravel(),
            "correlation_bin_left": correlation_left.ravel(),
            "correlation_bin_right": correlation_right.ravel(),
            "count": joint_counts.astype(int).ravel(),
        }
    )
    return _save_figure(fig, output_stem, dpi=dpi), joint_table


def _render_registration_score(
    registered_df: pd.DataFrame,
    *,
    edges: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> tuple[list[str], pd.DataFrame]:
    scores = pd.to_numeric(
        registered_df["cellreg_registration_score"], errors="coerce"
    ).to_numpy(dtype=float)
    scores = scores[np.isfinite(scores)]
    scores = scores[(scores >= float(edges[0])) & (scores <= float(edges[-1]))]
    histogram = _histogram_table(
        scores,
        edges=edges,
        group="registered_pairs",
        value_name="cellreg_registration_score",
    )
    fig, ax = plt.subplots(figsize=(2.75, 2.35), facecolor="white")
    ax.bar(
        histogram["bin_center"],
        histogram["count"],
        width=(edges[1] - edges[0]) * 0.92,
        color=REGISTERED_COLOR,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("CellReg cell score (cell_scores)")
    ax.set_ylabel("Registered-pair count")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.35, zorder=0)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.97)
    return _save_figure(fig, output_stem, dpi=dpi), histogram


def build_registration_quality_bundle(
    *,
    scope_csv: Path,
    pair_summary_csv: Path,
    footprint_root: Path,
    output_dir: Path,
    dpi: int,
) -> dict[str, object]:
    scope = pd.read_csv(_require_file(scope_csv), dtype={"animal": str})
    pair_summary = pd.read_csv(
        _require_file(pair_summary_csv), dtype={"animal_id": str}
    )
    required_scope = {
        "animal",
        "src_session",
        "dst_session",
        "transition",
        "experience",
    }
    missing_scope = required_scope.difference(scope.columns)
    if missing_scope:
        raise ValueError(f"Scope CSV is missing columns: {sorted(missing_scope)}")
    if scope.shape[0] != 36 or scope["animal"].nunique() != 14:
        raise ValueError(
            f"Expected the final 36-pair/14-animal Figure 2 scope; got "
            f"{scope.shape[0]} pairs and {scope['animal'].nunique()} animals."
        )
    expected_scope = pd.read_csv(DEFAULT_SCOPE_CSV, dtype={"animal": str})
    scope_keys = list(
        scope[["animal", "src_session", "dst_session"]].itertuples(
            index=False, name=None
        )
    )
    expected_scope_keys = list(
        expected_scope[["animal", "src_session", "dst_session"]].itertuples(
            index=False, name=None
        )
    )
    if scope_keys != expected_scope_keys:
        raise ValueError(
            "Scope rows do not exactly match the final ordered Figure 2 scaffold."
        )
    if ("1616", "1616day1", "1616bday1") in scope_keys:
        raise ValueError(
            "The short 1616 pair must not appear in the final Figure 2 scope."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidate_frames: list[pd.DataFrame] = []
    all_registered_frames: list[pd.DataFrame] = []
    all_activity_gate_frames: list[pd.DataFrame] = []
    pair_audits: list[dict[str, object]] = []

    for pair_index, row in scope.reset_index(drop=True).iterrows():
        animal = str(row["animal"])
        src_session = str(row["src_session"])
        dst_session = str(row["dst_session"])
        pair_label = f"{src_session}->{dst_session}"
        pair_code = _pair_code(row)
        animal_dir = footprint_root / animal
        cellreg_dir = animal_dir / f"cellreg_{pair_code}"
        modeled_path = cellreg_dir / "modeled_data_struct.mat"
        cellregistered_path, n_cellregistered_candidates = _latest_cellregistered(
            cellreg_dir
        )
        exported_cimap_path = animal_dir / f"cell_to_index_map_{pair_code}.mat"

        candidate_df, candidate_lookup, candidate_audit = _extract_candidate_features(
            modeled_path,
            animal=animal,
            pair_label=pair_label,
            pair_code=pair_code,
            src_session=src_session,
            dst_session=dst_session,
        )
        with h5py.File(cellregistered_path, "r") as mat_file:
            cleaned_cimap = np.asarray(
                mat_file["cell_registered_struct"]["cell_to_index_map"], dtype=float
            ).T.astype(int)
        exported_cimap = _load_exported_cimap(exported_cimap_path)
        (
            corrected_cimap,
            cimap_match_status,
            removed_counts,
        ) = _validate_cellreg_to_exported_cimap(
            cleaned_cimap,
            exported_cimap=exported_cimap,
            animal_dir=animal_dir,
            pair_code=pair_code,
        )
        corrected_cimap_path = ""
        if cimap_match_status == "source_exported_unexpanded_cleaned_coordinates":
            corrected_cimap_dir = output_dir / "corrected_cimaps"
            corrected_cimap_dir.mkdir(parents=True, exist_ok=True)
            corrected_path = (
                corrected_cimap_dir
                / f"{animal}_cell_to_index_map_{pair_code}_zero_reinserted.mat"
            )
            sio.savemat(corrected_path, {"cell_to_index_map": corrected_cimap})
            corrected_cimap_path = str(corrected_path)
        registered_df, symmetry_deltas = _extract_registered_features(
            cellregistered_path,
            corrected_cimap=corrected_cimap,
            source_exported_cimap=exported_cimap,
            candidate_lookup=candidate_lookup,
            microns_per_pixel=float(candidate_audit["microns_per_pixel"]),
            animal=animal,
            pair_label=pair_label,
            pair_code=pair_code,
            src_session=src_session,
            dst_session=dst_session,
        )
        analysis_meta, activity_gate_df = _load_analysis_metadata_and_activity_gate(
            pair_summary,
            row,
            registered_df,
        )
        n_registered = int(registered_df.shape[0])
        if n_registered != int(analysis_meta["analysis_n_pairs_initial"]):
            raise ValueError(
                f"CellReg registered-pair count {n_registered} does not match Figure 2 initial "
                f"count {analysis_meta['analysis_n_pairs_initial']} for {pair_label}."
            )
        if registered_df["cellreg_registration_score"].isna().any():
            raise ValueError(f"Non-finite registration score found for {pair_label}")
        if (registered_df["cellreg_registration_score"] > 1.0 + 1e-10).any():
            raise ValueError(f"Registration score exceeds 1 for {pair_label}")

        all_candidate_frames.append(candidate_df)
        all_registered_frames.append(registered_df)
        all_activity_gate_frames.append(activity_gate_df)
        active_rows = np.all(cleaned_cimap > 0, axis=1)
        n_registered_ids_changed_by_correction = int(
            np.sum(
                np.any(
                    corrected_cimap[active_rows] != exported_cimap[active_rows],
                    axis=1,
                )
            )
        )
        pair_audits.append(
            {
                "pair_index": int(pair_index + 1),
                "animal": animal,
                "src_session": src_session,
                "dst_session": dst_session,
                "pair_label": pair_label,
                "transition": str(row["transition"]),
                "experience": int(row["experience"]),
                "pair_code": pair_code,
                "cellreg_dir": str(cellreg_dir),
                "modeled_data_struct": str(modeled_path),
                "cellregistered_mat": str(cellregistered_path),
                "n_cellregistered_candidates": int(n_cellregistered_candidates),
                "exported_cimap": str(exported_cimap_path),
                "corrected_cimap_audit_copy": corrected_cimap_path,
                "cimap_match_status": cimap_match_status,
                "n_registered_pair_ids_changed_by_coordinate_correction": (
                    n_registered_ids_changed_by_correction
                ),
                "all_zero_footprints_reinserted_session1": int(removed_counts[0]),
                "all_zero_footprints_reinserted_session2": int(removed_counts[1]),
                "n_registered_pairs": n_registered,
                "max_forward_reverse_feature_abs_delta": float(
                    np.max(symmetry_deltas) if symmetry_deltas.size else 0.0
                ),
                **candidate_audit,
                **analysis_meta,
            }
        )
        print(
            f"[{pair_index + 1:02d}/{scope.shape[0]}] {pair_label}: "
            f"registered={n_registered}, candidates={candidate_df.shape[0]}, "
            f"cimap={cimap_match_status}"
        )

    candidate_long = pd.concat(all_candidate_frames, ignore_index=True)
    registered_long = pd.concat(all_registered_frames, ignore_index=True)
    activity_gate_long = pd.concat(all_activity_gate_frames, ignore_index=True)
    pair_audit = pd.DataFrame(pair_audits)
    n_registered = int(registered_long.shape[0])
    n_analysis_initial = int(pair_audit["analysis_n_pairs_initial"].sum())
    n_saved_user_drops = int(pair_audit["analysis_n_pairs_dropped_by_user_stale"].sum())
    n_saved_activity_drops = int(pair_audit["analysis_n_pairs_dropped_by_rate"].sum())
    n_saved_final = int(pair_audit["analysis_n_pairs_final_saved_stale"].sum())
    n_source_map_rate_pass = int(
        pair_audit["analysis_n_pairs_rate_pass_source_exported"].sum()
    )
    n_corrected_activity_drops = int(
        pair_audit["corrected_n_pairs_dropped_by_rate"].sum()
    )
    corrected_analysis_count = int(pair_audit["corrected_n_pairs_rate_pass"].sum())
    if n_registered != n_analysis_initial or n_registered != 21412:
        raise ValueError(
            f"Expected 21,412 exact Figure 2 registered pairs; got CellReg={n_registered}, "
            f"analysis={n_analysis_initial}."
        )
    if (
        n_saved_user_drops != 4
        or n_saved_activity_drops != 4524
        or n_saved_final != 16884
        or n_source_map_rate_pass != 16888
        or n_corrected_activity_drops != 4511
        or corrected_analysis_count != 16901
    ):
        raise ValueError(
            "Figure 2 denominator audit changed unexpectedly: "
            f"saved_final={n_saved_final}, user={n_saved_user_drops}, "
            f"saved_activity={n_saved_activity_drops}, "
            f"source_map_pass={n_source_map_rate_pass}, "
            f"corrected_activity={n_corrected_activity_drops}, "
            f"corrected={corrected_analysis_count}."
        )

    candidate_path = output_dir / "candidate_features_directed_long.csv"
    registered_path = output_dir / "registered_pair_features_long.csv"
    activity_gate_path = output_dir / "activity_gate_pair_id_audit.csv"
    pair_audit_path = output_dir / "pair_source_and_count_audit.csv"
    candidate_long.to_csv(candidate_path, index=False)
    registered_long.to_csv(registered_path, index=False)
    activity_gate_long.to_csv(activity_gate_path, index=False)
    pair_audit.to_csv(pair_audit_path, index=False)

    closest_summary = (
        candidate_long.loc[candidate_long["candidate_class"] == "closest_by_centroid"]
        .groupby(["animal", "pair_label"], as_index=False)
        .agg(
            n_closest_candidates=("candidate_class", "size"),
            median_closest_distance_um=("centroid_distance_um", "median"),
            median_closest_correlation=("spatial_footprint_correlation", "median"),
        )
    )
    other_summary = (
        candidate_long.loc[candidate_long["candidate_class"] == "other_within_radius"]
        .groupby(["animal", "pair_label"], as_index=False)
        .agg(
            n_other_candidates=("candidate_class", "size"),
            median_other_distance_um=("centroid_distance_um", "median"),
            median_other_correlation=("spatial_footprint_correlation", "median"),
        )
    )
    candidate_pair_summary = closest_summary.merge(
        other_summary,
        on=["animal", "pair_label"],
        how="inner",
        validate="one_to_one",
    )
    candidate_pair_summary["closest_has_lower_median_distance"] = (
        candidate_pair_summary["median_closest_distance_um"]
        < candidate_pair_summary["median_other_distance_um"]
    )
    candidate_pair_summary["closest_has_higher_median_correlation"] = (
        candidate_pair_summary["median_closest_correlation"]
        > candidate_pair_summary["median_other_correlation"]
    )
    registered_pair_summary = registered_long.groupby(
        ["animal", "pair_label"], as_index=False
    ).agg(
        n_registered_pairs=("cluster_index", "size"),
        median_registered_distance_um=("centroid_distance_um", "median"),
        median_registered_correlation=("spatial_footprint_correlation", "median"),
        median_cellreg_registration_score=("cellreg_registration_score", "median"),
        fraction_cellreg_registration_score_ge_0p95=(
            "cellreg_registration_score",
            lambda values: float(np.mean(values >= 0.95)),
        ),
        median_pairwise_p_same=("p_same", "median"),
    )
    candidate_pair_summary_path = (
        output_dir / "candidate_separation_summary_by_pair.csv"
    )
    registered_pair_summary_path = output_dir / "registered_feature_summary_by_pair.csv"
    candidate_pair_summary.to_csv(candidate_pair_summary_path, index=False)
    registered_pair_summary.to_csv(registered_pair_summary_path, index=False)

    _configure_matplotlib()
    distance_edges = np.linspace(0.0, 14.0, 41)
    correlation_edges = np.linspace(0.0, 1.0, 41)
    score_edges = np.linspace(0.0, 1.0, 51)
    centroid_paths, centroid_hist = _render_candidate_histogram(
        candidate_long,
        value_column="centroid_distance_um",
        edges=distance_edges,
        xlabel=r"Centroid distance ($\mu$m)",
        output_stem=output_dir / "panel_g1_centroid_distance",
        dpi=dpi,
    )
    correlation_paths, correlation_hist = _render_candidate_histogram(
        candidate_long,
        value_column="spatial_footprint_correlation",
        edges=correlation_edges,
        xlabel="Spatial footprint correlation (r)",
        output_stem=output_dir / "panel_g2_spatial_footprint_correlation",
        dpi=dpi,
    )
    joint_paths, joint_hist = _render_registered_joint(
        registered_long,
        distance_edges=distance_edges,
        correlation_edges=correlation_edges,
        output_stem=output_dir / "panel_h_registered_pair_joint_distribution",
        dpi=dpi,
    )
    score_paths, score_hist = _render_registration_score(
        registered_long,
        edges=score_edges,
        output_stem=output_dir / "panel_i_cellreg_registration_score",
        dpi=dpi,
    )

    centroid_hist_path = output_dir / "panel_g1_centroid_distance_histogram.csv"
    correlation_hist_path = (
        output_dir / "panel_g2_spatial_footprint_correlation_histogram.csv"
    )
    joint_hist_path = output_dir / "panel_h_registered_pair_joint_histogram.csv"
    score_hist_path = output_dir / "panel_i_cellreg_registration_score_histogram.csv"
    centroid_hist.to_csv(centroid_hist_path, index=False)
    correlation_hist.to_csv(correlation_hist_path, index=False)
    joint_hist.to_csv(joint_hist_path, index=False)
    score_hist.to_csv(score_hist_path, index=False)

    score = registered_long["cellreg_registration_score"].to_numpy(dtype=float)
    p_same = registered_long["p_same"].to_numpy(dtype=float)
    canonical_src_ids = np.where(
        candidate_long["query_session_index"].to_numpy(dtype=int) == 1,
        candidate_long["query_cell_id_cleaned"].to_numpy(dtype=int),
        candidate_long["candidate_cell_id_cleaned"].to_numpy(dtype=int),
    )
    canonical_dst_ids = np.where(
        candidate_long["query_session_index"].to_numpy(dtype=int) == 2,
        candidate_long["query_cell_id_cleaned"].to_numpy(dtype=int),
        candidate_long["candidate_cell_id_cleaned"].to_numpy(dtype=int),
    )
    unique_candidate_pairs = pd.DataFrame(
        {
            "pair_label": candidate_long["pair_label"].to_numpy(),
            "src_cell_id_cleaned": canonical_src_ids,
            "dst_cell_id_cleaned": canonical_dst_ids,
        }
    ).drop_duplicates()
    if candidate_long.shape[0] != 2 * unique_candidate_pairs.shape[0]:
        raise ValueError(
            "Expected every physical candidate pair in both query directions."
        )
    summary = {
        "scope": "final_sorted_figure2_36_pairs_no_1616_short",
        "n_session_pairs": int(scope.shape[0]),
        "n_animals": int(scope["animal"].nunique()),
        "n_registered_pairs_before_activity_gate": n_registered,
        "n_directed_candidate_pairs": int(candidate_long.shape[0]),
        "n_unique_cross_session_candidate_pairs": int(unique_candidate_pairs.shape[0]),
        "n_directed_closest_candidates": int(
            candidate_long["is_closest_by_centroid"].sum()
        ),
        "n_directed_other_candidates": int(
            (~candidate_long["is_closest_by_centroid"]).sum()
        ),
        "n_candidate_negative_correlations_excluded_from_panel_g2": int(
            (candidate_long["spatial_footprint_correlation"] < 0).sum()
        ),
        "n_pairs_closest_lower_median_distance": int(
            candidate_pair_summary["closest_has_lower_median_distance"].sum()
        ),
        "n_pairs_closest_higher_median_correlation": int(
            candidate_pair_summary["closest_has_higher_median_correlation"].sum()
        ),
        "median_cellreg_registration_score": float(np.median(score)),
        "mean_cellreg_registration_score": float(np.mean(score)),
        "n_negative_cellreg_registration_scores": int(np.sum(score < 0.0)),
        "n_negative_cellreg_registration_scores_excluded_from_panel_i": int(
            np.sum(score < 0.0)
        ),
        "fraction_cellreg_registration_score_ge_0p95": float(np.mean(score >= 0.95)),
        "equal_pair_mean_fraction_cellreg_registration_score_ge_0p95": float(
            registered_pair_summary[
                "fraction_cellreg_registration_score_ge_0p95"
            ].mean()
        ),
        "pair_median_cellreg_registration_score_range": [
            float(registered_pair_summary["median_cellreg_registration_score"].min()),
            float(registered_pair_summary["median_cellreg_registration_score"].max()),
        ],
        "pair_fraction_cellreg_registration_score_ge_0p95_range": [
            float(
                registered_pair_summary[
                    "fraction_cellreg_registration_score_ge_0p95"
                ].min()
            ),
            float(
                registered_pair_summary[
                    "fraction_cellreg_registration_score_ge_0p95"
                ].max()
            ),
        ],
        "directed_candidate_count_range_per_session_pair": [
            int(
                (
                    candidate_pair_summary["n_closest_candidates"]
                    + candidate_pair_summary["n_other_candidates"]
                ).min()
            ),
            int(
                (
                    candidate_pair_summary["n_closest_candidates"]
                    + candidate_pair_summary["n_other_candidates"]
                ).max()
            ),
        ],
        "registered_pair_count_range_per_session_pair": [
            int(registered_pair_summary["n_registered_pairs"].min()),
            int(registered_pair_summary["n_registered_pairs"].max()),
        ],
        "median_pairwise_p_same": float(np.median(p_same)),
        "mean_pairwise_p_same": float(np.mean(p_same)),
        "n_source_exported_cimaps_in_unexpanded_cleaned_coordinates": int(
            (
                pair_audit["cimap_match_status"]
                == "source_exported_unexpanded_cleaned_coordinates"
            ).sum()
        ),
        "n_registered_pair_ids_changed_by_coordinate_correction": int(
            pair_audit["n_registered_pair_ids_changed_by_coordinate_correction"].sum()
        ),
        "saved_figure2_metric_observations": n_saved_final,
        "saved_stale_manual_cell_id_drops_not_used_here": n_saved_user_drops,
        "manual_only_restored_count_using_source_exported_maps": n_source_map_rate_pass,
        "activity_gate_drops_using_source_exported_maps": n_saved_activity_drops,
        "activity_gate_drops_using_corrected_cimaps": n_corrected_activity_drops,
        "corrected_figure2_metric_observations_after_activity_gate": corrected_analysis_count,
        "figure2_saved_artifact_status": (
            "the saved 16,884-observation Figure 2 artifacts use an unexpanded 842 0-to-3 "
            "source map and require rerun; restoring only four stale manual exclusions "
            "gives 16,888 but does not fix that coordinate error"
        ),
        "candidate_definition": (
            "directed query-candidate observations within the run-specific 14 um radius; "
            "closest and other classes are defined by centroid distance for both g panels"
        ),
        "pooling_note": (
            "panels show unnormalized pooled technical observations; session pairs with more "
            "candidate or registered cells contribute more counts"
        ),
        "registration_score_definition": (
            "cell_registered_struct.cell_scores for clusters active in both sessions; "
            "this internal competition score is not a probability and can be negative"
        ),
        "p_same_definition": "maximum finite value in p_same_registered_pairs for each active pair",
        "interpretation_limit": (
            "candidate features and cell scores are CellReg inputs/internal diagnostics, "
            "not independent ground-truth registration accuracy"
        ),
    }
    summary_path = output_dir / "registration_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "scope_csv": str(scope_csv),
        "pair_summary_csv": str(pair_summary_csv),
        "footprint_root": str(footprint_root),
        "output_dir": str(output_dir),
        "source_tables": {
            "candidate_features": str(candidate_path),
            "registered_pair_features": str(registered_path),
            "activity_gate_pair_id_audit": str(activity_gate_path),
            "pair_audit": str(pair_audit_path),
            "candidate_pair_summary": str(candidate_pair_summary_path),
            "registered_pair_summary": str(registered_pair_summary_path),
            "summary": str(summary_path),
        },
        "histogram_tables": {
            "centroid_distance": str(centroid_hist_path),
            "spatial_footprint_correlation": str(correlation_hist_path),
            "registered_pair_joint": str(joint_hist_path),
            "cellreg_registration_score": str(score_hist_path),
        },
        "panels": {
            "g1_centroid_distance": centroid_paths,
            "g2_spatial_footprint_correlation": correlation_paths,
            "h_registered_pair_joint": joint_paths,
            "i_cellreg_registration_score": score_paths,
        },
        "important_metric_distinction": (
            "The existing P_same histogram is not the same metric as the reference panel-i "
            "CellReg registration score. This bundle plots cell_scores and retains P_same "
            "only as an auditable source column."
        ),
        "panel_i_display": (
            "cell_scores are displayed from 0 to 1 in 0.02-wide bins; three negative "
            "scores are retained in registered_pair_features_long.csv and omitted rather "
            "than clipped into the zero bin"
        ),
        "important_cimap_correction": (
            "The source-exported 842 0-to-3 CIMap remained in cleaned coordinates after "
            "12 all-zero session-1 footprints were removed. The audit copy reinserts those "
            "indices; the QC feature distributions are unchanged, but the corrected Figure 2 "
            "post-rate-gate denominator is 16,901 rather than 16,888."
        ),
    }
    manifest_path = output_dir / "registration_quality_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = _parse_args()
    result = build_registration_quality_bundle(
        scope_csv=args.scope_csv,
        pair_summary_csv=args.pair_summary_csv,
        footprint_root=args.footprint_root,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
