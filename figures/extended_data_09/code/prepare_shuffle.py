"""Freeze the Level population-vector histogram from already saved ID shuffles.

Only reads derived maps/nulls: no raw data, new permutations, fitting or plotting.
Each null draw averages supported levels within pair, then the 36 pairs equally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "output/analysis/mm_proposed_panels_2026-09-24/level_pv"
DEFAULT_SOURCE = ROOT / "frozen" if (ROOT / "frozen/MANIFEST.json").exists() else ORIGINAL
DEFAULT_OUTPUT = ROOT / "output/figures/ed9_three_row_level_pv_2026-09-25/source_data/population_vector_shuffle"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path: Path) -> dict:
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def row_unit(raw: np.ndarray) -> np.ndarray:
    centered = raw - raw.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norm, out=np.full_like(centered, np.nan), where=norm > 1e-12)


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen sources: {output}")
    upstream = json.loads((source / "MANIFEST.json").read_text())
    inventory = {row["path"]: row for row in upstream["output_hashes"]}
    filenames = ["pair_summary.csv", "pair_by_depth.csv", "pair_depth_cross_similarity_matrices.csv",
                 "mean_cross_depth_matrix.csv", "matrix_pair_support.csv", "registered_neuron_depth_maps.csv",
                 "source_support_and_registration_audit.csv", "global_clustered_inference.csv",
                 "frozen_source_selection.csv", "inherited_session_alignment_audit.csv",
                 "independent_numeric_validation.json", "raw_C_reconstruction_audit.csv",
                 "raw_validation_depth_exposure.csv"]
    for name in filenames:
        assert sha(source / name) == inventory[name]["sha256"], name
    producer = source / "analyze_mm_level_population_vectors.py"
    if not producer.exists():
        producer = ROOT / "scripts/analyze_mm_level_population_vectors.py"
    expected_code = next(row["sha256"] for row in upstream["input_hashes"]
                         if row["path"].endswith("analyze_mm_level_population_vectors.py"))
    assert sha(producer) == expected_code
    pair_source = pd.read_csv(source / "pair_summary.csv")
    levels = pd.read_csv(source / "pair_by_depth.csv").set_index(["pair_key", "depth"])
    neurons = pd.read_csv(source / "registered_neuron_depth_maps.csv")
    matrix_table = pd.read_csv(source / "pair_depth_cross_similarity_matrices.csv")
    mean_matrix = pd.read_csv(source / "mean_cross_depth_matrix.csv").set_index("train_depth").to_numpy(float)
    assert len(pair_source) == 36 and pair_source.animal_id.nunique() == 14
    assert not pair_source.pair_key.duplicated().any() and pair_source.n_neurons.sum() == 16901
    reference = source / "figure5_matrix_reference.csv"
    if not reference.exists():
        reference = ROOT / "output/figures/figure5_ed9_panel_exchange_2026-09-25/source_data/level_pv/level_pv_matrix.csv"
    assert sha(reference) == sha(source / "mean_cross_depth_matrix.csv")
    pair_rows, mask_rows, paths, pair_nulls, observed, matrices = [], [], [], [], [], []
    max_matrix_error = max_scipy_error = max_null_level_difference = 0.0
    for i, pair in enumerate(pair_source.itertuples(index=False)):
        path = source / "intermediates" / f"{i+1:02d}_{pair.src_session}_to_{pair.dst_session}.npz"
        paths.append(path)
        with np.load(path) as cache:
            a, b = cache["train_map"], cache["test_map"]
            mask = cache["common_mask"].astype(bool)
            assert a.shape == b.shape == (7, pair.n_neurons)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            assert np.array_equal(mask, cache["train_mask"] & cache["test_mask"])
            assert mask.sum() == pair.n_depths
            registered = neurons[neurons.pair_key.eq(pair.pair_key)]
            assert np.array_equal(registered.train_id.to_numpy(), cache["train_ids"])
            assert np.array_equal(registered.test_id.to_numpy(), cache["test_ids"])
            np.testing.assert_allclose(a, registered[[f"train_depth_{d}_C_per_s" for d in range(7)]].to_numpy().T, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(b, registered[[f"test_depth_{d}_C_per_s" for d in range(7)]].to_numpy().T, rtol=1e-12, atol=1e-12)
            u, v = row_unit(a), row_unit(b)
            np.testing.assert_allclose(u, cache["train_centered_unit_vectors"], rtol=0, atol=1e-15, equal_nan=True)
            np.testing.assert_allclose(v, cache["test_centered_unit_vectors"], rtol=0, atol=1e-15, equal_nan=True)
            common = mask[:, None] & mask[None, :]
            reconstructed = u @ v.T
            reconstructed[~common] = np.nan
            saved_matrix = cache["matrix"]
            np.testing.assert_allclose(reconstructed, saved_matrix, rtol=0, atol=1e-15, equal_nan=True)
            max_matrix_error = max(max_matrix_error, float(np.nanmax(abs(reconstructed-saved_matrix))))
            table_matrix = matrix_table[matrix_table.pair_key.eq(pair.pair_key)].pivot(index="train_depth", columns="test_depth", values="pearson_r").to_numpy()
            np.testing.assert_allclose(table_matrix, saved_matrix, rtol=0, atol=1e-15, equal_nan=True)
            for d1 in np.flatnonzero(mask):
                for d2 in np.flatnonzero(mask):
                    max_scipy_error = max(max_scipy_error, abs(float(stats.pearsonr(a[d1], b[d2]).statistic)-saved_matrix[d1,d2]))
            matrices.append(saved_matrix.copy())
            value = float(np.diag(saved_matrix)[mask].mean())
            assert abs(value-pair.same_depth) < 1e-15
            observed.append(value)
            raw_null = cache["neuron_id_shuffle_by_depth"]
            assert raw_null.dtype == np.float32 and raw_null.shape == (10000, 7)
            null = raw_null.astype(np.float64)
            assert np.isfinite(null[:,mask]).all() and np.isnan(null[:,~mask]).all()
            means = null[:,mask].mean(axis=1)
            pair_nulls.append(means)
            bound = float(np.max(abs(np.spacing(raw_null[:,mask])))) / 2
            difference = abs(float(means.mean())-pair.neuron_id_shuffle)
            assert difference <= bound + 1e-15
            pair_rows.append({**pair._asdict(), "observed_mean_correlation": value,
                              "cached_null_mean": float(means.mean()), "cached_null_sd": float(means.std(ddof=1)),
                              "source_null_mean_abs_difference": difference, "float32_half_ulp_bound": bound,
                              "pair_weight": 1/36, "supported_level_weight": 1/int(mask.sum()),
                              "repeat_count": len(means), "cache_sha256": sha(path)})
            for d in range(7):
                old = levels.loc[(pair.pair_key,d)]
                assert bool(old.supported)==bool(mask[d])
                assert bool(old.train_mask)==bool(cache["train_mask"][d]) and bool(old.test_mask)==bool(cache["test_mask"][d])
                null_mean = float(null[:,d].mean()) if mask[d] else None
                delta = abs(null_mean-float(old.neuron_id_shuffle)) if mask[d] else None
                if mask[d]:
                    assert delta <= bound + 1e-15
                    max_null_level_difference = max(max_null_level_difference, delta)
                mask_rows.append(dict(pair_key=pair.pair_key, animal_id=pair.animal_id, level=d,
                    train_supported=bool(cache["train_mask"][d]), test_supported=bool(cache["test_mask"][d]),
                    common_supported=bool(mask[d]), observed_correlation=float(saved_matrix[d,d]) if mask[d] else None,
                    cached_null_mean=null_mean, original_null_mean=float(old.neuron_id_shuffle) if mask[d] else None,
                    null_mean_abs_difference=delta, n_registered_neurons=pair.n_neurons,
                    within_pair_weight=1/int(mask.sum()) if mask[d] else 0))
    matrices = np.asarray(matrices)
    np.testing.assert_allclose(np.nanmean(matrices,axis=0), mean_matrix, rtol=0, atol=1e-15)
    support = np.isfinite(matrices).sum(axis=0)
    assert set(np.unique(support)) == {32,36}
    assert np.array_equal(np.diag(support), [36,36,36,36,36,36,32])
    pair_nulls = np.asarray(pair_nulls,dtype=np.float64)
    null = pair_nulls.mean(axis=0)
    obs = float(np.mean(observed))
    greater = int(np.count_nonzero(null >= obs))
    inherited = pd.read_csv(source / "global_clustered_inference.csv")
    inherited_row = inherited.loc[inherited.contrast.eq("identity_preservation")].iloc[0].to_dict()
    original_delta = pair_source.identity_preservation.to_numpy()
    animal = pair_source.animal_id.to_numpy(); groups = np.unique(animal)
    scores = np.array([(original_delta[animal==g]-original_delta.mean()).sum() for g in groups])
    se = np.sqrt(len(groups)/(len(groups)-1)*np.sum(scores**2)/len(original_delta)**2)
    p = float(2*stats.t.sf(abs(original_delta.mean()/se),len(groups)-1))
    assert np.isclose(p,inherited_row["p_two_sided"],rtol=1e-12,atol=0)
    output.mkdir(parents=True)
    frozen = output / "frozen"
    frozen.mkdir()
    copied = {}
    def copy(src: Path, relative: str):
        dst = frozen / relative; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dst); assert sha(dst)==sha(src)
        copied[f"frozen/{relative}"] = {**info(dst), "source": str(src)}
    for name in [*filenames, "MANIFEST.json"]:
        copy(source/name,name)
    for path in paths:
        copy(path,path.relative_to(source).as_posix())
    copy(producer,"analyze_mm_level_population_vectors.py")
    copy(reference,"figure5_matrix_reference.csv")
    code_dir=output/"source_code";code_dir.mkdir();shutil.copy2(Path(__file__),code_dir/"prepare_shuffle.py")
    pd.DataFrame({"repeat":np.arange(len(null)),"mean_correlation":null}).to_csv(output/"null_draws.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(output/"pair_summary.csv",index=False)
    pd.DataFrame(mask_rows).to_csv(output/"level_masks.csv",index=False)
    pd.DataFrame({"pair_key":np.repeat(pair_source.pair_key.to_numpy(),len(null)),
                  "animal_id":np.repeat(pair_source.animal_id.to_numpy(),len(null)),
                  "repeat":np.tile(np.arange(len(null)),len(pair_source)),
                  "mean_correlation":pair_nulls.ravel()}).to_csv(output/"pair_null_draws.csv.gz",index=False,compression="gzip")
    density,edges=np.histogram(null,bins=80,density=True)
    pd.DataFrame({"left":edges[:-1],"right":edges[1:],"density":density}).to_csv(output/"histogram_80bins.csv",index=False)
    summary = {
        "status":"PASS", "observed_mean":obs, "null_mean":float(null.mean()),
        "null_sd":float(null.std(ddof=1)), "null_min":float(null.min()), "null_max":float(null.max()),
        "null_quantile025":float(np.quantile(null,.025)), "null_quantile975":float(np.quantile(null,.975)),
        "shuffle_count":len(null), "n_pairs":36, "n_animals":14, "registered_neuron_pair_observations":16901,
        "levels":[0,1,2,3,4,5,6], "pair_support_per_level":np.diag(support).tolist(),
        "matrix_cell_pair_support":support.tolist(), "pair_weights":"equal, 1/36",
        "level_weights":"Within each pair, equal among its fixed 6 or 7 commonly supported levels.",
        "conditional_p_right":(greater+1)/(len(null)+1), "conditional_null_exceedances":greater,
        "conditional_p_definition":"(1 + number of cached global null means >= observed equal-pair mean)/(10000+1); descriptive fixed-sample neuron-ID randomization, not a biological-replicate p-value.",
        "inherited_animal_clustered_inference":inherited_row,
        "inherited_family":"Two original global contrasts: same-level minus neuron-ID control and same-level minus different-level correlation. Two-sided animal-clustered CR1 Student t(13), BH2 retained unchanged.",
        "shuffle_rule":"Independent uniform target-neuron permutations within each pair; one permutation is shared across all levels in a repeat. Repeat r combines the r-th independent draw from every pair. Masks, source maps and registered cohort remain fixed.",
        "representation":"Ordinary seven-level maze code; C-derived activity maps before within-neuron display normalization. Center and unit-normalize across neurons within each level solely to compute Pearson correlation.",
        "within_neuron_level_normalization":"none", "signal":"C", "activity_gate":"S activity at least 0.1/min in both sessions; exact inherited registered IDs, no manual exclusions.",
        "occupancy_support":"Inherited positive occupancy in both sessions, not a new 5-second gate. Unsupported levels remain excluded, never set to zero.",
        "reward_scope":"31 rewarded-source pairs and 5 initially unrewarded-source pairs; all target sessions rewarded at canonical node94, matching original analysis.",
        "source_null_storage":"float32; loaded and aggregated as float64; no new permutations generated.",
        "source_null_pair_mean_max_abs_difference":float(max(row["source_null_mean_abs_difference"] for row in pair_rows)),
        "source_null_level_mean_max_abs_difference":max_null_level_difference,
        "simple_diagonal_mean_of_cellwise_matrix_not_used":float(np.diag(mean_matrix).mean()),
        "weighting_note":"Observed line averages per-pair diagonal means before averaging pairs; directly averaging the displayed matrix diagonal gives a different weighting because level6 has32 rather than36 pairs.",
        "source_matrix_reconstruction_max_abs_difference":max_matrix_error,
        "independent_scipy_pearson_max_abs_difference":float(max_scipy_error),
        "new_raw_analysis_runs":0, "new_permutations":0,
    }
    write_json(output/"summary.json",summary)
    (output/"README.md").write_text(
        "# Level population-vector neuron-identity shuffle histogram\n\n"
        "The observed line is the equal-pair mean of same-level population-vector correlations across 36 recording pairs from 14 mice. "
        "Each of 10,000 null draws averages the cached neuron-identity shuffle at the same supported levels within each pair, then averages pairs equally. "
        "The masks, C-derived maps, registered neuron cohort and within-pair permutations are inherited unchanged. No within-neuron level normalization or new permutations are introduced.\n\n"
        "`null_draws.csv` is the histogram input. `pair_null_draws.csv.gz`, `pair_summary.csv` and `level_masks.csv` expose every aggregation. "
        "`summary.json` separates the conditional randomization p-value from the original animal-clustered BH2 result. "
        "The histogram spread is permutation variability, not a biological confidence interval. Repeated pairs are retained; their shuffled identities are independent within pairs and do not impose a joint longitudinal identity permutation across shared sessions.\n\n"
        "The original manifest hashes the summary/map tables and producer, but does not hash its intermediate NPZ files. "
        "Their maps, IDs, masks and matrix entries are verified against those hashed tables; cached null means match within their float32 precision. "
        "This bundle now freezes and hashes every complete NPZ input.\n\n"
        "Rebuild from frozen derived inputs into a new directory: `python source_code/prepare_shuffle.py --source frozen --output rebuilt`. "
        "No raw recording, biological rerun or rendering is required.\n",encoding="utf-8")
    outputs={f.relative_to(output).as_posix():info(f) for f in sorted(output.rglob("*")) if f.is_file()}
    manifest={"schema":"ed9_level_population_vector_shuffle/v1", "status":"PASS", "outputs":outputs,
              "source_inventory":copied, "source_analysis_manifest_sha256":sha(source/"MANIFEST.json"),
              "upstream_intermediate_hash_limitation":"Original manifest does not include intermediate NPZ hashes; numerical maps/IDs/masks and matrices verified against hashed exported tables, null means verified within float32 quantization; all NPZ inputs hashed here.",
              "inherited_tests_unchanged":True,"figure5_matrix_sha256":sha(reference),
              "new_raw_analysis_runs":0,"new_permutations":0}
    write_json(output/"manifest.json",manifest)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    summary=prepare(args.source,args.output)
    print(json.dumps({k:summary[k] for k in ("status","observed_mean","null_mean","null_sd","conditional_p_right","source_null_pair_mean_max_abs_difference")},indent=2))


if __name__=="__main__":
    main()
