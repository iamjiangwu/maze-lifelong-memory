from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster
def zscore(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError('Cannot standardize nonfinite values')
    sd = float(np.std(values, ddof=0))
    if not math.isfinite(sd) or sd <= 1e-12:
        raise RuntimeError('Cannot standardize a constant vector')
    return (values - float(np.mean(values))) / sd

def design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    pieces = [pd.DataFrame({'intercept': np.ones(len(frame), dtype=float)})]
    for column in columns:
        pieces.append(pd.DataFrame({column: np.asarray(frame[column], dtype=float)}))
    design = pd.concat(pieces, axis=1)
    if np.linalg.matrix_rank(design.to_numpy(float)) != design.shape[1]:
        raise RuntimeError(f'Rank-deficient design: {design.columns.tolist()}')
    return design

def residualize(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)
    (beta, *_) = np.linalg.lstsq(nuisance, values, rcond=None)
    return values - nuisance @ beta

def sign_matrix(n_groups: int) -> np.ndarray:
    values = np.arange(2 ** n_groups, dtype=np.uint32)[:, None]
    bits = (values >> np.arange(n_groups, dtype=np.uint32) & 1).astype(float)
    return 2.0 * bits - 1.0

def synchronized_exact(names: list[str], score_vectors: list[np.ndarray], family: str) -> pd.DataFrame:
    if not score_vectors:
        return pd.DataFrame()
    score = np.vstack(score_vectors)
    standardized = score / np.sqrt(np.sum(score * score, axis=1, keepdims=True))
    signs = sign_matrix(score.shape[1])
    permuted = signs @ standardized.T
    observed = standardized.sum(axis=1)
    rows = []
    for (index, name) in enumerate(names):
        rows.append({'family': family, 'claim': name, 'score_t': float(observed[index]), 'exact_p_two_sided': float(np.mean(np.abs(permuted[:, index]) >= abs(observed[index]) - 1e-15)), 'n_patterns': int(len(signs))})
    return pd.DataFrame(rows)

def r2_of(frame: pd.DataFrame, outcome: str, predictors: list[str]) -> float:
    return float(sm.OLS(np.asarray(frame[outcome], dtype=float), design_matrix(frame, predictors).to_numpy(float)).fit().rsquared)

def commonality(frame: pd.DataFrame, outcome: str, a: str, b: str, nuisance: list[str], label: str) -> dict[str, Any]:
    baseline = r2_of(frame, outcome, nuisance)
    r2_a = r2_of(frame, outcome, [*nuisance, a])
    r2_b = r2_of(frame, outcome, [*nuisance, b])
    full = r2_of(frame, outcome, [*nuisance, a, b])
    return {'analysis': label, 'outcome': outcome, 'predictor_a': a, 'predictor_b': b, 'nuisance': '+'.join(nuisance), 'r2_baseline': baseline, 'r2_a': r2_a, 'r2_b': r2_b, 'r2_full': full, 'full_increment': full - baseline, 'unique_a': full - r2_b, 'shared': r2_a - baseline + (r2_b - baseline) - (full - baseline), 'unique_b': full - r2_a}

def _stacked_reward_level_model(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cells = [('early_reward', 'earlier_first_long', 'reward_location'), ('early_level', 'earlier_first_long', 'structure'), ('explore_reward', 'exploration_efficiency', 'reward_location'), ('explore_level', 'exploration_efficiency', 'structure')]
    work = frame.copy()
    variables = sorted({value for (_, outcome, predictor) in cells for value in (outcome, predictor)})
    for variable in variables:
        work[variable] = zscore(work[variable])
    block_designs = []
    block_names = []
    outcomes = []
    groups = []
    for (cell, outcome, predictor) in cells:
        design = design_matrix(work, [predictor])
        block_designs.append(design.to_numpy(float))
        block_names.append([f'{cell}__{name}' for name in design.columns])
        outcomes.append(np.asarray(work[outcome], dtype=float))
        groups.extend(work['animal_id'].astype(str).tolist())
    block_width = block_designs[0].shape[1]
    stacked = np.zeros((len(cells) * len(work), len(cells) * block_width))
    for (index, design) in enumerate(block_designs):
        start = index * len(work)
        column = index * block_width
        stacked[start:start + len(work), column:column + block_width] = design
    y = np.concatenate(outcomes)
    model = sm.OLS(y, stacked).fit()
    group_array = np.asarray(groups, dtype=str)
    covariance = cov_cluster(model, group_array, use_correction=True)
    names = [name for group in block_names for name in group]
    slope_indices = {cell: names.index(f'{cell}__{predictor}') for (cell, _, predictor) in cells}
    contrasts = {'first_long_reward_minus_level': {'early_reward': 1.0, 'early_level': -1.0}, 'exploration_reward_minus_level': {'explore_reward': 1.0, 'explore_level': -1.0}, 'reward_level_profile_crossover': {'early_reward': 1.0, 'early_level': -1.0, 'explore_reward': -1.0, 'explore_level': 1.0}}
    rows = []
    score_vectors: dict[str, np.ndarray] = {}
    animal_order = sorted(work['animal_id'].astype(str).unique())
    base_animals = work['animal_id'].astype(str).to_numpy()

    def stacked_score(y_values: np.ndarray, target_values: np.ndarray, nuisance_values: np.ndarray, animal_values: np.ndarray) -> np.ndarray:
        y_residual = residualize(y_values, nuisance_values)
        x_residual = residualize(target_values, nuisance_values)
        products = y_residual * x_residual
        return np.asarray([float(products[animal_values == animal].sum()) for animal in animal_order], dtype=float)
    direct_scores: dict[str, np.ndarray] = {}
    for (prefix, outcome) in (('first_long', 'earlier_first_long'), ('exploration', 'exploration_efficiency')):
        y_values = np.concatenate([work[outcome], work[outcome]]).astype(float)
        x_values = np.concatenate([work['reward_location'], work['structure']]).astype(float)
        code_b = np.concatenate([np.zeros(len(work)), np.ones(len(work))])
        animals = np.concatenate([base_animals, base_animals])
        direct_scores[f'{prefix}_reward_minus_level'] = stacked_score(y_values, -x_values * code_b, np.column_stack([np.ones(len(y_values)), x_values, code_b]), animals)
    y_values = np.concatenate([work['earlier_first_long'], work['earlier_first_long'], work['exploration_efficiency'], work['exploration_efficiency']]).astype(float)
    x_values = np.concatenate([work['reward_location'], work['structure'], work['reward_location'], work['structure']]).astype(float)
    code_b = np.tile(np.concatenate([np.zeros(len(work)), np.ones(len(work))]), 2)
    behavior_explore = np.concatenate([np.zeros(2 * len(work)), np.ones(2 * len(work))])
    x_code = x_values * code_b
    x_behavior = x_values * behavior_explore
    code_behavior = code_b * behavior_explore
    x_code_behavior = x_values * code_b * behavior_explore
    crossover_scores = stacked_score(y_values, x_code_behavior, np.column_stack([np.ones(len(y_values)), x_values, code_b, behavior_explore, x_code, x_behavior, code_behavior]), np.tile(base_animals, 4))
    score_vectors = {'first_long_reward_minus_level': direct_scores['first_long_reward_minus_level'], 'exploration_reward_minus_level': direct_scores['exploration_reward_minus_level'], 'reward_level_profile_crossover': crossover_scores}
    for (claim, terms) in contrasts.items():
        contrast = np.zeros(len(names), dtype=float)
        for (cell, coefficient) in terms.items():
            contrast[slope_indices[cell]] = coefficient
        estimate = float(contrast @ model.params)
        se = float(np.sqrt(contrast @ covariance @ contrast))
        t_value = estimate / se
        df = len(animal_order) - 1
        rows.append({'claim': claim, 'estimate': estimate, 'cr1_se': se, 'cr1_t': t_value, 'cr1_df': df, 'cr1_p': float(2.0 * stats.t.sf(abs(t_value), df)), 'ci95_low': estimate - float(stats.t.ppf(0.975, df)) * se, 'ci95_high': estimate + float(stats.t.ppf(0.975, df)) * se, 'n_pairs': int(len(work)), 'n_animals': int(len(animal_order))})
    return (pd.DataFrame(rows), score_vectors)

def reward_level_profile_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the two pooled bars and three captioned profile contrasts.

    Each contrast enumerates all animal-level score sign flips. Benjamini--
    Hochberg correction uses exactly the three captioned pooled comparisons;
    no stage, speed, weighting or leave-one-animal-out sensitivity is run.
    """
    commonality_rows = []
    for outcome in ('earlier_first_long', 'exploration_efficiency'):
        row = commonality(frame, outcome, 'reward_location', 'structure', [], f'reward_vs_level__{outcome}__pooled')
        row['specification'] = 'pooled'
        row['role'] = 'reward_level_commonality'
        commonality_rows.append(row)
    profile, scores = _stacked_reward_level_model(frame)
    exact = synchronized_exact(list(scores), [scores[claim] for claim in scores], 'reward_level_profile__pooled__BH3')
    profile = profile.merge(exact, on='claim', validate='one_to_one')
    profile['bh3_adjusted_p'] = multipletests(profile['exact_p_two_sided'], method='fdr_bh')[1]
    profile['specification'] = 'pooled'
    profile['weighting'] = 'pair_weighted'
    profile['role'] = 'captioned_pooled_profile_test'
    return pd.DataFrame(commonality_rows), profile
