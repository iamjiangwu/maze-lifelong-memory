import csv
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .trajectory import LoadTraj


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def read_csv_to_dict(path):
    rows = read_csv(path)
    if not rows:
        return []
    keys = rows[0]
    return [dict(zip(keys, r)) for r in rows[1:]]


def read_mat_file(neural_dir, exp_date, datasets, verbose=False):
    """
    Read neural matrices from supported MATLAB/HDF5 naming patterns.

    Legacy datasets are commonly stored as ``final_day{exp_date}.mat``.
    Some older exports may use ``{exp_date}.mat`` or ``day{exp_date}.mat``.
    """
    nd = Path(neural_dir)
    candidates = [
        nd / f'final_day{exp_date}.mat',
        nd / f'{exp_date}.mat',
        nd / f'day{exp_date}.mat',
        nd / f'final_{exp_date}.mat',
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        if verbose:
            print(f'No neural mat file found under {neural_dir} for exp_date={exp_date}')
        return {}

    out = {}
    with h5py.File(p, 'r') as f:
        for d in datasets:
            if d not in f:
                continue
            obj = f[d]
            if isinstance(obj, h5py.Group):
                out[d] = load_sparse_matrix(obj)
            else:
                out[d] = np.array(obj)
            if verbose:
                print(f'loaded {d} from {p.name}')
    return out


def load_sparse_matrix(group):
    """
    Reconstruct a CSC sparse matrix from MATLAB/HDF5 sparse group fields.
    """
    data = group['data'][:]
    indices = group['ir'][:]
    indptr = group['jc'][:]
    return sp.csc_matrix((data, indices, indptr)).toarray()


def _extract_info(file_name):
    pattern = r'^([0-9]+[a-z]?)(?:-?(\d*))?(?:[RL]+)?(?:day)?(\d+)?$'
    match = re.match(pattern, file_name)
    if not match:
        return None
    animal_id = match.group(1)
    experiment_type = int(match.group(2)) if match.group(2) else 0
    experiment_day = int(match.group(3)) if match.group(3) else 1
    return animal_id, experiment_type, experiment_day


def load_behav_neural(nickname, behav_wd=None, neural_wd=None, video=None, ma=None):
    """
    Load behavior + neural arrays for one session.

    Behavior tables are loaded from selected_data CSVs.
    Legacy JW fallback behavior processing has been removed.
    """
    if behav_wd is None:
        behav_wd = os.environ.get('BEHAV_WD')
    if neural_wd is None:
        neural_wd = os.environ.get('NEURAL_WD')
    if behav_wd is None or neural_wd is None:
        raise ValueError(
            'behav_wd and neural_wd are required. '
            'Set BEHAV_WD / NEURAL_WD env vars or pass them explicitly.'
        )

    parsed = _extract_info(nickname)
    if parsed is None:
        raise ValueError(f'Could not parse nickname: {nickname}')
    mouse_id, exp_type, exp_date = parsed

    try:
        tr = LoadTraj(nickname + '-tf')
    except Exception:
        tr = None

    selected_path = Path(behav_wd) / 'selected_data' / f'{nickname}_selected_data.csv'
    nose_interp_path = Path(behav_wd) / 'selected_data' / f'{nickname}_nose_interp_data.csv'
    if not selected_path.exists() or not nose_interp_path.exists():
        raise FileNotFoundError(
            'Missing preprocessed behavior CSV(s) for '
            f'{nickname}. Expected:\n'
            f'- {selected_path}\n'
            f'- {nose_interp_path}\n'
            'Generate and place these files under selected_data before running load_behav_neural.'
        )
    selected_df = pd.read_csv(selected_path)
    nose_interp_df = pd.read_csv(nose_interp_path)

    c_raw = c = s = cn = None
    neural_dir = os.path.join(neural_wd, mouse_id)
    if os.path.exists(neural_dir):
        if exp_type != 0:
            mouse_id = mouse_id + '-' + str(exp_type)
            neural_dir = os.path.join(neural_wd, mouse_id)
        datasets = ['C_raw', 'C', 'S', 'Cn']
        data = read_mat_file(neural_dir, exp_date, datasets)
        if 'C_raw' in data:
            c_raw = data['C_raw'].T
        if 'C' in data:
            c = data['C'].T
        if 'S' in data:
            s = data['S'].T
        if 'Cn' in data:
            cn = data['Cn'].T
    else:
        print(f'Neural data for {mouse_id} not found in {neural_wd}')

    return tr, selected_df, nose_interp_df, c_raw, c, s, cn
