"""ED5 source maps and animal-clustered six-comparison inference."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed05_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def load_panel(letter):
    return pd.read_csv(DATA / f'panel_{letter}_values.csv', low_memory=False)

def gallery():
    values = pd.read_csv(DATA/'gallery_examples.csv')
    arrays = np.load(DATA/'gallery_maps.npz',allow_pickle=False)
    pairs, maps, correlations = [], [], []
    for _, group in values.groupby('visual_rank', sort=True):
        pairs.append(SimpleNamespace(highlighted_reference_example=bool(group.highlighted_reference_example.iloc[0])))
        maps.append(tuple(arrays[group.loc[group.side.eq(side),'array_key'].iloc[0]] for side in ('source','target')))
        correlations.append(float(group.place_map_correlation.iloc[0]))
    return SimpleNamespace(pairs=pairs, normalized_maps=maps, correlations=correlations)

def calculate():
    pairs = []
    for letter, metric in zip('bcd', ('pv_correlation','place_map_correlation','ensemble_rate_correlation')):
        pairs.append(load_panel(letter).query("record_type == 'session_pair'").assign(metric=metric))
    pairs = pd.concat(pairs, ignore_index=True)
    context = load_panel('e').query("record_type == 'session_pair'").copy()
    return pairs, science.summarize_day_combination_values(pairs), science.calculate_day_combination_statistics(pairs), context, science.calculate_context_statistics(context)

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _, summary, tests, _, context_tests = calculate()
    results = {'day_combination_summary':summary, 'day_combination_bh6':tests, 'maze_context':context_tests}
    for name, frame in results.items(): frame.to_csv(destination/f'ed5_{name}_statistics.csv', index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True, exist_ok=True)
    science.REPOSITORY_ROOT = destination
    return science.assemble_long_gap_stability_controls(gallery(), *calculate(), run_mode='full', output_dir=destination)
