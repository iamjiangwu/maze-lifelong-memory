"""ED6: saved display curves and paired cycle-level statistical inputs."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed06_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def records(letters, kind):
    frames = [pd.read_csv(DATA/f'panel_{letter}_values.csv', low_memory=False) for letter in letters]
    frame = pd.concat(frames, ignore_index=True)
    return frame.loc[frame.record_type.eq(kind)].dropna(axis=1, how='all').drop_duplicates().reset_index(drop=True)

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    fixed = science.calculate_fixed_reference_slope_statistics(records('ceg','pair_slope'))
    pivot = science.calculate_sliding_pivot_statistics(records('dfh','pair_curve_summary'))
    results = {'fixed_reference_bh12':fixed, 'sliding_pivot_bh3':pivot}
    for name, frame in results.items(): frame.to_csv(destination/f'ed6_{name}_statistics.csv', index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True, exist_ok=True)
    science.REPOSITORY_ROOT = destination
    fixed = records('ceg','group_curve_and_support')
    pivot = records('dfh','group_curve_and_support')
    limits = records('dfh','display_range')
    return science.assemble_within_cycle_stability(fixed, pivot, limits, run_mode='full', output_dir=destination)
