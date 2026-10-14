"""ED7: age associations adjusted for categorical experience or transition."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed07_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def records(letters, kind):
    frame = pd.concat([pd.read_csv(DATA/f'panel_{letter}_values.csv') for letter in letters], ignore_index=True)
    return frame.loc[frame.record_type.eq(kind)].dropna(axis=1, how='all').reset_index(drop=True)

def calculate():
    within = records('abc','point')
    long = records('def','point')
    companion = pd.read_csv(DATA/'level_age_family_points.csv')
    within_models = science.fit_age_model_family(within, metrics=tuple(science.DISPLAY_METRICS), age_column='age_day_i_days', experience_column='experience_index', family_name='p_bh3')
    long_models = science.fit_age_model_family(pd.concat([long,companion],ignore_index=True), metrics=(*science.DISPLAY_METRICS,'level_map_correlation'), age_column='destination_age_days', experience_column='experience', family_name='p_bh4')
    return within, long, within_models, long_models

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _, _, within, long = calculate()
    results = {'within_cycle_age_bh3':within, 'long_gap_age_bh4':long}
    for name, frame in results.items(): frame.to_csv(destination/f'ed7_{name}_statistics.csv', index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True, exist_ok=True)
    science.REPOSITORY_ROOT = destination
    within, long, within_models, long_models = calculate()
    lines = []
    for points, models, is_long in ((within,within_models,False),(long,long_models,True)):
        lines.append(pd.concat([science.prepare_age_display_lines(points, models.set_index('metric').loc[metric], metric=metric, long_gap=is_long) for metric in science.DISPLAY_METRICS],ignore_index=True))
    return science.assemble_age_association_page(within, lines[0], long, lines[1], run_mode='full', output_dir=destination, random_seed=0)
