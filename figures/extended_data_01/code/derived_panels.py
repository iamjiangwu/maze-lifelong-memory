"""Load derived ED1 tables, recompute inference, and draw the page."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'

def _load(name):
    spec = importlib.util.spec_from_file_location('ed1_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

inference = _load('inference')
drawing = _load('drawing')

def table(name):
    return pd.read_csv(DATA / (name + '.csv'))

def endpoint_observations():
    return table('acquisition_points').pivot(index=['session', 'animal_id', 'endpoint_key', 'endpoint_label', 'group_key'], columns='axis_key', values='value').reset_index()

def statistics(output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = {
        'acquisition_bh18': inference._endpoint_statistics(endpoint_observations()),
        'long_bout_relation': inference.compute_long_bout_relation(table('long_bout_observations'))[2],
    }
    for name, values in results.items():
        values.to_csv(destination / (name + '_statistics.csv'), index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve()
    tests = statistics(destination)
    points, line, _, _ = inference.compute_long_bout_relation(table('long_bout_observations'))
    figure = drawing.compose_behavior_learning_overview(
        reward_time_curve=table('reward_time_curve'),
        long_bout_time_curve=table('long_bout_time_curve'),
        reward_transition_curve=table('reward_step_curve'),
        long_bout_transition_curve=table('long_bout_step_curve'),
        acquisition_endpoint_points=table('acquisition_points'),
        acquisition_endpoint_statistics=tests['acquisition_bh18'],
        long_bout_relation_points=points,
        long_bout_relation_line=line,
        session_speed_summary=table('speed_summary'),
        coverage_visit_sequence=table('coverage_visits'),
        coverage_cycles=table('coverage_cycles'),
        time_efficiency_summary=table('time_efficiency'),
        transition_efficiency_summary=table('step_efficiency'),
    )
    outputs = drawing.save_behavior_learning_overview(figure, destination)
    return SimpleNamespace(outputs=outputs)
