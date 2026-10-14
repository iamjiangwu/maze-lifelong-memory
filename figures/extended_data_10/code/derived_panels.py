"""ED10: current coordinated readouts, clustered inference and final drawing."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed10_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def load_panel(letter):
    return pd.read_csv(DATA / f'panel_{letter}_values.csv', keep_default_na=True)

def readouts():
    points = load_panel('a').query("record_type == 'pair_value'")
    index = ['pair_key', 'animal_id']
    neural = points.pivot_table(index=index, columns='neural_readout', values='neural_value', aggfunc='first')
    behavior = points.pivot_table(index=index, columns='behavior_measure', values='behavior_value', aggfunc='first')
    return neural.join(behavior).reset_index()

def policies():
    points = load_panel('c').query("record_type == 'point'")
    index = ['pair_key', 'animal_id', 'source_session', 'target_session', 'level_map_correlation']
    return points.pivot_table(index=index, columns='policy_key', values='policy_value', aggfunc='first').reset_index()

def calculate():
    pairs = readouts()
    matrix = science.calculate_neural_behavior_correlations(pairs)
    commonality = pd.concat([science.calculate_commonality_components(pairs), science.calculate_neural_behavior_profile_tests(pairs)], ignore_index=True)
    policy = load_panel('c')
    inference = science.calculate_policy_inference(policies(), 'full').assign(record_type='inference')
    inference['policy_index'] = inference.policy_key.map({key:index for index,key in enumerate(science.POLICY_ORDER)})
    policy = pd.concat([policy.loc[~policy.record_type.eq('inference')], inference], ignore_index=True)
    return matrix, commonality, policy

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    matrix, commonality, policy = calculate()
    results = {'neural_behavior': matrix.query("record_type == 'correlation'"),
               'commonality_and_profile': commonality.loc[~commonality.record_type.eq('pair_value')],
               'policy': policy.query("record_type == 'inference'")}
    for name, frame in results.items(): frame.to_csv(destination / f'ed10_{name}_statistics.csv', index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True, exist_ok=True)
    science.REPOSITORY_ROOT = destination
    values = calculate()
    return science.assemble_neural_behavior_controls(*values, run_mode='full', output_dir=destination, random_seed=0)
