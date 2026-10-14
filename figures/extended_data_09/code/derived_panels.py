"""ED9: tree-depth controls from saved observations and full test families."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed09_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def panel(letter, kind=None):
    frame = pd.read_csv(DATA/f'panel_{letter}_values.csv',low_memory=False)
    if kind is not None: frame = frame.loc[frame.record_type.eq(kind)]
    return frame.dropna(axis=1,how='all').copy()

def experience_family(filename, x_column):
    frame = pd.read_csv(DATA/filename)
    result = pd.DataFrame([{'metric':metric, **science.clustered_slope_test(group,x_column=x_column,value_column='value')} for metric,group in frame.groupby('metric',sort=False)])
    result['p_bh4'] = science.benjamini_hochberg(result.p_two_sided.to_numpy(float))
    return result

def calculate():
    recurrence = science.summarize_recurrence_points(pd.read_csv(DATA/'recurrence_family_points.csv'))
    null = panel('c')
    observed = float(null.observed_mean_correlation.iloc[0])
    right_tail = (1 + int((null.shuffle_mean_correlation >= observed).sum())) / (len(null)+1)
    return {
      'mean_level_null':pd.DataFrame([{'observed_mean_correlation':observed,'n_shuffles':len(null),'right_tail_p':right_tail}]),
      'preferred_level_bh2':science.summarize_preferred_level_inference(panel('e','pair_summary')),
      'partition_bh2':science.summarize_eight_bin_inference(panel('h','pair_wide')),
      'spatial_null':science.spatial_null_statistic(SimpleNamespace(pairs=panel('j','pair'))),
      'decoder':pd.DataFrame([science.summarize_decoder_accuracy(panel('d'))]),
      'recurrence_bh4':recurrence,
      'first30_bh4':experience_family('first30_family_points.csv','cycle_index'),
      'longgap_bh4':experience_family('longgap_family_points.csv','experience'),
    }

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    results = calculate()
    for name,frame in results.items(): frame.to_csv(destination/f'ed9_{name}_statistics.csv',index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True,exist_ok=True)
    tests = calculate()
    preference = SimpleNamespace(transition=panel('e','transition_cell'),distance=panel('f','shift_distance'))
    spatial = SimpleNamespace(pairs=panel('j','pair'),draws=panel('j','cohort_shuffle_draw'))
    figure = science.render_tree_depth_control_summary(
        preferred_level_values=preference, preferred_level_statistics=tests['preferred_level_bh2'],
        spatial_reassignment_values=spatial, spatial_reassignment_display=panel('j','display_summary'),
        spatial_reassignment_statistic=tests['spatial_null'], partition_correlations=panel('h','pair_wide'),
        partition_display_summary=panel('h','display_summary'), partition_statistics=tests['partition_bh2'],
        joint_neuron_values=panel('i'), gap_values=panel('k'), recurrence_values=panel('l','point'),
        recurrence_display_trend=panel('l','display_trend'), recurrence_statistic=tests['recurrence_bh4'].query("metric == 'level'"),
        within_cycle_display=panel('m','display_curve'), experience_values=panel('o','pair_point'),
        experience_display_fit=panel('o','display_fit'), experience_statistics=tests['longgap_bh4'],
        population_histogram=panel('a'), population_null=panel('b'), mean_level_null=panel('c'),
        decoder_values=panel('d'), decoder_inference=tests['decoder'].iloc[0].to_dict(),
        first30_values=panel('n','pair_point'), first30_display_fit=panel('n','display_fit'),
        first30_statistics=tests['first30_bh4'])
    science.save_tree_depth_control_page(figure,destination)
    science.plt.close(figure)
    return sorted(destination.glob('Extended_Data_Figure_9.*'))
