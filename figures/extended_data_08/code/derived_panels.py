"""ED8: saved decoder, PV and route observations with declared test families."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
_spec = importlib.util.spec_from_file_location('ed08_current_science', Path(__file__).with_name('current_science.py'))
science = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = science
_spec.loader.exec_module(science)

def panel(letter):
    return pd.read_csv(DATA/f'panel_{letter}_values.csv',low_memory=False)

def observations(letter):
    """Keep animal observations separate from saved reported test rows."""
    values = panel(letter)
    if "record_type" in values.columns:
        return values.loc[values.record_type.eq("animal")].copy()
    return values

def calculate():
    balanced = panel('b'); balanced = science.summarize_balanced_decoder_pairs(balanced.loc[balanced.row_type.eq('pair')])
    recurrence = panel('c'); recurrence = science.summarize_pv_recurrence_pairs(recurrence.loc[recurrence.row_type.eq('pair')])
    family = pd.concat([observations(letter).assign(group=lambda x: prefix+':'+x.group) for letter,prefix in [('e','reliability'),('f','equal_frame'),('i','node')]],ignore_index=True)
    route_tests = science.summarize_reward_contrasts(family)
    outcome = observations('g').pivot(index='animal',columns='group',values='advantage').dropna()
    outcome_test = pd.DataFrame([{'group':'rewarded minus timeout','p_exact':science.exact_sign_flip((outcome.rewarded-outcome.timeout).to_numpy())}])
    cross = observations('j').query("measure == 'cross direction across gap'").pivot(index='animal',columns='bin',values='pv')
    cross_test = pd.DataFrame([{'group':'cross direction across gap','p_exact':science.exact_sign_flip((cross[21]-cross.loc[:,range(21)].mean(axis=1)).to_numpy())}])
    return balanced, recurrence, route_tests, outcome_test, cross_test

def statistics(output_dir):
    destination = Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    balanced, recurrence, routes, outcome, cross = calculate()
    day = panel('d')
    results = {'balanced_decoder_bh4':balanced.loc[balanced.row_type.eq('statistics')],
               'pv_recurrence_bh2':recurrence.loc[recurrence.row_type.eq('statistics')],
               'route_bh8':routes, 'rewarded_timeout':outcome, 'cross_direction':cross,
               'day_first':science.summarize_reward_contrasts(day.loc[day.day.eq('first')]),
               'day_second':science.summarize_reward_contrasts(day.loc[day.day.eq('second')])}
    for name,frame in results.items(): frame.to_csv(destination/f'ed8_{name}_statistics.csv',index=False)
    return results

def render(output_dir):
    destination = Path(output_dir).resolve(); destination.mkdir(parents=True,exist_ok=True)
    balanced, recurrence, routes, outcome, cross = calculate()
    day = panel('d')
    figures = {
      'a':science.plot_decoder_error_map(panel('a')),
      'b':science.plot_balanced_decoder(balanced),
      'c':science.plot_pv_recurrence(recurrence),
      'd':science.plot_day_pair_route_advantages(day.loc[day.day.eq('first')],day.loc[day.day.eq('second')]),
      'g':science.plot_reward_delivery_route_control(observations('g'),outcome),
      'h':science.draw_route_schematic()[0],
      'j':science.plot_cross_direction_route_profiles(observations('j'),cross),
    }
    for letter,prefix,function in [('e','reliability',science.plot_reliability_adjusted_routes),('f','equal_frame',science.plot_equal_frame_route_control),('i','node',science.plot_node_route_controls)]:
        tests = routes.loc[routes.group.str.startswith(prefix+':')].copy()
        tests['group'] = tests.group.str.removeprefix(prefix+':')
        figures[letter] = function(observations(letter),tests)
    sources = {}
    for letter,figure in figures.items():
        paths = science.save_figure(figure,destination/f'extended_data_figure_8_panel_{letter}',formats=('svg','pdf','png'),raster_dpi=600,transparent=True,preserve_canvas=True)
        sources[letter] = {path.suffix:path for path in paths}
        science.plt.close(figure)
    return science._assemble_page_from_sources(sources=sources,output_dir=destination,output_stem='Extended_Data_Figure_8')
