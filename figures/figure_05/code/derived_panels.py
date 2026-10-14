"""Draw maze level representations and recalculate their behavioral association."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import pandas as pd
from release_rendering.common import save_panel_output

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'data'


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f'level_summary_{name}', HERE/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA/name, float_precision='round_trip')


def statistics(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Fit the association with animal clustered errors from the 36 pair rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = _read('stability_and_exploration.csv')
    inference = _module('inference').summarize_level_stability_vs_efficiency(values)
    result = pd.DataFrame([inference])
    result.to_csv(output_dir/'association_statistics.csv', index=False)
    return {'association_statistics':result}


def render(output_dir: Path):
    """Load derived level maps and correlations; draw the five accepted panels."""
    output_dir = Path(output_dir).resolve()
    drawing, inference = _module('drawing'), _module('inference')
    drawing.REPOSITORY_ROOT = output_dir
    stats = statistics(output_dir/'statistics')['association_statistics'].iloc[0].to_dict()
    values = _read('stability_and_exploration.csv')
    display = inference.prepare_level_stability_vs_efficiency_display(values, stats)
    schematic, definition = drawing.draw_tree_depth_schematic()
    maps = _read('ordered_level_maps.csv')
    correlations = _read('level_population_vectors.csv')
    matrix = correlations.groupby(['train_depth','test_depth']).pearson_r.mean().unstack('test_depth')
    histogram = _read('level_correlation_histogram.csv')
    panels = [
        ('a',schematic,definition),
        ('b',drawing.plot_level_heatmaps(maps),maps),
        ('c',drawing.plot_level_population_vectors(matrix),correlations),
        ('d',drawing.plot_level_correlation_distribution(histogram),histogram),
        ('e',drawing.plot_level_stability_vs_efficiency(values,stats,display),values),
    ]
    results = [save_panel_output(artifact_key='main_figure_5',panel=label,kind='quantitative',
        run_mode='full',output_dir=output_dir,random_seed=20260602,figure=figure,values=table,
        record_style=drawing.record_style_settings()) for label,figure,table in panels]
    return drawing.assemble_tree_depth_results(run_mode='full',output_dir=output_dir,
        depth_definition=results[0],ordered_level_maps=results[1],level_population_vectors=results[2],
        neuron_identity_distribution=results[3],stability_efficiency_association=results[4])
