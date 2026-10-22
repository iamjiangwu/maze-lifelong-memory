"""Load the plotted observations, recalculate inference, and draw Figure 2."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f'representation_summary_{name}', HERE/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA/name, float_precision='round_trip')


def _mean_shuffle(metric: str) -> tuple[pd.DataFrame, dict]:
    """Compare the observed pair mean with the saved shuffled pair means."""
    rows = _read(f'{metric}_mean_shuffle.csv')
    observed = float(rows.loc[rows.record_type.eq('observed_pair'), 'value'].mean())
    null = rows.loc[rows.record_type.eq('shuffle_repeat'), 'value'].to_numpy(float)
    null = null[np.isfinite(null)]
    probability = float((1 + np.count_nonzero(null >= observed))/(len(null)+1))
    combined = pd.concat([rows, pd.DataFrame([{'record_type':'observed_mean','value':observed}])], ignore_index=True)
    return combined, {'metric':metric,'observed_mean':observed,'n_shuffles':len(null),'p_right':probability}


def statistics(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Recalculate recurrence, mean-shuffle and days-apart shuffle tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = _module('inference')
    recurrence = inference.summarize_recurrence_points(_read('recurrence_points.csv'))
    shuffle = pd.DataFrame([_mean_shuffle(metric)[1] for metric in ('pv','place','rate')])
    gaps = pd.concat([_read(f'{metric}_gap.csv').assign(metric=metric)
                      for metric in ('pv', 'place', 'rate')], ignore_index=True)
    gap_tests = inference.gap_shuffle_statistics(gaps)
    results = {'recurrence_statistics':recurrence, 'mean_shuffle_statistics':shuffle,
               'gap_shuffle_statistics':gap_tests}
    for name, frame in results.items():
        frame.to_csv(output_dir/f'{name}.csv', index=False)
    return results


def _examples() -> SimpleNamespace:
    """Unpack only the trajectories and activity direction markers displayed."""
    identities = _read('matched_neurons.csv')
    examples = []
    with np.load(DATA/'matched_neuron_trajectories.npz') as archive:
        for example in sorted(identities.example.unique()):
            sides = []
            for side in ('source','target'):
                prefix = f'example_{int(example)}_{side}'
                trajectory = archive[prefix+'_trajectory']
                markers = archive[prefix+'_markers']
                sides.append(SimpleNamespace(trajectory_x=trajectory[:,0], trajectory_y=trajectory[:,1],
                    body_x=markers[:,0],body_y=markers[:,1],nose_x=markers[:,2],nose_y=markers[:,3]))
            row = identities.loc[identities.example.eq(example)].iloc[0]
            examples.append((*sides, row.place_map_pearson_r_15x15, int(row.place_map_effective_support_count_15x15)))
    return SimpleNamespace(examples=tuple(examples))


def render(output_dir: Path):
    """Build the page without loading neural sessions or regenerating shuffles."""
    output_dir = Path(output_dir).resolve()
    drawing, inference = _module('drawing'), _module('inference')
    results = statistics(output_dir/'statistics')
    points = _read('recurrence_points.csv')
    context = []
    for name in ('422_corr.npz','422-3_corr.npz'):
        with np.load(DATA/name) as archive:
            context.append(SimpleNamespace(values=archive['image'], colormap=archive['colormap']))
    metrics = ('pv','place','rate')
    metric_points = {metric:points.loc[points.metric.eq(metric)].copy() for metric in metrics}
    metric_stats = {metric:results['recurrence_statistics'].set_index('metric').loc[metric] for metric in metrics}
    return drawing.render_representation_stability_summary(
        context_images=SimpleNamespace(sources=tuple(context)), matched_neurons=_examples(),
        schematic_values={m:_read(f'{m}_schematic.csv') for m in metrics},
        distributions={m:_read(f'{m}_distribution.csv') for m in metrics},
        mean_shuffles={m:_mean_shuffle(m)[0] for m in metrics},
        gap_curves={m:_read(f'{m}_gap.csv') for m in metrics},
        recurrence_points=metric_points, recurrence_statistics=metric_stats,
        recurrence_displays={m:inference.prepare_recurrence_display(metric_points[m],metric_stats[m],m) for m in metrics},
        output_dir=output_dir, raster_dpi=600)
