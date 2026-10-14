"""Reproduce the figure from included derived observations and display arrays."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import json
import numpy as np
import pandas as pd

DATA=Path(__file__).resolve().parents[1]/"data"

def _load(name):
    path=Path(__file__).with_name(name+".py")
    spec=importlib.util.spec_from_file_location(Path(__file__).parents[1].name+"_"+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

inference=_load("inference")
drawing=_load("drawing")

def route_display():
    with np.load(DATA/"route_heatmaps.npz") as arrays:
        heatmaps={group:(arrays[short+"_day_i"],arrays[short+"_day_i90"]) for group,short in (("home_to_reward","h2r"),("reward_to_home","r2h"))}
    return inference.route_summary(heatmaps,pd.read_csv(DATA/"route_pair_profiles.csv"))


def statistics(output_dir):
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
    results={
        "whole_maze":inference.combine_whole_maze_shuffle_values([pd.read_csv(DATA/"whole_maze_pairs.csv")]),
        "reward_control":inference.combine_reward_control_values([pd.read_csv(DATA/"reward_control_pairs.csv")]),
        "decoder_behavior":inference.calculate_reward_decoding_association(pd.read_csv(DATA/"decoder_behavior_pairs.csv")),
        "route":route_display().statistics,
    }
    for name,values in results.items():values.to_csv(destination/(name+"_statistics.csv"),index=False)
    return results


def schematic():
    metadata=json.loads((DATA/"spatial_decoder_example.json").read_text(encoding="utf8"))
    with np.load(DATA/"spatial_decoder_example.npz") as arrays:
        return SimpleNamespace(
            source_session=metadata["src_session"],target_session=metadata["dst_session"],
            source_ids=np.asarray(metadata["selected_train_ids"]),target_ids=np.asarray(metadata["selected_test_ids"]),
            place_maps=tuple(arrays["place_maps"]),activity=arrays["activity"],posterior=arrays["posterior"],
            highlighted_row=metadata["vline_idx"],decoded_xy=tuple(metadata["posterior_decoded_xy"]),observed_xy=tuple(metadata["posterior_true_xy"]),
        )


def render(output_dir):
    destination=Path(output_dir).resolve();destination.mkdir(parents=True,exist_ok=True)
    drawing.REPOSITORY_ROOT=destination
    values=statistics(destination)
    band=inference.prepare_reward_decoding_vs_first_bout_display(values["decoder_behavior"])
    return drawing.assemble_spatial_decoder_results(schematic(),values["whole_maze"],pd.read_csv(DATA/"example_error_map.csv"),values["reward_control"],route_display(),values["decoder_behavior"],band,run_mode="full",output_dir=destination,random_seed=20260823)
