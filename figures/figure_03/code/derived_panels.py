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

def statistics(output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    results={
        "first_window_bh4": inference.experience_statistics(pd.read_csv(DATA/"first_window_observations.csv"),"cycle_index"),
        "successive_cycle_bh4": inference.experience_statistics(pd.read_csv(DATA/"successive_cycle_observations.csv"),"experience"),
        "convergence_bh12": inference.convergence_statistics(pd.read_csv(DATA/"convergence_cycle_slopes.csv")),
    }
    for name,values in results.items():values.to_csv(output_dir/(name+"_statistics.csv"),index=False)
    return results


def registered_example():
    sessions=pd.read_csv(DATA/"example_sessions.csv")
    result=[]
    with np.load(DATA/"example_coordinates.npz") as arrays:
        for prefix,row in zip(("A","A2","B","B2"),sessions.itertuples(index=False),strict=True):
            result.append(SimpleNamespace(
                session_id=row.session_id,display_day=row.display_day,
                trajectory_x=arrays[prefix+"_traj_x"],trajectory_y=arrays[prefix+"_traj_y"],
                body_x=arrays[prefix+"_px"],body_y=arrays[prefix+"_py"],
                nose_x=arrays[prefix+"_ex"],nose_y=arrays[prefix+"_ey"],
            ))
    return result


def render(output_dir):
    destination=Path(output_dir).resolve();destination.mkdir(parents=True,exist_ok=True)
    drawing.REPOSITORY_ROOT=destination
    tests=statistics(destination)
    within=SimpleNamespace(curves=pd.read_csv(DATA/"convergence_curves.csv"),first_points=pd.read_csv(DATA/"first_window_observations.csv"),first_stats=tests["first_window_bh4"],slope_stats=tests["convergence_bh12"])
    successive=SimpleNamespace(points=pd.read_csv(DATA/"successive_cycle_observations.csv"),statistics=tests["successive_cycle_bh4"])
    return drawing.present_return_session_reconstruction(registered_activity=registered_example(),within_cycle_stability=within,successive_cycle_stability=successive,run_mode="full",output_dir=destination,random_seed=20260823)
