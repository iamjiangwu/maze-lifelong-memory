"""Apply the Figure 1 i/j axes geometry before drawing the included panels."""
from pathlib import Path
import importlib.util
import sys

AXES_MM = {"i": [(81.3, 156, 39.7, 31)], "j": [(135.3, 156, 39.7, 31)]}
SLOTS_MM = {"i": (70, 155, 51, 39), "j": (124, 155, 52, 39)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(output):
    output = Path(output).resolve()
    source = output / "frozen_source"
    if not source.is_dir():
        raise FileNotFoundError("Run Figure 1 through reproduce.py to supply its inputs")
    entrypoint = load("figure1_source_entrypoint", source / "source_code/rebuild_figure1.py")
    original_load = entrypoint.load

    def configure_geometry(name, path):
        module = original_load(name, path)
        if name == "figure1_panel_layout":
            module.AXES_MM.update(AXES_MM)
            module.SLOTS_MM.update(SLOTS_MM)
        return module

    entrypoint.load = configure_geometry
    return entrypoint.rebuild(source, output)
