from __future__ import annotations

import os
from pathlib import Path

FPS = 25.0
DT = 0.04
N_BINS_X = 15
N_BINS_Y = 15
MAZE_ID = 6
REWARD_NODE = 94
LEGACY_REWARD_NODE = 116

_DEFAULT_BEHAV_WD = r"d:\combineddata"
_DEFAULT_NEURAL_WD = r"d:\neuralData"
_DEFAULT_FOOTPRINTS_ROOT = r"F:\preprocess\footprints"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_behav_wd() -> str:
    return os.environ.get("BEHAV_WD", _DEFAULT_BEHAV_WD)


def get_neural_wd() -> str:
    return os.environ.get("NEURAL_WD", _DEFAULT_NEURAL_WD)


def get_parity_metafile() -> str:
    return os.environ.get("PARITY_METAFILE", os.path.join(get_behav_wd(), "decoding_metafile.csv"))


def get_footprints_root() -> str:
    return os.environ.get("FOOTPRINTS_WD", _DEFAULT_FOOTPRINTS_ROOT)


def get_cimap_path(
    *,
    animal: str = "451",
    filename: str = "cell_to_index_map_36.mat",
) -> str:
    default = str(Path(get_footprints_root()) / str(animal) / str(filename))
    return os.environ.get("CIMAP_PATH", default)


def get_decoding_results_root() -> str:
    return os.environ.get(
        "DECODING_RESULTS_ROOT",
        str(Path(get_behav_wd()) / "decoding_results_102025"),
    )
