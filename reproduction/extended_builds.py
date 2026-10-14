"""Input adapters for the behavioral and registration displays."""
from __future__ import annotations
import contextlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator
VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]

@contextlib.contextmanager
def frozen_stage(record: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix='edsrc_') as raw:
        stage = Path(raw)
        shutil.copytree(record / 'source_code', stage / 'scripts')
        runtime_source = VOR_ROOT / 'runtime'
        runtime_root = stage / 'shared_runtime'
        shutil.copytree(runtime_source, runtime_root)
        inserted = [stage, stage / 'scripts', runtime_root]
        for path in reversed(inserted):
            sys.path.insert(0, str(path))
        try:
            yield stage
        finally:
            for path in inserted:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(path))

def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import frozen renderer: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render_ed2(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    module = load_module(stage / 'scripts/render_ed2.py', 'ed2_current_renderer')
    return module.render(record, output, 600)

def render_ed3(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / 'scripts/render_ed03_reward_path_fluency_assembly.py', 'ed3_source_rebuild')
    plot = record / 'plot_ready'
    m.REPO_ROOT = REPO_ROOT
    m.load_fluency_module = lambda : load_module(stage / 'scripts/fluency_panels.py', 'ed3_source_rebuild_fluency')
    m.FLUENCY_ROWS = plot / 'fluency_plot_rows.csv'
    m.FLUENCY_STATS = plot / 'fluency_statistics.csv'
    m.DEPTH_TABLE = plot / 'depth_small_multiples.csv'
    m.ROUTE_DEFINITION = plot / 'route_definition.csv'
    result = m.run(output)
    if result['validation_status'] != 'pass':
        raise RuntimeError('ED3 source renderer did not pass')
    return {'status': 'PASS', 'validation': result['validation']}

def render_ed4(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / 'scripts/render_ed04_registration_quality_assembly.py', 'ed4_source_rebuild')
    m.REPO_ROOT = REPO_ROOT
    m.SOURCE = record / 'source_from_raw'
    m.ALIGNMENT = m.SOURCE
    m.LOCAL_CORRELATION_IMAGE = record / 'embedded_sources/ed05_four_session_418_local_correlation_paper.png'
    m.COREGISTRATION_IMAGE = record / 'embedded_sources/ed05_four_session_418_coregistration_paper.png'
    original_canvas_save = m.Canvas.save

    def canvas_save_compatibility(canvas: Any, stem: Path, *, vector_dpi: float | None=None) -> list[Path]:
        if vector_dpi is not None:
            canvas.fig.set_dpi(vector_dpi)
        return original_canvas_save(canvas, stem)
    m.Canvas.save = canvas_save_compatibility
    try:
        result = m.build(output)
    finally:
        m.Canvas.save = original_canvas_save
    if str(result['validation_status']).upper() != 'PASS':
        raise RuntimeError('ED4 source renderer did not pass')
    return {'status': 'PASS', 'validation': result['validation']}
