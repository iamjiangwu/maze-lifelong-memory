"""Build ED2 once from its first-30-minute derived tables and saved simulations."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import ed2_panels as panels
from ed2_export import export


PANEL_FRAMES = {
    'a': {'left': 2, 'top': 20.178340, 'width': 148, 'height': 202.119941},
    'b': {'left': 152, 'top': 80, 'width': 282, 'height': 139.333824},
    'c': {'left': 436, 'top': 80, 'width': 282, 'height': 139.333824},
    'd': {'left': 2, 'top': 224.298282, 'width': 139.476654, 'height': 236.761689},
    'e': {'left': 143.476654, 'top': 224.298282, 'width': 143.698408, 'height': 236.761689},
    'f': {'left': 289.175063, 'top': 224.298282, 'width': 428.824937, 'height': 236.761689},
    'g': {'left': 2, 'top': 497.697222, 'width': 171.175063, 'height': 202.124437},
    'h': {'left': 175.175063, 'top': 482.202860, 'width': 112, 'height': 217.618800},
    'i': {'left': 289.175063, 'top': 463.059971, 'width': 428.824937, 'height': 236.761689},
}
LABEL_FRAMES = {
    'a': {'left': 4, 'top': 27.5, 'width': 18, 'height': 16},
    'b': {'left': 154, 'top': 27.5, 'width': 18, 'height': 16},
    'c': {'left': 438, 'top': 27.5, 'width': 18, 'height': 16},
    'd': {'left': 4, 'top': 229.5, 'width': 18, 'height': 16},
    'e': {'left': 145.476654, 'top': 229.5, 'width': 18, 'height': 16},
    'f': {'left': 291.175063, 'top': 229.5, 'width': 18, 'height': 16},
    'g': {'left': 4, 'top': 463.0, 'width': 18, 'height': 16},
    'h': {'left': 177.175063, 'top': 473.5, 'width': 18, 'height': 16},
    'i': {'left': 291.175063, 'top': 473.5, 'width': 18, 'height': 16},
}


def build_panel_h(renderer, frame):
    """Centre the four-parameter label above its marker; keep the depth label 0.018 data units left of its interval."""
    fig, tables = panels.build_panel_h(renderer, frame)
    rows = tables['four_vs_depth_gap'].set_index('model_norm')
    specifications = {
        'Four-parameter': ('four_bias', 'Four-parameter'),
        r'Depth $P_{BS}$': ('depth_stratified', r'Depth $\mathrm{P}_{\mathrm{BS}}$'),
    }
    found = set()
    for label in fig.axes[0].texts:
        source_text = label.get_text()
        if source_text not in specifications:
            continue
        model, display_text = specifications[source_text]
        label.set_text(display_text)
        if model == 'four_bias':
            # One line, centred just above the marker; a wrapped label left of the marker would pass the panel edge.
            label.set_x(float(rows.loc[model, 'delta_simulated']))
            label.set_y(1.14)
            label.set_ha('center')
            label.set_va('bottom')
            found.add(model)
            continue
        interval_left = float(rows.loc[model, 'delta_simulated_ci_low'])
        label_x = interval_left - 0.018
        label.set_x(label_x)
        label.set_ha('right')
        if abs((interval_left - label_x) - 0.018) >= 1e-12:
            raise RuntimeError('Panel-h model-label clearance changed')
        found.add(model)
    if found != {'four_bias', 'depth_stratified'}:
        raise RuntimeError(f'Could not locate both panel-h model labels: {found}')
    return fig, tables


def render(record: Path, output: Path, dpi: int = 600):
    """Render each final panel exactly once, then assemble the vector page."""
    record, output = Path(record).resolve(), Path(output).resolve()
    source = panels.source
    panels.configure_style(source)
    frames, input_paths = source.load_sources(record/'analysis_inputs/strict', record/'analysis_inputs/natural')
    b_points, b_contrasts = source.prepare_two_axis_scatter(
        frames['strict_policy_rows'], frames['strict_policy_contrasts'], ('naive','LTM'), 'return_vs_naive')
    c_points, c_contrasts = source.prepare_two_axis_scatter(
        frames['natural_policy_rows'], frames['natural_policy_contrasts'], ('LTM','experienced'), 'second_vs_return')
    builders = {
        'a': lambda: panels.build_panel_a(source),
        'b': lambda: panels.build_policy_scatter(source,b_points,b_contrasts,('naive','LTM'),'b'),
        'c': lambda: panels.build_policy_scatter(source,c_points,c_contrasts,('LTM','experienced'),'c'),
        'd': lambda: panels.build_panel_d(source,frames['strict_contributions']),
        'e': lambda: panels.build_panel_e(source,frames['natural_policy_rows']),
        'f': lambda: panels.build_calibration(source,frames['natural_simulation_rows'],frames['natural_calibration'],'four_bias','f'),
        'g': lambda: panels.build_panel_g(source,frames['natural_policy_rows'],frames['natural_depth_contrasts']),
        'h': lambda: build_panel_h(source,frames['strict_gap']),
        'i': lambda: panels.build_calibration(source,frames['natural_simulation_rows'],frames['natural_calibration'],'depth_stratified','i'),
    }
    assembly = output/'source_assembly'
    assembly.mkdir(parents=True,exist_ok=True)
    panel_rows, table_rows = [], []
    for letter, build in builders.items():
        figure, tables = build()
        if letter in ('f','i'):
            for axis in figure.axes:
                labels = axis.get_xticklabels()
                if labels:
                    labels[0].set_ha('left')
                    labels[-1].set_ha('right')
        save = panels.save_matched_figure if letter in ('d','h') else panels.save_figure
        paths = save(figure,assembly/'panels'/panels.PANEL_STEM[letter],dpi)
        for path in paths:
            panel_rows.append(dict(panel=letter,role=panels.PANEL_ROLE[letter],format=path.suffix[1:],
                path=path.relative_to(assembly).as_posix(),sha256=panels.sha256(path),bytes=path.stat().st_size))
        for name, table in tables.items():
            path = panels.write_table(table,assembly/'plot_ready'/f'panel_{letter}_{name}.csv')
            table_rows.append(dict(panel=letter,table=name,path=path.relative_to(assembly).as_posix(),
                rows=len(table),sha256=panels.sha256(path),bytes=path.stat().st_size))
    panels.build_shared_bc_legend(assembly,dpi)
    for filename, rows in [('PANEL_MANIFEST.csv',panel_rows),('TABLE_MANIFEST.csv',table_rows)]:
        with (assembly/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frame_map = copy.deepcopy(PANEL_FRAMES)
    g_frame = frame_map['g']
    bottom = float(g_frame['top']) + float(g_frame['height'])
    crop = panels.LAST_G_SAVE_GEOMETRY['crop_bbox_in']
    aspect = (crop[2]-crop[0])/(crop[3]-crop[1])
    g_frame['height'] = float(g_frame['width'])/aspect
    g_frame['top'] = bottom-float(g_frame['height'])
    manifest = {
        'status':'PASS','window_minutes':30,'panel_count':9,
        'layout':{'panel_frames_720':frame_map,'panel_label_frames_720':LABEL_FRAMES},
        'shared_bc_legend':{'frame_720':{'left':337,'top':57,'width':196,'height':18}},
        'inputs':{key:{'path':path.relative_to(record).as_posix(),'sha256':panels.sha256(path)} for key,path in input_paths.items()},
        'panel_a_geometry':panels.LAST_A_GEOMETRY,'panel_e_geometry':panels.LAST_E_GEOMETRY,
        'panel_g_geometry':panels.LAST_G_SAVE_GEOMETRY,
    }
    (assembly/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    composite=export(assembly,output/'composite',composite_stem='ed02_version_record',
                     manifest_name='ED02_SOURCE_REBUILD_COMPOSITE_MANIFEST.json')
    if composite['vector_audit']['embedded_raster_images'] != 0 or composite['png_audit']['size'] != [4500,4500]:
        raise RuntimeError('ED2 vector/raster export contract changed')
    return {'status':'PASS','panel_count':9,'composite':composite}
