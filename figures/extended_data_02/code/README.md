# Extended Data Figure 2: code

The supported entry point is [../build.py](../build.py), run from the repository root. The shared base renderer reads first-30-minute session and saved simulation results; the local drawing modules successively define the component plots, typography and current a–i layout. `export_ed02_vector_figure.py` supplies vector export. These modules form an import chain, not alternative analyses to run separately. See the [panel/data mapping](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

- [export_ed02_vector_figure.py](export_ed02_vector_figure.py)
- [render_ed02_slide3_aligned.py](render_ed02_slide3_aligned.py)
- [render_ed02_slide3_caption_minimal.py](render_ed02_slide3_caption_minimal.py)
- [render_ed02_slide3_compact.py](render_ed02_slide3_compact.py)
- [render_ed02_slide3_layout_optimized.py](render_ed02_slide3_layout_optimized.py)
- [render_ed02_slide3_marker_consistent.py](render_ed02_slide3_marker_consistent.py)
- [render_ed02_slide3_panel_g_labels.py](render_ed02_slide3_panel_g_labels.py)
- [render_ed02_slide3_refinement.py](render_ed02_slide3_refinement.py)
- [render_ed02_slide3_schematic_g_fix.py](render_ed02_slide3_schematic_g_fix.py)
- [render_ed02_slide3_shared_legend.py](render_ed02_slide3_shared_legend.py)
- [render_ed02_slide3_typography.py](render_ed02_slide3_typography.py)
- [render_ed02_slide3_visible_alignment.py](render_ed02_slide3_visible_alignment.py)
- [render_ed02_depth_common_i_adapter.py](../../../reproduction/library/render_ed02_depth_common_i_adapter.py)
- [render_ed02_slide3_upright_policy.py](../../../reproduction/library/render_ed02_slide3_upright_policy.py)
