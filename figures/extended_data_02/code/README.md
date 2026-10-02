# Extended Data Figure 2: code

See the [code reading map](../../../docs/CODE.md) for the wrapper, drawing functions and statistical entry points.

The supported entry point is [../build.py](../build.py), run from the repository root. The shared base renderer reads first-30-minute session and saved simulation results; the local drawing modules successively define the component plots, typography and current a–i layout. `export_ed02_vector_figure.py` supplies vector export. These modules form an import chain, not alternative analyses to run separately. See the [panel/data mapping](../data/README.md).

These are implementation modules called by the launcher, not independent raw-data pipelines. To recompute the figure's inferential results together with the other recorded analyses, use `python reproduce.py --statistics-only --output-dir outputs/statistics`. [Environment/setup](../../../docs/REPRODUCING.md).

The [ED driver](../../../reproduction/snapshot/shared/rebuild_current_extended_data_figures.py) combines two drawing passes: `refinement.render()` supplies the assembly, and `aligned.render()` supplies matched-naïve panels d/h. Both use the same current numerical inputs. The import chain, from outer wrapper to base drawing functions, is:

1. [refinement](render_ed02_slide3_refinement.py)
2. [upright policy labels](../../../figures/extended_data_02/code/render_ed02_slide3_upright_policy.py)
3. [panel-g labels](render_ed02_slide3_panel_g_labels.py)
4. [schematic placement](render_ed02_slide3_schematic_g_fix.py)
5. [aligned layout](render_ed02_slide3_aligned.py)
6. [compact layout](render_ed02_slide3_compact.py)
7. [marker consistency](render_ed02_slide3_marker_consistent.py)
8. [shared legend](render_ed02_slide3_shared_legend.py)
9. [typography](render_ed02_slide3_typography.py)
10. [visible alignment](render_ed02_slide3_visible_alignment.py)
11. [panel layout](render_ed02_slide3_layout_optimized.py)
12. [component plots](render_ed02_slide3_caption_minimal.py)

The last layer loads the [depth-support adapter](../../../figures/extended_data_02/code/render_ed02_depth_common_i_adapter.py), which provides the [base renderer](../../../reproduction/snapshot/shared/SOURCE_REBUILD_ED2_ROUND23_BASE_RENDERER.py). The adapter applies the same all-depth-strata support to panel i as to g/h. [export_ed02_vector_figure.py](export_ed02_vector_figure.py) handles vector assembly/export. These layers control drawing or cohort support; they do not introduce additional behavioral windows or statistical families.
