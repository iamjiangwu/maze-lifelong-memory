# Reading the reproduction code

Start with a figure's `data/README.md` for the scientific quantities and with its `build.py` to run the figure. The table below identifies the functions that draw it. [Reproduction commands](REPRODUCING.md) and [statistical definitions](STATISTICS.md) are separate guides.

## How inputs reach a renderer

Every figure/table `build.py` forwards its identifier to the root [reproduce.py](../reproduce.py). That launcher verifies the included inputs using [layout.json](../reproduction/layout.json), then stages them under the relative paths expected by the rendering modules. Each layout entry connects a browsable `repository_path` with its temporary `snapshot_path`; a filename seen inside a renderer can be looked up there.

The staged [snapshot launcher](../reproduction/snapshot/reproduce.py), through `native_build()`, selects the figure-specific wrapper listed below. The wrapper supplies input paths, shared imports and the output directory before calling the drawing functions. Shared `ca_analysis` modules are readable under [reproduction/runtime](../reproduction/runtime/); they are staged as `shared/runtime`. Path names retained inside the rendering code refer to this temporary layout and do not require an external analysis repository.

## Figure and table functions

The supported command for every row is `python reproduce.py --figure ID --output-dir outputs/NAME`, from the repository root. These functions are reading entry points within that workflow, rather than separate commands.

| ID | Wrapper and drawing functions |
|---|---|
| `1` | [build_figure1_binary_tree_preview.py](../figures/figure_01/code/build_figure1_binary_tree_preview.py) `build()` sets the maze drawing and calls [build_figure1_ij_spacing.py](../figures/figure_01/code/build_figure1_ij_spacing.py) `build()`, which wraps [rebuild_figure1.py](../figures/figure_01/code/rebuild_figure1.py) `rebuild()`. The latter assembles numerical panels, timelines and trajectories. |
| `2` | [Main driver](../reproduction/snapshot/shared/rebuild_current_main_figures.py) `_render_main2()` supplies inputs to [render_main_figs_v3_publication.py](../figures/figure_02/code/render_main_figs_v3_publication.py) `render_figure2()`. |
| `3` | [rebuild_figure.py](../figures/figure_03/code/rebuild_figure.py) `build()` calls the main driver's `_render_main3()`, which uses [render_main_figs_v3_publication.py](../figures/figure_03/code/render_main_figs_v3_publication.py) `render_figure3()`. |
| `4` | [build_figure4_standardized_pv.py](../figures/figure_04/code/build_figure4_standardized_pv.py) `render()` draws route panels and combines the decoder schematic, decoder results and behavioral fit through its imported panel helpers. |
| `5` | [rebuild_figure5.py](../figures/figure_05/code/rebuild_figure5.py) `configure()` supplies inputs; `rebuild()` reconstructs the level-PV matrix and draws the figure using panel helpers in [preview_figure3_assembly.py](../reproduction/library/preview_figure3_assembly.py). |
| `ed1` | [ED driver](../reproduction/snapshot/shared/rebuild_current_extended_data_figures.py) `render_ed1()` supplies numerical tables to [render_ed01_no_trajectories_source.py](../figures/extended_data_01/code/render_ed01_no_trajectories_source.py) `render()`, using drawing functions in [render_extended_data_figure1.py](../figures/extended_data_01/code/render_extended_data_figure1.py). |
| `ed2` | ED driver `render_ed2()` calls [render_ed02_slide3_refinement.py](../figures/extended_data_02/code/render_ed02_slide3_refinement.py) `render()` and [render_ed02_slide3_aligned.py](../figures/extended_data_02/code/render_ed02_slide3_aligned.py) `render()` for the component panels, then assembles their exports. The [local code guide](../figures/extended_data_02/code/README.md) lists the helper chain. |
| `ed3` | ED driver `render_ed3()` supplies paths to [render_ed03_reward_path_fluency_assembly.py](../figures/extended_data_03/code/render_ed03_reward_path_fluency_assembly.py) `run()` and `build_figure()`. |
| `ed4` | ED driver `render_ed4()` calls [render_ed04_registration_quality_assembly.py](../reproduction/library/render_ed04_registration_quality_assembly.py) `build()`. |
| `ed5` | ED driver `render_ed5()` supplies selected maps and comparison tables to [render_ed05_example_place_maps_assembly.py](../figures/extended_data_05/code/render_ed05_example_place_maps_assembly.py) `assemble()`. |
| `ed6` | ED driver `render_ed6()` supplies saved curve summaries, masks and statistics to [render_ed06_within_session_convergence_assembly.py](../figures/extended_data_06/code/render_ed06_within_session_convergence_assembly.py) `render()`. |
| `ed7` | ED driver `render_ed7()` supplies age-model inputs and drawing helpers to [render_ed07_slide9_powerpoint_matched.py](../figures/extended_data_07/code/render_ed07_slide9_powerpoint_matched.py) `render_six_panel_assembly()`. |
| `ed8` | [build_ed8_standardized_pv.py](../figures/extended_data_08/code/build_ed8_standardized_pv.py) `main()` reads the supplied inputs; `read()` forms animal means, `verify()` recalculates route comparisons, and `draw_accepted()`, `draw_route()` and `compose()` produce the figure. |
| `ed9` | [rebuild_ed9.py](../figures/extended_data_09/code/rebuild_ed9.py) `build()` reads the level-code inputs and assembles the 15 panels. |
| `ed10` | [rebuild_figure.py](../figures/extended_data_10/code/rebuild_figure.py) `build()` calls ED driver `render_ed10()`, which configures [render_ed10_slide12_powerpoint_matched.py](../figures/extended_data_10/code/render_ed10_slide12_powerpoint_matched.py) `main()` and `render_assembly()`. |
| `table1` | [build_workbook.py](../tables/table_01/code/build_workbook.py) `build()` derives cohort counts and writes XLSX; [build_pdf.py](../tables/table_01/code/build_pdf.py) `main()` renders the PDF. |
| `table2` | [rebuild.py](../tables/table_02/code/rebuild.py) `main()` coordinates [build_source.py](../tables/table_02/code/build_source.py), [build_workbook.py](../tables/table_02/code/build_workbook.py), [render_publication_table.py](../tables/table_02/code/render_publication_table.py) and [validate_workbook.py](../tables/table_02/code/validate_workbook.py). |

## Statistical calculations

`python reproduce.py --statistics-only --output-dir outputs/statistics` stages the same inputs and calls [recompute_statistics.py](../reproduction/snapshot/recompute_statistics.py). Its `main()` invokes `run()` in three modules:

| Module | Analysis functions |
|---|---|
| [source_release_main_statistics.py](../reproduction/snapshot/shared/source_release_main_statistics.py) | `main1()` through `main5()`, plus `ed1()`: behavioral contrasts, neural correlations, experience fits, decoder/route results and level-code associations. |
| [source_release_ed_statistics.py](../reproduction/snapshot/shared/source_release_ed_statistics.py) | `ed2()`, `ed3()`, `ed5()`, `ed7()` and `ed9()`: turning biases, dwell comparisons, day combinations, age models and level-code controls. |
| [source_release_root_statistics.py](../reproduction/snapshot/shared/source_release_root_statistics.py) | `ed6()`, `ed8()` and `ed10()`: convergence/variability, reward controls and neural-behavior analyses. |

The modules recalculate results and compare them with the supplied statistics tables. The [ED10 profile helper](../figures/extended_data_10/code/profile_helpers.py) separately defines commonality components and animal-score sign-flip tests. Table counts and calendar intervals are recalculated by the table builders. Figure 3's first-column curves start from saved mean/s.e.m./support tables; its experience fits start from individual observations.
