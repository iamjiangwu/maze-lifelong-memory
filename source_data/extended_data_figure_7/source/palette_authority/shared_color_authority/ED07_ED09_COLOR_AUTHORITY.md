# ED7-ED9 semantic color authority

Status: active hard gate, 2026-08-08. This file governs the rebuilt slides 9-11.
Scientific source currentness and color currentness are separate checks: a current analysis may
still carry obsolete or ad hoc display colors.

## Frozen authorities

- Main Figure 3 accepted manifest:
  `docs/figure_version_records/main_figure3_2026-08-08_full_audit/figure3_source_manifest.json`,
  SHA256 `0038713863766a12aa986aea711dad2c28be9e4bc3f41177be8f84f1cf4b968a`.
  This is the authority for representation colors and metric-specific experience gradients.
- Main Figure 4 accepted manifest:
  `docs/figure_version_records/main_figure4_2026-08-08_full_audit/figure4_source_manifest.json`,
  SHA256 `b2bc2aa95b81d8c104887cfd4260d8770148da4823d76bb5a19a341c7f91ad6c`.
  This explicitly requires decoder data and reward markers to use orange `#E29957`.
- Main Figure 5 accepted manifest:
  `docs/figure_version_records/main_figure5_2026-08-08_full_audit/figure5_source_manifest.json`,
  SHA256 `c69ef80e826542938088e57a5dab373eb2c99d02f85a986bf7c0a4330158e5cc`.
  This is the authority for level purple `#7976A2` and the full-range `magma` level schematic.
- Main Figure 1 accepted schematic source:
  `docs/figure_version_records/main_figure1_2026-08-08_full_audit/source_code/preview_figure1_assembly.py`,
  SHA256 `050865ecd4f1bdea8e4457bc6ea072c37b57a5eca19dc31c0b013a97d52edf17`.
  This freezes the maze quadrant colors and path/endpoint red `#D82929`.
- Shared semantic palette:
  `src/ca_analysis/figure_colors.py`, SHA256
  `1c8997b36cd9fc7db0a46d94d502a48b15ac365aa9ca39aaadc52cfc97173d53`.
- Extended Data color audit:
  `docs/extended_data_figure_panel_refresh_2026-07-12/COLOR_CONSISTENCY_AUDIT_2026-07-16.md`,
  SHA256 `fcd9110baf620cc2474a698be01bca62c469b3527a9e4ce638bbf26a441fe8ab`.
- Current tree-depth control contract:
  `scripts/render_ed12_control_focused.py`, SHA256
  `03e04e796902352968cdf94ca15c2aec4b4340ffd9cb9c82558b821e16e66185`.
  It freezes level `#7976A2`, shuffle/control `#7F7F7F`, full-range `magma`, and a black maze outline.

## Canonical representation and experience colors

| Representation | Base | Naive | 3 mo | 6 mo | 9 mo | 12 mo |
| --- | --- | --- | --- | --- | --- | --- |
| Place map | `#B95A58` | `#D89C98` | `#CB817D` | `#BF6965` | `#A84E4B` | `#78312F` |
| Population vector / decoder | `#E29957` | `#EDBD8F` | `#E8A76F` | `#E29957` | `#BD7137` | `#8A4A1F` |
| Ensemble rate | `#4A5E65` | `#9FB0B5` | `#7D9198` | `#637980` | `#4A5E65` | `#2C3A3F` |
| Level map | `#7976A2` | `#BDB9D6` | `#A19CC3` | `#8A85B2` | `#6F6B99` | `#4E4A75` |

## Figure-specific locks

### ED7 / slide 9

- Panel titles and overall fits use the base color for that representation.
- Same-stage points and within-stage segments use the corresponding metric-specific experience
  gradient, not one generic purple stage palette.
- Long-gap 0-to-3, 3-to-6, 6-to-9 and 9-to-12 rows map to the 3-, 6-, 9- and 12-month
  colors of the displayed metric.
- The generic stage colors `#777777/#AAA9C7/#8885B2/#67629A/#443F7A` and ad hoc transition
  colors `#63666A/#3B82F6/#F97316/#8B5CF6` are forbidden as data-bearing colors.

### ED8 / slide 10

- Reward-proximal decoder points, summaries, proxy landmark and outline use decoder orange
  `#E29957`; level purple is forbidden for this role.
- Matched controls use a neutral gray family. Stable-place evidence uses place red `#B95A58`;
  unstable or null values use neutral gray.
- The coordinate schematic preserves the accepted Main Figure 1 maze quadrants and path/endpoint
  red. Decoder proxy and physical endpoint remain different landmarks and different semantic colors.
- Day-template, route-direction, route-start and rewarded-versus-timeout category colors may remain
  distinct only when the panel crosswalk names the encoded variable and the final version record
  freezes the exact values. A shared key such as `proxy` may not be reused for unrelated semantics.

### ED9 / slide 11

- Tree-depth and level evidence use `#7976A2`; experience panels use the exact level gradient above.
- Place-map evidence uses `#B95A58`; population-vector and ensemble-rate evidence use their base
  representation colors when those identities are displayed.
- Shuffle and null controls use `#7F7F7F`; the tree/maze schematic uses full-range `magma` and a
  black maze outline.
- Ad hoc preferred-level summary colors `#6F63A8` and `#2878B5` are forbidden. Distinguish two
  level-derived subroles with marker shape, line style, or facet structure rather than changing
  representation identity.

## Validation contract

Every final panel must export editable SVG and declare, in a machine-readable crosswalk:

1. the variable encoded by each data-bearing color;
2. its authority file and SHA256;
3. exact required and allowed semantic colors;
4. exact forbidden legacy/ad hoc colors;
5. any continuous colormap and its normalization;
6. any deliberate category-palette exception.

Validators must inspect rendered SVG colors, not only renderer constants. Neutral axes,
antialiasing shades, text, and continuous-colormap samples must be separated from semantic series
colors. A semantic mismatch fails visual approval and blocks PowerPoint integration.
