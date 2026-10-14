# Shared reproduction code

`layout.json` lists all 18 artifacts and checksums for their inputs and code.
Its optional `loader_paths` entries provide local paths expected by the
behavioral and table renderers. The root launcher creates those paths in a
temporary directory.

`runtime/release_rendering/` contains shared drawing, geometry, statistics
and export helpers. `runtime/ca_analysis/` contains the compact plotting
helpers used by the behavioral displays. The remaining three Python modules
adapt behavioral inputs and calculate their statistics.

No completed publication figure, raw neural recording, historical audit or
research analysis pipeline is required here.
