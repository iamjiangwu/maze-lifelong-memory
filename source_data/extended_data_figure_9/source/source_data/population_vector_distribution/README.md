# Level population-vector distribution preview

Standalone counterpart of current Figure 2d (Round13, population-vector row first).
Purple: 248 supported same-level Pearson correlations across registered neurons,
from 36 approximately 3-month session pairs in 14 mice. Gray: 2,480,000 saved
neuron-ID-shuffled correlations (10,000 per supported pair-level observation).
Seven ordinary maze levels 0-6; four pairs lack level 6, which is omitted rather
than zero-filled. Same S-rate-qualified registered population and raw C/exposure
level maps as the current level population-vector heatmap and ED9a.

This pools supported pair-level observations equally, analogous to Figure2d's
pooled supported pair-spatial-bin distribution. Pairs supporting seven levels
therefore contribute seven values and the four other pairs contribute six.
This is distinct from ED9a, which averages levels within each pair and pairs equally
before displaying the permutation distribution of that mean. No neuron-wise level
normalization, new permutations, smoothing, significance test or new inference was
introduced. Levels and shuffle draws are not independent biological replicates.
Five initially unrewarded source sessions remain included; other sources and all
targets are rewarded at node 94. Reward conditions are in the observation table.

The same 100 bin edges [-1,1], visible range [-0.45,1], gray shuffle and overlapping
bars/alpha as Figure2d are used; level purple follows the revised level figures.
Each density integrates to one. Source values, all 252 support decisions and
per-pair-level shuffle histograms are exported. Frozen inputs include all cached
permutations, raw maps, IDs, masks and numerical references with input hashes.

Rebuild: `python rebuild_preview.py --inputs frozen --output rebuilt`

This is a preview only. Current assembled figures, version records and ltm-figs
PNGs were not modified.
