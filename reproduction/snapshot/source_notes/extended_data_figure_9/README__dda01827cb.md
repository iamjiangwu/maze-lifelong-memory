# Level population-vector neuron-identity shuffle histogram

The observed line is the equal-pair mean of same-level population-vector correlations across 36 recording pairs from 14 mice. Each of 10,000 null draws averages the cached neuron-identity shuffle at the same supported levels within each pair, then averages pairs equally. The masks, C-derived maps, registered neuron cohort and within-pair permutations are inherited unchanged. No within-neuron level normalization or new permutations are introduced.

`null_draws.csv` is the histogram input. `pair_null_draws.csv.gz`, `pair_summary.csv` and `level_masks.csv` expose every aggregation. `summary.json` separates the conditional randomization p-value from the original animal-clustered BH2 result. The histogram spread is permutation variability, not a biological confidence interval. Repeated pairs are retained; their shuffled identities are independent within pairs and do not impose a joint longitudinal identity permutation across shared sessions.

The original manifest hashes the summary/map tables and producer, but does not hash its intermediate NPZ files. Their maps, IDs, masks and matrix entries are verified against those hashed tables; cached null means match within their float32 precision. This bundle now freezes and hashes every complete NPZ input.

Rebuild from frozen derived inputs into a new directory: `python source_code/prepare_shuffle.py --source frozen --output rebuilt`. No raw recording, biological rerun or rendering is required.
