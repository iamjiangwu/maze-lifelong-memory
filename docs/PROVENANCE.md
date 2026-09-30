# Source-data snapshot provenance

This standalone repository was prepared from the already executed and verified source-data release dated **2026-09-30**, rather than from the full scientific worktree or its Git history. The 17 artifact folders reproduce the latest approved local main figures, Extended Data figures and supplementary tables selected in that release.

## Frozen inputs

- Original sharing archive: `source_data_release_2026-09-30_source_only.zip`.
- Archive SHA256: `b20996ec8fb661001ac60fca5d522d66f07a8a9d32381fc6569f1102e8f44c8a`.
- Frozen `source_data/RELEASE_MANIFEST.json` SHA256: `bbf3381b7cfa7c51e870029414b65f0a5a10dfcaa48da61206b444824c07a992`.
- Distributed hash-bound source/code/documentation files: **969**, plus the manifest itself.
- Source scope: sufficient derived numerical observations, example maps, saved null/simulation draws and fixed illustrations; no raw acquisition files and no accepted composite artwork as input.

Every extracted file under `source_data/` is retained byte-for-byte. The outer repository README, Git attributes/ignore rules and validation receipts are packaging additions and do not alter the frozen source release. Historical local absolute paths in manifests and receipts identify the originating builds; reproduction uses the included relative inputs and stages them in temporary folders.

The [release guide](../source_data/README.md) describes its estimands and replay boundary. Rendering uses frozen numerical plot inputs; the separate statistical replay recalculates the recorded means, intervals and test results from included per-unit-derived observations, using the accepted statistical units and correction families. It does not rerun acquisition-to-measurement processing.

## Validation

`validation/REPRODUCTION_REPORT.json` records all 17 approved artwork/table comparisons. `validation/STATISTICS_REPORT.json` records the 53 analysis-group replay. `validation/ARCHIVE_VALIDATION.json` records extraction/input integrity and statistical replay of the source-only archive, without finished references. They are evidence from the original executed snapshot, not future generated output.

Git stores the frozen files without line-ending conversion. Future scientific or visual updates should create a new explicitly versioned source snapshot with updated provenance, checksums and executed validation; do not silently relabel changed inputs as this snapshot.

## Licensing and citation

The source worktree did not provide a repository-wide licence to carry into this snapshot. No new licence, DOI, manuscript citation or public-access claim is invented here. Preserve existing notices in individual files. The author must select appropriate code/data licences and supply the manuscript citation before public release. Creating a private GitHub repository does not resolve these publication metadata fields.

The separate raw-data/minimal-analysis repository is not included or modified by this preparation.
