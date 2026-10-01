"""Create/archive the corrected ED1 Round 20 release successor."""

from __future__ import annotations

import promote_ed01_canonical_animal_round19 as base


base.PARENT_NAME = "extended_data_fig01_2026-08-17_round19_canonical_animal_clusters"
base.TARGET_NAME = "extended_data_fig01_2026-08-17_round20_canonical_animal_release"
base.PARENT = base.RECORDS / base.PARENT_NAME
base.TARGET = base.RECORDS / base.TARGET_NAME


if __name__ == "__main__":
    raise SystemExit(base.main())
