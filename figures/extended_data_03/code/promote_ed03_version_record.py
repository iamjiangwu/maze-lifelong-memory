"""Freeze the accepted Extended Data Figure 3 build as an immutable version record.

Mirrors the main-figure promotion pattern: copy the accepted exports, review deck,
plot-ready inputs and rendering code into a dated folder, then re-verify byte identity
of every copy and re-assert the scientific contract before writing the manifest and
hash inventory.

Extended Data Figure 3 is the reward-path fluency and first-visit choice figure built
from slide 4 of ``sup-figs-v2.pptx`` under the merged numbering. It is unrelated to the
old Extended Data Figure 3 (branch-policy exploration), which is slide 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed03_reward_path_fluency"
)
ASSEMBLY = BUILD_ROOT / "assembly"
REVIEW = BUILD_ROOT / "review_2026-08-05"
SOURCE_TABLES = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "extended_data_figures"
    / "extended_data_fig_04"
    / "04_source_tables"
)
RECORD_ROOT = (
    REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig03_2026-08-06_round7"
)
POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"

STEM = "ed03_reward_path_fluency_assembly"
ACCEPTED_DECK = "ed03_revision_review_round7.pptx"
ACCEPTED_DECK_RENDER = "ed03_review_round7_powerpoint.png"

# (source, destination relative to the record root)
PAYLOAD: tuple[tuple[Path, str], ...] = (
    (ASSEMBLY / f"{STEM}.svg", "ed03_version_record.svg"),
    (ASSEMBLY / f"{STEM}.pdf", "ed03_version_record.pdf"),
    (ASSEMBLY / f"{STEM}.png", "ed03_version_record_600dpi.png"),
    (ASSEMBLY / f"{STEM}.tiff", "ed03_version_record_600dpi.tiff"),
    (ASSEMBLY / "ED03_ASSEMBLY_MANIFEST.json", "ED03_ASSEMBLY_MANIFEST.json"),
    (ASSEMBLY / "ED03_VALIDATION.csv", "ED03_VALIDATION.csv"),
    (REVIEW / ACCEPTED_DECK, "ed03_review_record.pptx"),
    (REVIEW / ACCEPTED_DECK_RENDER, "ed03_review_record_render.png"),
    (REVIEW / "REPORT.md", "REVIEW_REPORT.md"),
    (SOURCE_TABLES / "fluency_plot_rows.csv", "plot_ready/fluency_plot_rows.csv"),
    (SOURCE_TABLES / "fluency_statistics.csv", "plot_ready/fluency_statistics.csv"),
    (
        SOURCE_TABLES / "ed04_depth_small_multiples_current_plus_new12.csv",
        "plot_ready/ed04_depth_small_multiples_current_plus_new12.csv",
    ),
    (
        SOURCE_TABLES / "ed04_route_definition_current_plus_new12.csv",
        "plot_ready/ed04_route_definition_current_plus_new12.csv",
    ),
    (
        REPO_ROOT / "scripts" / "render_ed03_reward_path_fluency_assembly.py",
        "source_code/render_ed03_reward_path_fluency_assembly.py",
    ),
    (
        REPO_ROOT / "scripts" / "render_ed04_fluency_story_current.py",
        "source_code/render_ed04_fluency_story_current.py",
    ),
    (
        REPO_ROOT / "scripts" / "render_ed02_ed04_current_panels.py",
        "source_code/render_ed02_ed04_current_panels.py",
    ),
    (Path(__file__), "source_code/promote_ed03_version_record.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    missing = [str(src) for src, _ in PAYLOAD if not src.exists()]
    if missing:
        raise SystemExit(f"Refusing to promote; missing payload: {missing}")

    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, str, int]] = []
    for source, relative in PAYLOAD:
        destination = RECORD_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copy_hash = sha256(destination)
        record(
            f"copy_identical_{relative.replace('/', '_')}",
            source_hash == copy_hash,
            f"{relative} sha256 {copy_hash[:16]}",
        )
        copied.append((relative, copy_hash, destination.stat().st_size))

    # Re-assert the scientific contract from the frozen payload, not from the live build.
    validation = list(
        csv.DictReader((RECORD_ROOT / "ED03_VALIDATION.csv").open(encoding="utf-8"))
    )
    failed = [row for row in validation if row["status"] != "PASS"]
    record(
        "frozen_validation_all_pass",
        not failed and len(validation) >= 34,
        f"{len(validation)} checks, {len(failed)} failing",
    )

    manifest = json.loads(
        (RECORD_ROOT / "ED03_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit = manifest["audit"]
    claims = audit["panels"]["a-d"]
    expected_claims = {
        "ed4_broad_pre_reward_explore": (-0.307587, 63, 20),
        "ed4_broad_post_reward_explore": (-0.082389, 65, 21),
        "ed4_pre_reward_direct_chain_specificity_check": (0.181756, 61, 20),
        "ed4_reward_approach_after_reward": (-0.186428, 65, 21),
    }
    for claim_id, (beta, rows, mice) in expected_claims.items():
        got = claims[claim_id]
        record(
            f"frozen_claim_{claim_id}",
            abs(got["beta"] - beta) < 1e-5
            and got["n_rows"] == rows
            and got["n_mice"] == mice,
            f"beta={got['beta']:.6f} n={got['n_rows']}/{got['n_mice']}",
        )

    support = audit["panels"]["f"]["support"]
    expected_support = {
        "naive": (23, 23),
        "experienced_day2": (74, 24),
        "return": (56, 17),
        "latent_day2": (9, 9),
    }
    for condition, (sessions, animals) in expected_support.items():
        got = support[condition]
        record(
            f"frozen_support_{condition}",
            got["n_sessions"] == sessions and got["n_animals"] == animals,
            f"{got['n_sessions']} sessions / {got['n_animals']} mice",
        )
    record(
        "frozen_latent_cohort_is_nine_not_thirteen",
        support["latent_day2"]["n_animals"] == 9,
        "guards against the superseded 13-animal UnrewardedDay2 table",
    )
    record(
        "frozen_reward_node_94",
        audit["panels"]["e"]["reward_node"] == 94
        and audit["panels"]["e"]["terminal_cell_xy"] == [14, 6],
        "canonical reward node 94 at grid (14,6)",
    )
    record(
        "frozen_no_background_grid",
        audit["grid_lines_visible"] == 0,
        f"{audit['grid_lines_visible']} gridlines",
    )
    record(
        "frozen_return_colour",
        audit["panels"]["f"]["palette"]["return"] == "#E15759"
        and audit["fluency_return_colour"] == "#E15759",
        "softened Return day red applied consistently",
    )

    hashes_path = RECORD_ROOT / "RECORD_HASHES.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for relative, digest, size in sorted(copied):
            writer.writerow([relative, digest, size])

    result = {
        "figure": "Extended Data Figure 3 (merged numbering; slide 4 of sup-figs-v2.pptx)",
        "record": str(RECORD_ROOT.relative_to(REPO_ROOT)),
        "accepted_round": 7,
        "payload_files": len(copied),
        "validation": checks,
        "validation_status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
    }
    (RECORD_ROOT / "RECORD_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = promote()
    failed = [c for c in result["validation"] if c["status"] != "PASS"]
    print(
        json.dumps(
            {
                "record": result["record"],
                "payload_files": result["payload_files"],
                "checks": len(result["validation"]),
                "status": result["validation_status"],
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
