"""Shared fail-closed neural eligibility gates for the ED10 K=166 chain."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


NEURAL_CONTRACT_REL = Path(
    "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
    "extended_data_fig_11/fixed_neuron_correlation_synthesis_2026-08-09/"
    "NEURAL_ELIGIBILITY_CONTRACT.json"
)
BIOLOGICAL_CONTRACT_REL = Path(
    "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "EXCLUDED_SESSION_CONTRACT.json"
)
ED08_CONTRACT_REL = Path(
    "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed08_reward_proximal_controls/revision_round4_2026-08-09/"
    "ED08_NEURAL_EXCLUSION_CONTRACT.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo: Path, value: Path | str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Eligibility authority is not a JSON object: {path}")
    return payload


def load_neural_eligibility(
    repo: Path,
    neural_contract: Path | str = NEURAL_CONTRACT_REL,
    biological_contract: Path | str = BIOLOGICAL_CONTRACT_REL,
    ed08_contract: Path | str = ED08_CONTRACT_REL,
) -> dict[str, Any]:
    """Validate included cohort schemas, identifier sets and exclusion policy."""

    repo = Path(repo).resolve()
    neural_path = _resolve(repo, neural_contract)
    biological_path = _resolve(repo, biological_contract)
    ed08_path = _resolve(repo, ed08_contract)
    neural = _load_json(neural_path)
    biological = _load_json(biological_path)
    ed08 = _load_json(ed08_path)
    contracts = (
        (neural, "ed10_fixed_neuron_neural_eligibility/v1", ("behavior_only_animals", "neural_excluded_sessions")),
        (biological, "extended_data_biological_design_exclusions/v1", ("excluded_animals", "excluded_sessions")),
        (ed08, "ed08-neural-exclusion-contract/v1", ("excluded_animals", "excluded_sessions")),
    )
    for contract, schema, fields in contracts:
        if contract.get("schema") != schema:
            raise RuntimeError(f"Unexpected eligibility schema; expected {schema}")
        if contract.get("status") != "VALIDATED_COHORT":
            raise RuntimeError(f"Eligibility contract is not validated: {schema}")
        for field, identifier in zip(fields, ("animal", "session"), strict=True):
            rows = contract.get(field)
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not str(row.get(identifier, "")).strip()
                for row in rows
            ):
                raise RuntimeError(f"Invalid {field} identifier list in {schema}")

    excluded_animals = {
        str(row["animal"]) for row in neural.get("behavior_only_animals", [])
    }
    excluded_sessions = {
        str(row["session"]) for row in neural.get("neural_excluded_sessions", [])
    }
    ed08_animals = {str(row["animal"]) for row in ed08.get("excluded_animals", [])}
    ed08_sessions = {str(row["session"]) for row in ed08.get("excluded_sessions", [])}
    biological_animals = {
        str(row["animal"]) for row in biological.get("excluded_animals", [])
    }
    biological_sessions = {
        str(row["session"]) for row in biological.get("excluded_sessions", [])
    }
    if not excluded_animals or not excluded_sessions:
        raise RuntimeError("ED10 neural eligibility contract is empty")
    if excluded_animals != ed08_animals or excluded_sessions != ed08_sessions:
        raise RuntimeError("ED10 and ED8 neural eligibility sets differ")
    if not biological_animals <= excluded_animals:
        raise RuntimeError("Biological excluded animals are not a neural subset")
    if not biological_sessions <= excluded_sessions:
        raise RuntimeError("Biological excluded sessions are not a neural subset")
    policy = neural.get("required_policy", {})
    required_policy = {
        "apply_before_neural_qc",
        "forbid_behavior_only_animals_in_neural_rows",
        "forbid_neural_excluded_sessions",
        "scan_all_identifier_bearing_outputs",
        "forbid_models_and_multiplicity_families",
    }
    if not all(bool(policy.get(key)) for key in required_policy):
        raise RuntimeError("ED10 neural eligibility policy is not uniformly fail-closed")
    return {
        "excluded_animals": excluded_animals,
        "excluded_sessions": excluded_sessions,
        "neural_path": neural_path,
        "neural_sha256": sha256_file(neural_path),
        "biological_path": biological_path,
        "biological_sha256": sha256_file(biological_path),
        "ed08_path": ed08_path,
        "ed08_sha256": sha256_file(ed08_path),
        "n_excluded_animals": len(excluded_animals),
        "n_excluded_sessions": len(excluded_sessions),
        "apply_before_technical_qc": True,
    }


def split_pair(value: object) -> tuple[str, str]:
    text = str(value)
    parts = text.split("->") if "->" in text else text.split("__")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"Invalid pair identifier: {value}")
    return str(parts[0]), str(parts[1])


def session_animal(session: object) -> str:
    match = re.match(r"^(\d+)", str(session))
    if match is None:
        raise RuntimeError(f"Cannot parse animal from session: {session}")
    return match.group(1)


def audit_pair_universe(
    frame: pd.DataFrame,
    authority: dict[str, Any],
    *,
    pair_column: str,
    animal_column: str | None = None,
) -> pd.DataFrame:
    """Return one eligibility row per unique pair before any technical gate."""

    if pair_column not in frame:
        raise KeyError(f"Missing pair column: {pair_column}")
    source = frame[[pair_column] + ([animal_column] if animal_column else [])].copy()
    source[pair_column] = source[pair_column].astype(str)
    grouped = source.groupby(pair_column, sort=True, dropna=False)
    rows: list[dict[str, Any]] = []
    excluded_animals = set(authority["excluded_animals"])
    excluded_sessions = set(authority["excluded_sessions"])
    for pair_id, subset in grouped:
        src, dst = split_pair(pair_id)
        parsed_animals = {session_animal(src), session_animal(dst)}
        explicit_animals: set[str] = set()
        if animal_column:
            explicit_animals = {
                str(value).removesuffix(".0")
                for value in subset[animal_column].dropna().astype(str)
            }
            if explicit_animals and not explicit_animals <= parsed_animals:
                raise RuntimeError(
                    f"Explicit animal does not match pair sessions: {pair_id} / "
                    f"{sorted(explicit_animals)}"
                )
        animals = parsed_animals | explicit_animals
        animal_hits = sorted(animals & excluded_animals)
        session_hits = sorted({src, dst} & excluded_sessions)
        rows.append(
            {
                "pair_id": str(pair_id),
                "source_session": src,
                "target_session": dst,
                "animal_ids": "|".join(sorted(animals)),
                "n_input_rows": int(len(subset)),
                "excluded_animal_hits": "|".join(animal_hits),
                "excluded_session_hits": "|".join(session_hits),
                "eligibility_decision": (
                    "EXCLUDE_BEFORE_QC" if animal_hits or session_hits else "USE_NEURAL"
                ),
            }
        )
    return pd.DataFrame(rows)
