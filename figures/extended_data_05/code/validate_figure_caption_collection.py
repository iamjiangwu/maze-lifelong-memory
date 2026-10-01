"""Validate inventory, terminology, and basic LaTeX structure of the caption collection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = (
    REPO_ROOT
    / "docs"
    / "manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)
EVIDENCE_AUDIT = CAPTIONS.with_name("CAPTION_EVIDENCE_AUDIT.md")
MAIN1_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/main_figure1_2026-08-13_round12_no_port_mouse1636"
)
MAIN5_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/main_figure5_2026-08-15_round8_neuron_id_shuffle"
)
MAIN3_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/main_figure3_2026-08-16_round9_available_window_slopes"
)
ED3_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/extended_data_fig03_2026-08-13_round3_latent_day_labels"
)
ED9_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/extended_data_fig09_2026-08-16_round33_panel_f_all36"
)
ED10_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order"
)
ED1_RECORD = (
    REPO_ROOT
    / "docs/figure_version_records/extended_data_fig01_2026-08-15_round18_group80_matched153"
)

EXPECTED = {
    "Figure 1": "abcdefgh",
    "Figure 2": "abcdefghijklmnopq",
    "Figure 3": "abcdefghijklm",
    "Figure 4": "abcdef",
    "Figure 5": "abcdef",
    "Extended Data Figure 1": "abcdefghijk",
    "Extended Data Figure 2": "abcdefghi",
    "Extended Data Figure 3": "abcdef",
    "Extended Data Figure 4": "abcdefghi",
    "Extended Data Figure 5": "abcde",
    "Extended Data Figure 6": "abcdefgh",
    "Extended Data Figure 7": "abcdef",
    "Extended Data Figure 8": "abcdefghi",
    "Extended Data Figure 9": "abcdefghijkl",
    "Extended Data Figure 10": "abc",
}

EXPECTED_MAIN_TITLES = {
    "Figure 1": "Lifelong memory of a complex navigation task.",
    "Figure 2": "CA1 retains a persistent trace after long periods without maze experience.",
    "Figure 3": "Return sessions rebuild detailed maps from an experience-dependent initial state.",
    "Figure 4": "Selective place stability is concentrated near the reward location.",
    "Figure 5": "A compact structural code is associated with exploration efficiency.",
}

FORBIDDEN = (
    "TODO",
    "TBD",
    "PLACEHOLDER",
    "16,884",
    "16884",
    "16,897",
    "16897",
    "reward-proxy decoding error",
    "Day-$i$",
    "matched-neuron rows",
    "neuron-ID-shuffle controls",
    "q_{\\mathrm{BH",
    "CR1",
    "maxT",
    "animal-sign",
    "directional animal-clustered",
    "no pre-sort manual neuron-ID exclusion",
    "BH-adjusted",
    "BH correction",
    "BH-corrected",
    "adjusted $q",
    "corrected $q",
    "saved across-pair",
    "maintained scope",
    "strict joint support",
    "full-session",
    "whole-session",
    "recording day",
    "two-day recording session",
    "two-day session",
    "coverage cycle",
    "No. of experiences",
    "animal-by-experience",
    "place-map",
    "level-map",
    "population-vector",
    "ensemble-rate",
    "whole-maze",
    "reward-approach",
    "reward-decoder",
    "maze-tile",
    "long-bout",
    "long-gap",
    "reward-directed",
    "first-long bout",
    "grey",
    "colour",
    "behaviour",
    "behavioural",
    "centred",
    "centre",
    "analysed",
    "towards",
    "labelled",
    "anaesthetized",
    "anaesthetic",
    "titre",
    "licence",
    "travelled",
    "latent Day 1",
    "latent Day 2",
    "latent day 1",
    "latent day 2",
    "$P_{SF}$",
    "$P_{SA}$",
    "$P_{BF}$",
    "$P_{BS}$",
)

REQUIRED_BY_FIGURE = {
    "Figure 1": (
        "Circles numbered 0 to 5 mark the six decision nodes",
        "Longitudinal design of recurring recording cycles",
        "Each cycle comprised two sessions",
        "Histograms show the fraction of sessions within each condition",
        "Example end node visit timelines from latent control mouse 1636",
        "water port was replaced by a maze wall on latent day1",
        "restored on latent day2",
        "one of the nine animals analyzed in \\textbf{g,h}",
        "dashed horizontal line marks chance performance",
    ),
    "Figure 2": (
        "Each red arrow represents a calcium event",
        "underlying gray lines show animal trajectories",
        "16,901 matched neuron observations",
        "neuron ID shuffle",
        "Values are means across session pairs with 95\\% confidence intervals based on Student's $t$ distribution",
        "47 Day 1 to Day 2 session pairs within the same cycle",
        "13 bins of equal width",
        "bins containing fewer than three observations were omitted",
    ),
    "Figure 3": (
        "observations from 22 animals, each representing one animal in one recording cycle",
        "full Day $i+1$ representation used as the template",
        "No. of cycles",
        "including one short interval pair",
        "adjustment family comprised four correlations",
        "Benjamini--Hochberg-adjusted $p=0.037$ for each",
    ),
    "Figure 4": (
        "Place maps from Day $i$ were combined with calcium activity from matched neurons on Day $i+90$ to decode position across a long gap",
        "The gray region indicates the 360 ms time window used for decoding",
        "Reward decoding error",
        "neuron ID shuffle",
        "data, $5.7\\pm 0.2$ bins; neuron ID shuffle, $6.6\\pm 0.1$ bins",
        "$p=0.014$",
        "($3.7\\pm 0.5$ bins) than at three matched controls ($7.0\\pm 0.3$ bins)",
        "8,600 matched neuron observations",
        "8,438 observations",
    ),
    "Figure 5": (
        "neuron ID shuffle",
        "data, $0.384\\pm0.014$; neuron ID shuffle, $0.243\\pm0.006$",
        "$p=9.2\\times10^{-7}$",
        "16,901 matched neuron observations",
    ),
    "Extended Data Figure 1": (
        "coverage round",
        "complete groups",
        "last 50 transition grid point directly observed in at least 80\\% of sessions",
        "Body mass tracking speed",
        "first 30 min",
        "same 0--14 cm s$^{-1}$ y axis",
        "Each endpoint plot uses its own y axis",
    ),
    "Extended Data Figure 4": ("No. of cycles",),
    "Extended Data Figure 6": ("59 recording cycles from 22 animals",),
    "Extended Data Figure 7": (
        "58 observations from 21 animals, each representing one animal in one recording cycle",
    ),
    "Extended Data Figure 8": (
        "orange outline marks the adjacent decoder reward bin",
        "mean correlation between each matched control location across sessions",
        "Near port fractions for stable and unstable fields",
    ),
    "Extended Data Figure 9": (
        "12,541 matched neuron observations that changed level",
        "from 16,901 observations",
        "Across 36 session pairs from 14 animals",
        "all 36 eligible session pairs from 14 animals",
        "intercept only regressions with standard errors clustered by animal",
        "Student $t(13)$ reference",
        "One sided $P$ values were adjusted together by the Benjamini--Hochberg procedure",
        "no multiplicity correction",
        "1,000 pair weighted eligible cohort null means",
        "Values are means across session pairs with 95\\% confidence intervals based on Student's $t$ distribution",
        "59 recording cycles from 22 animals",
        "No. of cycles",
        "including one short interval pair",
        "Benjamini--Hochberg-adjusted $p=0.034$ in a family of four correlations",
    ),
    "Extended Data Figure 10": (
        "30 session pairs from 14 animals",
        "50 coordinated draws of $K=166$ registered neurons",
        "12.2\\%",
        "36 session pairs from the same 14 animals",
        "18 tests",
        "do not establish mediation or causation",
        "Higher values in both decoder rows therefore indicate poorer decoding",
        "Reward decoding error correlated",
        "whole maze decoding error correlations",
    ),
}

REQUIRED_EVIDENCE_TERMS = (
    "No pre-sort manual neuron-ID exclusion is used",
    "Identity-by-index matching is not treated as biologically interpretable",
    "16,901 matched-neuron observations",
)

MAX_SENTENCE_WORDS = 45


def _strip_comments(text: str) -> str:
    # Preserve escaped percent signs (``\%``), which are ordinary caption text.
    # Only an unescaped percent sign starts a LaTeX comment.
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in text.splitlines())


def _expand_marker(marker: str) -> set[str]:
    result: set[str] = set()
    for token in marker.split(","):
        token = token.strip()
        if re.fullmatch(r"[a-z]", token):
            result.add(token)
            continue
        match = re.fullmatch(r"([a-z])--([a-z])", token)
        if match:
            start, end = (ord(value) for value in match.groups())
            result.update(chr(value) for value in range(start, end + 1))
    return result


def _caption_sections(text: str) -> dict[str, str]:
    escaped_names = sorted((re.escape(name) for name in EXPECTED), key=len, reverse=True)
    pattern = re.compile(r"\\textbf\{(" + "|".join(escaped_names) + r")\.")
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[match.start() : stop]
    return sections


def _sentence_word_counts(line: str) -> list[int]:
    """Return approximate prose sentence lengths after removing LaTeX markup."""

    prose = line.replace("s.e.m.", "SEM").replace("e.g.", "eg").replace("i.e.", "ie")
    prose = re.sub(r"\\textbf\{[^{}]*\}", " ", prose)
    prose = re.sub(r"\\[A-Za-z]+", " ", prose)
    prose = prose.replace("$", "").replace(r"\%", "%")
    sentences = re.split(r"(?<=[.!?])\s+", prose)
    return [
        len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", sentence))
        for sentence in sentences
        if sentence.strip()
    ]


PROBABILITY_LITERAL = re.compile(
    r"(?<![0-9.])(?:"
    r"[0-9]+(?:\.[0-9]+)?\\times10\^\{[+-]?[0-9]+\}"
    r"|[0-9]+\.[0-9]+"
    r")(?![0-9.])"
)


def _significant_digits(literal: str) -> int:
    mantissa = literal.split(r"\times", maxsplit=1)[0]
    digits = re.sub(r"\D", "", mantissa).lstrip("0")
    return len(digits)


def _probability_precision_errors(text: str) -> list[str]:
    """Require two significant digits for every displayed p or q value."""

    errors: list[str] = []
    assignments = re.compile(r"\$(?:p|q)(?:=(?:p|q))?=(?P<body>[^$]+)\$")

    for line_number, line in enumerate(text.splitlines(), start=1):
        candidates: list[str] = []
        for match in assignments.finditer(line):
            candidates.extend(PROBABILITY_LITERAL.findall(match.group("body")))

        sentences = re.split(r"(?<!\d)\.(?=(?:\s+[A-Z\\]|$))", line)
        for sentence in sentences:
            list_match = re.search(
                r"\$[pq]\$ values[^;]{0,160}?\b(?:were|of)\b(?P<body>.*)",
                sentence,
                flags=re.IGNORECASE,
            )
            if list_match:
                candidates.extend(PROBABILITY_LITERAL.findall(list_match.group("body")))

        for literal in candidates:
            digits = _significant_digits(literal)
            if digits != 2:
                errors.append(
                    f"probability_precision:line={line_number}:value={literal}:"
                    f"significant_digits={digits}:expected=2"
                )

    return errors


def _source_authority_errors(sections: dict[str, str]) -> list[str]:
    errors: list[str] = []

    ed1_curves = pd.read_csv(
        ED1_RECORD
        / "plot_ready/ed01_tile_accumulation_whole_group_common_support_current153.csv"
    )
    ed1_counts = (
        ed1_curves.groupby("month_group_key", observed=True)["n_averaged_sessions"]
        .first()
        .astype(int)
        .to_dict()
    )
    if not (
        ed1_counts
        == {
            "experienced_day2": 74,
            "naive_0mo": 23,
            "return_12mo": 10,
            "return_3mo": 17,
            "return_6mo": 17,
            "return_9mo": 12,
        }
        and float(ed1_curves["axis_value_tiles"].min()) == 0.0
        and float(ed1_curves["axis_value_tiles"].max()) == 5200.0
        and ed1_curves["n_active_sessions"].eq(
            ed1_curves["n_averaged_sessions"]
        ).all()
    ):
        errors.append("source_lock:ed1_whole_group_transition_curves")

    ed1_speed = pd.read_csv(
        ED1_RECORD / "plot_ready/ed01_panel_h_first30_session_rows.csv"
    )
    ed1_speed["mode_time_distance_rate_cm_s"] = pd.to_numeric(
        ed1_speed["mode_time_distance_rate_cm_s"], errors="coerce"
    )
    ed1_speed_valid = ed1_speed.loc[
        ed1_speed["mode_time_distance_rate_cm_s"].notna()
    ]
    ed1_speed_counts = (
        ed1_speed_valid.groupby(
            ["analysis_mode", "day", "month"], observed=True
        )["session_id"]
        .nunique()
        .to_dict()
    )
    expected_ed1_speed_counts = {
        **{("explore", "day1", month): count for month, count in zip((0, 3, 6, 9, 12), (24, 14, 12, 7, 4), strict=True)},
        **{("explore", "day2", month): count for month, count in zip((0, 3, 6, 9, 12), (22, 14, 12, 7, 4), strict=True)},
        **{("reward", day, month): count for day in ("day1", "day2") for month, count in zip((0, 3, 6, 9, 12), (22, 14, 12, 7, 4), strict=True)},
    }
    if not (
        len(ed1_speed) == 240
        and ed1_speed["session_id"].nunique() == 120
        and set(ed1_speed["window_id"]) == {"first_30_min"}
        and ed1_speed_counts == expected_ed1_speed_counts
    ):
        errors.append("source_lock:ed1_panel_h_first30_shared_scale")

    main3_stats = pd.read_csv(
        MAIN3_RECORD
        / "source_data/long_gap/main3_experience_trend_clustered_stats_no_manual_mask.csv"
    ).set_index("metric")
    main3_family = {
        "place_map_correlation",
        "level_map_correlation",
        "pv_correlation",
        "ensemble_rate_correlation",
    }
    main3_family_rows = main3_stats.loc[list(main3_family)]
    main3_excluded_rows = main3_stats.drop(index=list(main3_family))
    if not (
        len(main3_stats) == 6
        and main3_family_rows["in_bh4_correlation_family"].astype(bool).all()
        and not main3_excluded_rows["in_bh4_correlation_family"].astype(bool).any()
        and main3_excluded_rows["q_bh4_correlation_family"].isna().all()
        and abs(float(main3_stats.at["place_map_correlation", "q_bh4_correlation_family"]) - 0.03669999885600253) < 1e-15
        and abs(float(main3_stats.at["level_map_correlation", "q_bh4_correlation_family"]) - 0.033881711485611685) < 1e-15
        and abs(float(main3_stats.at["pv_correlation", "q_bh4_correlation_family"]) - 0.03669999885600253) < 1e-15
        and abs(float(main3_stats.at["ensemble_rate_correlation", "q_bh4_correlation_family"]) - 0.1404357256002968) < 1e-15
    ):
        errors.append("source_lock:main3_successive_cycle_bh4")

    main_audit_paths = tuple(MAIN1_RECORD.rglob("*AUDIT*.json")) + (
        MAIN1_RECORD / "DISPLAY_CHANGE_AUDIT.json",
    )
    main_audit_path = next((path for path in main_audit_paths if path.is_file()), None)
    if main_audit_path is None:
        errors.append("source_lock:main1_af23_audit_missing")
        main_audit: dict[str, object] = {}
    else:
        main_audit = json.loads(main_audit_path.read_text(encoding="utf-8"))
    main_rows = pd.read_csv(
        MAIN1_RECORD / "plot_ready/exploration_first_chunk_current_plus_new_points.csv"
    )
    main_counts = main_rows.groupby("group").size().to_dict()
    if main_audit.get("status") != "PASS" or main_counts != {
        "LTM D1 (all)": 56,
        "Naive D1 (rewarded)": 23,
    }:
        errors.append("source_lock:main1_panel_e")

    route_candidates = (
        MAIN1_RECORD / "plot_ready/ed04_session_route_current_plus_new12.csv",
        MAIN1_RECORD / "plot_ready/session_route_summary.csv",
    )
    route_path = next((path for path in route_candidates if path.is_file()), None)
    if route_path is None:
        errors.append("source_lock:main1_af23_route_missing")
    else:
        route = pd.read_csv(route_path)
        session_col = "session_id" if "session_id" in route.columns else "session"
        row = route.loc[route[session_col].astype(str).eq("489-3day1")]
        route_ok = len(row) == 1 and all(
            (
                abs(float(row.iloc[0]["route_fraction_correct"]) - 4 / 6) < 1e-15,
                int(row.iloc[0]["n_decisions_observed"]) == 4,
                int(row.iloc[0]["n_decisions_inferred"]) == 2,
                int(row.iloc[0]["n_decisions_scored"]) == 6,
                str(row.iloc[0]["all6_observed"]).lower() == "false",
                str(row.iloc[0]["all6_resolved"]).lower() == "true",
            )
        )
        if not route_ok:
            errors.append("source_lock:main1_af23_route")

    ed3_audit_candidates = tuple(ED3_RECORD.rglob("*AUDIT*.json")) + (
        ED3_RECORD / "ED03_VALIDATION.json",
    )
    ed3_audit_path = next((path for path in ed3_audit_candidates if path.is_file()), None)
    if ed3_audit_path is None:
        errors.append("source_lock:ed3_af23_audit_missing")

    ed9 = json.loads(
        (
            ED9_RECORD
            / "source_validation/round19_panel_c_corrected36/source_data/"
            "D_CORRECTED36_VALIDATION.json"
        ).read_text(encoding="utf-8")
    )
    ed9_summary = ed9.get("summary", {})
    ed9_stats = ed9_summary.get("statistics", {})
    if not (
        ed9.get("status") == "PASS"
        and ed9_summary.get("n_pairs") == 36
        and ed9_summary.get("n_mice") == 14
        and ed9_summary.get("n_rate_gated_observations") == 16901
        and abs(float(ed9_summary.get("data_mean")) - 0.117744789365191) < 1e-15
        and abs(float(ed9_summary.get("shuffle_mean")) - 0.10882834112532862)
        < 1e-15
        and abs(float(ed9_stats.get("p_directional")) - 0.00947165887533162)
        < 1e-15
    ):
        errors.append("source_lock:ed9_panel_c")

    ed9_panel_f = pd.read_csv(
        ED9_RECORD
        / "source_data/panel_f_inference_round24/be_roles/"
        "B2_EXCLUSION_CLEAN32_CR1_BH2_STATISTICS.csv"
    ).set_index("contrast")
    ed9_panel_f_pairs = pd.read_csv(
        ED9_RECORD
        / "source_data/panel_f_inference_round24/be_roles/"
        "B2_EXCLUSION_CLEAN32_PAIR_CONTRASTS.csv"
    )
    exclusion_contract = json.loads(
        (ED9_RECORD / "exclusion_audit/EXCLUDED_SESSION_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    forbidden_animals = {
        str(row["animal"]) for row in exclusion_contract["excluded_animals"]
    }
    forbidden_sessions = {
        str(row["session"]) for row in exclusion_contract["excluded_sessions"]
    }
    pair_sessions = {
        session
        for pair_id in ed9_panel_f_pairs["pair_id"].astype(str)
        for session in pair_id.split("__")
    }
    pair_animals = set(ed9_panel_f_pairs["animal"].astype(str))
    panel_f_exact = (
        len(ed9_panel_f_pairs) == 32
        and ed9_panel_f_pairs["animal"].astype(str).nunique() == 14
        and len(ed9_panel_f) == 2
        and ed9_panel_f["cluster_df"].eq(13).all()
        and ed9_panel_f["n_pairs"].eq(32).all()
        and ed9_panel_f["n_mice"].eq(14).all()
        and ed9_panel_f["covariance"]
        .eq("CR1 intercept-only; Student t(G-1)")
        .all()
        and abs(
            float(
                ed9_panel_f.at[
                    "geometric8_minus_horizontal8",
                    "q_bh_two_raw_planned_contrasts",
                ]
            )
            - 2.8670561454901405e-05
        )
        < 1e-18
        and abs(
            float(
                ed9_panel_f.at[
                    "geometric8_minus_vertical8",
                    "q_bh_two_raw_planned_contrasts",
                ]
            )
            - 1.6728044637612076e-06
        )
        < 1e-18
        and not (pair_sessions & forbidden_sessions)
        and not (pair_animals & forbidden_animals)
    )
    if not panel_f_exact:
        errors.append("source_lock:ed9_panel_f_clean32_cr1_bh2")

    main5_manifest = json.loads(
        (MAIN5_RECORD / "figure5_source_manifest.json").read_text(encoding="utf-8")
    )
    main5_seriation = json.loads(
        (
            MAIN5_RECORD
            / "source_validation/refinement_2026_08_12/"
            "figure5b_historical_hierarchical_refined_order.json"
        ).read_text(encoding="utf-8")
    )
    main5_caption = sections.get("Figure 5", "")
    if not (
        main5_manifest.get("style", {}).get("heatmap_ordering_method")
        == "historical_average_linkage_then_stable_argmax"
        and main5_seriation.get("is_exact_permutation") is True
        and main5_seriation.get("row_count") == 16901
        and main5_seriation.get("linkage") == "average"
        and main5_seriation.get("refined_order_sha256")
        == "c7b3467afb3f04a22a96f006780ad4aad5336239323e0375090ade89469bd1d4"
        and "average linkage hierarchical clustering" in main5_caption
        and "stable grouping according to the preferred level on Day $i$" in main5_caption
    ):
        errors.append("source_lock:main5_seriation")

    ed10_manifest = json.loads(
        (ED10_RECORD / "ED10_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    ed10_validation = json.loads(
        (
            ED10_RECORD / "source_validation/ED10_ROUND6_SOURCE_VALIDATION.json"
        ).read_text(encoding="utf-8")
    )
    raw_matrix = pd.read_csv(
        ED10_RECORD / "plot_ready/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv"
    )
    sign_audit = json.loads(
        (
            ED10_RECORD
            / "source_validation/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json"
        ).read_text(encoding="utf-8")
    )
    commonality = pd.read_csv(ED10_RECORD / "plot_ready/REWARD_LEVEL_COMMONALITY.csv")
    pooled = commonality.loc[commonality["specification"].eq("pooled")].set_index(
        "outcome"
    )
    policy = pd.read_csv(
        ED10_RECORD / "plot_ready/level_policy_bridge_stats_corrected36.csv"
    ).set_index("y")
    ed10_exact = (
        ed10_manifest.get("status") == "ACCEPTED"
        and ed10_validation.get("status") == "PASS"
        and sign_audit.get("status") == "PASS"
        and all(sign_audit.get("checks", {}).values())
        and len(raw_matrix) == 18
        and raw_matrix["n_pairs"].eq(30).all()
        and raw_matrix["n_animals"].eq(14).all()
        and abs(
            float(
                raw_matrix.loc[
                    (raw_matrix["block"] == "reward_location")
                    & (raw_matrix["outcome"] == "first_long_bout_time"),
                    "pooled_pearson_r",
                ].iloc[0]
            )
            - 0.4583382549928935
        )
        < 1e-15
        and abs(float(pooled.at["earlier_first_long", "unique_a"]) - 0.12212963526456755)
        < 1e-15
        and abs(float(pooled.at["earlier_first_long", "shared"]) - 0.08794432072536273)
        < 1e-15
        and abs(float(pooled.at["exploration_efficiency", "unique_b"]) - 0.35136697389988525)
        < 1e-15
        and int(policy.at["Lo_depth_weighted", "n_pairs"]) == 36
        and int(policy.at["Lo_depth_weighted", "n_animals"]) == 14
        and abs(float(policy.at["Lo_depth_weighted", "cluster_p"]) - 2.130358637781239e-6)
        < 1e-18
        and abs(
            float(policy.at["Lo_depth_weighted", "cluster_q_bh_window_family"])
            - 1.4115440081044338e-5
        )
        < 1e-18
    )
    if not ed10_exact:
        errors.append("source_lock:ed10")

    pointer_text = (
        REPO_ROOT / "docs/figure_version_records/CURRENT_MAIN_FIGURES.md"
    ).read_text(encoding="utf-8") + (
        REPO_ROOT / "docs/figure_version_records/CURRENT_EXTENDED_DATA_FIGURES.md"
    ).read_text(encoding="utf-8")
    for record in (
        MAIN1_RECORD,
        MAIN3_RECORD,
        MAIN5_RECORD,
        ED3_RECORD,
        ED9_RECORD,
        ED10_RECORD,
    ):
        if record.name not in pointer_text:
            errors.append(f"current_pointer:{record.name}")

    return errors


def validate(path: Path = CAPTIONS) -> list[str]:
    errors: list[str] = []
    raw = path.read_text(encoding="utf-8")
    text = _strip_comments(raw)

    if text.count("\\begin{document}") != 1 or text.count("\\end{document}") != 1:
        errors.append("document_environment")
    if text.count("{") != text.count("}"):
        errors.append("unbalanced_braces")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if len(re.findall(r"(?<!\\)\$", line)) % 2:
            errors.append(f"unbalanced_math_delimiters:line={line_number}")
        if line.startswith(r"\textbf{"):
            counts = _sentence_word_counts(line)
            if counts and max(counts) > MAX_SENTENCE_WORDS:
                errors.append(
                    f"long_sentence:line={line_number}:words={max(counts)}:"
                    f"limit={MAX_SENTENCE_WORDS}"
                )
    for token in FORBIDDEN:
        if token.lower() in text.lower():
            errors.append(f"forbidden:{token}")
    errors.extend(_probability_precision_errors(text))
    if "end-node visits" in text.lower():
        errors.append("terminology:end node visits")
    sections = _caption_sections(text)
    if set(sections) != set(EXPECTED):
        errors.append(
            "figure_inventory:" + ",".join(sorted(set(EXPECTED).symmetric_difference(sections)))
        )

    for name in EXPECTED:
        heading_count = len(re.findall(rf"\\textbf\{{{re.escape(name)}\.", text))
        if heading_count != 1:
            errors.append(f"heading_count:{name}:{heading_count}")

    for name, title in EXPECTED_MAIN_TITLES.items():
        expected_heading = rf"\textbf{{{name}. {title}}}"
        if expected_heading not in text:
            errors.append(f"title_mismatch:{name}:expected={title}")

    for name, letters in EXPECTED.items():
        section = sections.get(name, "")
        observed: set[str] = set()
        for marker in re.findall(
            r"(?m)^\\textbf\{([a-z](?:,[a-z]|--[a-z])*,?)\}", section
        ):
            observed.update(_expand_marker(marker.rstrip(",")))
        expected = set(letters)
        if observed != expected:
            errors.append(
                f"panel_inventory:{name}:expected={''.join(sorted(expected))}:"
                f"observed={''.join(sorted(observed))}"
            )

    required_terms = (
        "16,901 matched neuron observations",
        "one animal in one recording cycle",
        "Reward decoding error",
        "neuron ID shuffle",
        "label shuffle",
        "time in maze",
        "maze tile transitions",
    )
    for term in required_terms:
        if term.lower() not in text.lower():
            errors.append(f"missing_term:{term}")

    for name, terms in REQUIRED_BY_FIGURE.items():
        section = sections.get(name, "")
        for term in terms:
            if term.lower() not in section.lower():
                errors.append(f"missing_scoped_term:{name}:{term}")

    figure4 = sections.get("Figure 4", "")
    for stale_role in ("of the source session", "of the return session"):
        if stale_role.lower() in figure4.lower():
            errors.append(f"verbose_role:Figure 4:{stale_role}")

    evidence_text = EVIDENCE_AUDIT.read_text(encoding="utf-8")
    for term in REQUIRED_EVIDENCE_TERMS:
        if term.lower() not in evidence_text.lower():
            errors.append(f"missing_evidence_term:{term}")

    errors.extend(_source_authority_errors(sections))

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {len(EXPECTED)} captions with exact panel inventories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
