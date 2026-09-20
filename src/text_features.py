"""Text-derived feature canonicalization and compact representation.

This module contains the text-feature processing logic used in the final
Pool2 and Pool3 analyses of the study.
"""

import re
import unicodedata

import numpy as np
import pandas as pd


HABIT_ALIAS_MAP = {
    "natacion": "NATACION",
    "nadar": "NATACION",
    "ciclismo": "CICLISMO",
    "bici": "CICLISMO",
    "bicicleta": "CICLISMO",
    "correr": "CORRER",
    "running": "CORRER",
    "senderismo": "SENDERISMO",
    "montanismo": "SENDERISMO",
    "monte": "SENDERISMO",
    "caminar": "CAMINAR",
    "andar": "CAMINAR",
    "pasear": "CAMINAR",
    "futbol": "FUTBOL",
    "baloncesto": "BALONCESTO",
    "basket": "BALONCESTO",
    "balonmano": "BALONMANO",
    "tenis": "TENIS",
    "padel": "PADEL",
    "paddel": "PADEL",
    "gimnasio": "GIMNASIO",
    "pilates": "PILATES",
    "gimnasia": "GIMNASIA",
    "aerobico": "AEROBICO",
    "golf": "GOLF",
    "boxeo": "BOXEO",
    "crossfit": "CROSS_FIT",
    "cross_fit": "CROSS_FIT",
    "yoga": "YOGA",
    "taichi": "TAICHI",
    "tai_chi": "TAICHI",
    "voley": "VOLEIBOL",
    "voleibol": "VOLEIBOL",
    "paddle_surf": "PADDLE_SURF",
    "hockey_sobrepatines": "HOCKEY_SOBREPATINES",
    "hockey_sobre_patines": "HOCKEY_SOBREPATINES",
}


TEXT_GROUP_MAP = {
    "TXTG_ENDURANCE": [
        "TXT_CAMINAR",
        "TXT_CORRER",
        "TXT_CICLISMO",
        "TXT_NATACION",
        "TXT_ATLETISMO",
        "TXT_SENDERISMO",
    ],
    "TXTG_GYM_STRENGTH": [
        "TXT_GIMNASIO",
        "TXT_CROSS_FIT",
        "TXT_BOXEO",
        "TXT_GIMNASIA",
    ],
    "TXTG_MIND_BODY": [
        "TXT_PILATES",
        "TXT_YOGA",
        "TXT_TAICHI",
        "TXT_BAILE",
    ],
    "TXTG_TEAM_RACKET": [
        "TXT_FUTBOL",
        "TXT_BALONCESTO",
        "TXT_BALONMANO",
        "TXT_VOLEIBOL",
        "TXT_TENIS",
        "TXT_PADEL",
        "TXT_PADDEL",
        "TXT_PADDLE_SURF",
    ],
    "TXTG_OUTDOOR": [
        "TXT_CAMINAR",
        "TXT_CORRER",
        "TXT_CICLISMO",
        "TXT_SENDERISMO",
        "TXT_GOLF",
    ],
}


def normalize_ascii_token(text):
    if text is None:
        return ""

    text = str(text).strip()

    if text == "":
        return ""

    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def canonical_habit_name(name):
    token = normalize_ascii_token(name)

    if token.startswith("txt_"):
        token = token[4:]

    if token == "":
        return None

    token = HABIT_ALIAS_MAP.get(token, token.upper())

    if not str(token).startswith("TXT_"):
        token = f"TXT_{str(token).upper()}"

    return token


def habit_value_to_binary(value):
    if value is None:
        return 0

    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass

    if isinstance(value, str):
        token = normalize_ascii_token(value)

        if token in {
            "",
            "nan",
            "none",
            "null",
            "false",
            "0",
            "no",
        }:
            return 0

        return 1

    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))

    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(value):
                return 0
        except Exception:
            pass

        return int(float(value) != 0.0)

    return int(bool(value))


def normalize_patient_record(record, starting_cols):
    normalized = {}

    for key, value in record.items():
        if key in starting_cols:
            normalized[key] = value
        else:
            canonical = canonical_habit_name(key)

            if canonical is None:
                continue

            normalized[canonical] = max(
                int(normalized.get(canonical, 0)),
                habit_value_to_binary(value),
            )

    return normalized


def rebuild_dataframe(patients, starting_cols, new_columns):
    """Rebuild the aligned dataframe after LLM extraction.

    Equivalent to rebuildDataframe() in the original analysis notebook.
    """
    df = pd.DataFrame(patients)

    for col in starting_cols:
        if col not in df.columns:
            df[col] = None

    for col in sorted(new_columns):
        if col not in df.columns:
            df[col] = 0

    habit_cols = [
        col for col in df.columns
        if col not in starting_cols
    ]

    for col in habit_cols:
        df[col] = (
            df[col]
            .map(habit_value_to_binary)
            .fillna(0)
            .astype(int)
        )

    ordered_cols = starting_cols + sorted(
        [c for c in df.columns if c not in starting_cols]
    )

    return df[ordered_cols]


def coalesce_columns(df, new_col, candidates):
    present = [
        c for c in candidates
        if c in df.columns
    ]

    if not present:
        return df

    df = df.copy()
    df[new_col] = df[present].bfill(axis=1).iloc[:, 0]
    df.drop(columns=present, inplace=True)

    return df


def finalize_text_feature_block(df, starting_cols):
    df = df.copy()

    text_cols = [
        c for c in df.columns
        if c not in starting_cols
    ]

    canonical_groups = {}

    for col in text_cols:
        canonical = canonical_habit_name(col)

        if canonical is None:
            continue

        canonical_groups.setdefault(
            canonical, []
        ).append(col)

    for canonical, cols in canonical_groups.items():
        block = df[cols].apply(
            lambda s: s.map(habit_value_to_binary)
        )
        df[canonical] = block.max(axis=1).astype(int)

    text_cols_after = sorted(
        [
            c for c in df.columns
            if c.startswith("TXT_")
        ]
    )

    keep_cols = [
        c for c in df.columns
        if c in starting_cols or c in text_cols_after
    ]

    df = df[keep_cols].copy()

    if text_cols_after:
        df["TXT_ANY_HABIT"] = (
            df[text_cols_after].sum(axis=1) > 0
        ).astype(int)

        df["TXT_HABIT_COUNT"] = (
            df[text_cols_after]
            .sum(axis=1)
            .astype(int)
        )

    return df


def _make_binary_block(df, cols):
    present = [
        c for c in cols
        if c in df.columns
    ]

    if not present:
        return pd.Series(
            0,
            index=df.index,
            dtype=int,
        )

    block = (
        df[present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    return (block > 0).any(axis=1).astype(int)


def add_compact_text_groups(
    df,
    drop_individual_habits=True,
):
    df = df.copy()

    for group_name, source_cols in TEXT_GROUP_MAP.items():
        df[group_name] = _make_binary_block(
            df,
            source_cols,
        )

    group_cols = [
        c for c in TEXT_GROUP_MAP.keys()
        if c in df.columns
    ]

    if group_cols:
        df["TXTG_N_GROUPS"] = (
            df[group_cols]
            .sum(axis=1)
            .astype(int)
        )

        any_from_groups = (
            df[group_cols].sum(axis=1) > 0
        ).astype(int)

        if "TXT_ANY_HABIT" in df.columns:
            txt_any = (
                pd.to_numeric(
                    df["TXT_ANY_HABIT"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            df["TXT_ANY_HABIT"] = np.maximum(
                txt_any,
                any_from_groups,
            )
        else:
            df["TXT_ANY_HABIT"] = any_from_groups

    if "TXT_HABIT_COUNT" in df.columns:
        df["TXT_HABIT_COUNT"] = (
            pd.to_numeric(
                df["TXT_HABIT_COUNT"],
                errors="coerce",
            ).fillna(0)
        )

    if drop_individual_habits:
        individual_cols = [
            c for c in df.columns
            if c.startswith("TXT_")
            and c not in {
                "TXT_ANY_HABIT",
                "TXT_HABIT_COUNT",
            }
        ]

        df = df.drop(
            columns=individual_cols,
            errors="ignore",
        )

    return df
