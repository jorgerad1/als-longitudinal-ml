"""Compact longitudinal T1-to-T2 feature construction.

This module implements the temporal representation used in the final
Pool3 analysis.
"""

import numpy as np
import pandas as pd


TEMPORAL_DOMAIN_MAP = {
    "SOCIO": [
        "TIPOCASA",
        "ESTUDIOS",
        "FORM_CONT",
        "SOCIOECO",
        "TIPOTRABAJO",
        "VIVESOLO",
    ],
    "ACTIVITY": [
        "DEPORTE",
        "DEPORTEFREC",
    ],
    "BODY": [
        "DAR",
        "TIPOPESO",
        "PESO",
        "ALTURA",
        "FORMA",
        "ACUMULOGRASO",
        "CAMBIOBRUSCOPESO",
    ],
    "SUBSTANCE_SLEEP": [
        "DROGAS",
        "CANNABIS",
        "COCAINA",
        "OPIACEOS",
        "DISEÑO",
        "OTRASDROGAS",
        "TABACO",
        "ALCOHOL",
        "HORASSUEÑO",
    ],
    "PSYCHOSOCIAL": [
        "PREOCUPAC",
        "PERSONALIDAD",
        "SOCIALIZA",
    ],
    "DIET": [
        "ALIMENTACION",
        "BEBIDASAZUCAR",
        "DULCES",
        "TIPOCOMIDA",
        "FASTFOOD",
        "FRITO",
        "PLANCHA",
        "EMPANADO",
        "HERVIDO",
        "GUISADO",
        "CRUDO",
        "OTRACOCINA",
        "COMIDAS_DIA",
        "CANT_COMIDA",
        "SACIEDAD",
    ],
}


SELECTED_DELTA_BASES = [
    "PESO",
    "ALTURA",
    "DEPORTEFREC",
    "HORASSUEÑO",
    "PREOCUPAC",
    "PERSONALIDAD",
    "SOCIALIZA",
    "ALIMENTACION",
    "BEBIDASAZUCAR",
    "DULCES",
    "FASTFOOD",
    "COMIDAS_DIA",
    "CANT_COMIDA",
    "SACIEDAD",
]


def infer_t1_t2_pairs(df):
    """Identify T1/T2 column pairs using the final Pool3 rules."""
    cols = set(df.columns)
    pair_map = {}

    for col in cols:
        if not isinstance(col, str):
            continue

        if "1" in col:
            c2 = col.replace("1", "2", 1)

            if c2 in cols:
                base = col.replace("1", "", 1)
                pair_map[base] = (col, c2)

    # Special naming case in the source questionnaire.
    if (
        "ACUMULOGRASO" in cols
        and "ACUMULOGRASO2" in cols
    ):
        pair_map["ACUMULOGRASO"] = (
            "ACUMULOGRASO",
            "ACUMULOGRASO2",
        )

    return dict(sorted(pair_map.items()))


def add_temporal_summary_features(
    df,
    drop_raw_t2=True,
):
    """Construct compact T1-to-T2 change descriptors.

    Raw paired T2 variables are removed by default after the temporal
    descriptors have been generated, reproducing the final Pool3
    representation.
    """
    df = df.copy()

    pair_map = infer_t1_t2_pairs(df)

    if not pair_map:
        return df

    change_df = pd.DataFrame(index=df.index)
    observed_df = pd.DataFrame(index=df.index)

    for base, (c1, c2) in pair_map.items():
        s1 = pd.to_numeric(
            df[c1],
            errors="coerce",
        )
        s2 = pd.to_numeric(
            df[c2],
            errors="coerce",
        )

        observed = (
            s1.notna() & s2.notna()
        ).astype(int)

        changed = pd.Series(
            np.where(
                observed.astype(bool),
                (s1 != s2).astype(int),
                np.nan,
            ),
            index=df.index,
        )

        change_df[base] = changed
        observed_df[base] = observed

    df["T12_N_OBSERVED_PAIRS"] = (
        observed_df.sum(axis=1).astype(int)
    )

    df["T12_N_CHANGED_ALL"] = (
        change_df
        .fillna(0)
        .sum(axis=1)
        .astype(int)
    )

    df["T12_PROP_CHANGED_ALL"] = (
        df["T12_N_CHANGED_ALL"]
        / df["T12_N_OBSERVED_PAIRS"].replace(
            0,
            np.nan,
        )
    )

    for domain, bases in TEMPORAL_DOMAIN_MAP.items():
        present = [
            b for b in bases
            if b in change_df.columns
        ]

        if not present:
            continue

        observed_count = (
            observed_df[present]
            .sum(axis=1)
        )

        df[f"T12_N_CHANGED_{domain}"] = (
            change_df[present]
            .fillna(0)
            .sum(axis=1)
            .astype(int)
        )

        df[f"T12_PROP_CHANGED_{domain}"] = (
            df[f"T12_N_CHANGED_{domain}"]
            / observed_count.replace(
                0,
                np.nan,
            )
        )

    delta_cols = []

    for base in SELECTED_DELTA_BASES:
        if base not in pair_map:
            continue

        c1, c2 = pair_map[base]

        s1 = pd.to_numeric(
            df[c1],
            errors="coerce",
        )
        s2 = pd.to_numeric(
            df[c2],
            errors="coerce",
        )

        delta_col = f"DELTA_{base}"

        df[delta_col] = s2 - s1

        delta_cols.append(delta_col)

    if delta_cols:
        delta_block = df[delta_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        df["T12_MEAN_ABS_DELTA"] = (
            delta_block.abs().mean(axis=1)
        )

        df["T12_MAX_ABS_DELTA"] = (
            delta_block.abs().max(axis=1)
        )

    if drop_raw_t2:
        raw_t2_cols = sorted(
            {
                c2
                for _, (_, c2) in pair_map.items()
                if c2 in df.columns
            }
        )

        df = df.drop(
            columns=raw_t2_cols,
            errors="ignore",
        )

    return df
