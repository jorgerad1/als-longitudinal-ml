"""Final compact Pool2 and Pool3 construction.

These functions reproduce the post-LLM feature-construction steps used
in the final analyses. They operate on already harmonized structured
data plus the aligned LLM-derived fields; they do not read the private
clinical workbook.
"""

import pandas as pd

from .text_features import (
    add_compact_text_groups,
    coalesce_columns,
    finalize_text_feature_block,
)
from .temporal_features import (
    add_temporal_summary_features,
)


def build_pool1(structured_baseline_df):
    """Return the structured baseline representation used as Pool1."""
    return structured_baseline_df.copy()


def build_pool2(
    df_updated,
    starting_cols,
):
    """Construct the final compact Pool2 representation.

    Steps:
    1. Canonicalize text-derived habit fields.
    2. Harmonize sex-specific FORMA1 fields.
    3. Collapse individual text habits into compact text descriptors.
    """
    df_complete = finalize_text_feature_block(
        df_updated,
        starting_cols,
    )

    df_complete = coalesce_columns(
        df_complete,
        "FORMA1",
        [
            "FORMA_FEMENINA1",
            "FORMA_MASCULINA1",
        ],
    )

    df_complete = add_compact_text_groups(
        df_complete,
        drop_individual_habits=True,
    )

    return df_complete


def build_pool3(
    df_updated,
    starting_cols,
):
    """Construct the final compact Pool3 representation.

    Pool3 extends the Pool2 transformation by harmonizing T2 variables,
    creating compact T1-to-T2 change descriptors, and removing paired
    raw T2 columns after descriptor construction.
    """
    df_complete = finalize_text_feature_block(
        df_updated,
        starting_cols,
    )

    df_complete = coalesce_columns(
        df_complete,
        "FORMA1",
        [
            "FORMA_FEMENINA1",
            "FORMA_MASCULINA1",
        ],
    )

    df_complete = coalesce_columns(
        df_complete,
        "FORMA2",
        [
            "FORMA_FEMENINA2",
            "FORMA_MASCULINA2",
        ],
    )

    if "TIPOPESO_2" in df_complete.columns:
        df_complete = df_complete.rename(
            columns={
                "TIPOPESO_2":
                    "TIPOPESO2"
            }
        )

    df_complete = add_compact_text_groups(
        df_complete,
        drop_individual_habits=True,
    )

    df_complete = add_temporal_summary_features(
        df_complete,
        drop_raw_t2=True,
    )

    return df_complete


def make_model_matrix(
    df,
    id_col="ID",
):
    """Remove the participant identifier before model development."""
    if id_col not in df.columns:
        return (
            df
            .reset_index(drop=True)
            .copy()
        )

    return (
        df
        .drop(columns=id_col)
        .reset_index(drop=True)
        .copy()
    )


def normalize_id_key(value):
    """Normalize identifiers for deterministic target alignment."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        if (
            isinstance(value, float)
            and float(value).is_integer()
        ):
            return str(int(value))
    except Exception:
        pass

    return str(value).strip()


def align_target_by_id(
    feature_df,
    source_df,
    target_column,
    id_col="ID",
):
    """Realign outcome labels to the final feature dataframe by ID."""

    if id_col not in feature_df.columns:
        raise ValueError(
            f"{id_col!r} is not present in feature_df."
        )

    if id_col not in source_df.columns:
        raise ValueError(
            f"{id_col!r} is not present in source_df."
        )

    if target_column not in source_df.columns:
        raise ValueError(
            f"{target_column!r} is not present in source_df."
        )

    target_lookup = (
        source_df[
            [
                id_col,
                target_column,
            ]
        ]
        .copy()
    )

    target_lookup["_id_key"] = (
        target_lookup[id_col]
        .map(normalize_id_key)
    )

    target_lookup = (
        target_lookup
        .dropna(
            subset=["_id_key"]
        )
        .drop_duplicates(
            subset="_id_key",
            keep="first",
        )
        .set_index("_id_key")[
            target_column
        ]
    )

    current_keys = (
        feature_df[id_col]
        .map(normalize_id_key)
    )

    y = current_keys.map(
        target_lookup
    )

    missing_targets = int(
        y.isna().sum()
    )

    if missing_targets > 0:
        raise ValueError(
            f"{missing_targets} feature rows "
            "could not be matched to the target by ID."
        )

    return (
        y.astype(str)
        .str.upper()
        .reset_index(drop=True)
    )
