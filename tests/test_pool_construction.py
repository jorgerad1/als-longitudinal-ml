import pandas as pd

from src.pool_construction import (
    build_pool2,
    build_pool3,
    make_model_matrix,
)


def synthetic_input():
    df = pd.DataFrame({
        "ID": [1, 2],

        "FORMA_FEMENINA1":
            [1, None],

        "FORMA_MASCULINA1":
            [None, 2],

        "FORMA_FEMENINA2":
            [2, None],

        "FORMA_MASCULINA2":
            [None, 2],

        "PESO1":
            [70, 80],

        "PESO2":
            [72, 78],

        "HORASSUEÑO1":
            [7, 8],

        "HORASSUEÑO2":
            [6, 8],

        "TXT_NATACION":
            [1, 0],
    })

    starting_cols = [
        "ID",
        "FORMA_FEMENINA1",
        "FORMA_MASCULINA1",
        "FORMA_FEMENINA2",
        "FORMA_MASCULINA2",
        "PESO1",
        "PESO2",
        "HORASSUEÑO1",
        "HORASSUEÑO2",
    ]

    return df, starting_cols


def test_pool2_compaction():
    df, starting_cols = (
        synthetic_input()
    )

    out = build_pool2(
        df,
        starting_cols,
    )

    assert (
        "TXTG_ENDURANCE"
        in out.columns
    )

    assert (
        "TXT_NATACION"
        not in out.columns
    )


def test_pool3_temporal_representation():
    df, starting_cols = (
        synthetic_input()
    )

    out = build_pool3(
        df,
        starting_cols,
    )

    assert (
        "DELTA_PESO"
        in out.columns
    )

    assert (
        "T12_N_CHANGED_ALL"
        in out.columns
    )

    assert (
        "PESO2"
        not in out.columns
    )


def test_identifier_removed_before_modeling():
    df, starting_cols = (
        synthetic_input()
    )

    out = build_pool3(
        df,
        starting_cols,
    )

    X = make_model_matrix(out)

    assert "ID" not in X.columns
