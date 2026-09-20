import numpy as np
import pandas as pd

from src.preprocessing import (
    BasicSanitizer,
    MissingnessFilter,
    SmartTypePreprocessor,
    LowSupportFilter,
    CorrelationFilter,
)


def test_missingness_threshold():
    df = pd.DataFrame({
        "keep": [1, 2, None, 4, 5, 6],
        "drop": [1, None, None, 4, 5, 6],
    })

    clean = BasicSanitizer().fit_transform(df)

    filt = MissingnessFilter(
        threshold=0.20
    )
    filt.fit(clean)

    assert "keep" in filt.kept_columns_
    assert "drop" in filt.dropped_columns_


def test_forced_numeric_features():
    df = pd.DataFrame({
        "DELTA_PESO": [1, 2, 3, 4],
        "T12_N_CHANGED_ALL": [0, 1, 2, 3],
        "TXT_HABIT_COUNT": [0, 1, 2, 1],
        "TXTG_N_GROUPS": [0, 1, 2, 3],
    })

    prep = SmartTypePreprocessor()
    prep.fit(df)

    assert set(df.columns).issubset(
        set(prep.numeric_cols_)
    )


def test_low_support_filter():
    X = np.array([
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ], dtype=float)

    filt = LowSupportFilter(
        min_nonzero_count=3
    )
    filt.fit(X)

    assert 0 in filt.drop_idx_
    assert 1 in filt.keep_idx_


def test_correlation_filter():
    X = np.array([
        [1, 1, 6],
        [2, 2, 1],
        [3, 3, 5],
        [4, 4, 2],
        [5, 5, 4],
        [6, 6, 3],
    ], dtype=float)

    filt = CorrelationFilter(
        threshold=0.90
    )
    filt.fit(X)

    assert 1 in filt.drop_idx_
