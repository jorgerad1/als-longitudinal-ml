"""Leakage-free preprocessing used in the final ML pipeline.

All transformers in this module are fitted only on the training
partition when embedded inside the scikit-learn Pipeline.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


# Exact defaults used in the final published pipeline.
MISSING_THRESHOLD = 0.20
CATEGORICAL_MAX_UNIQUE = 15
NZV_THRESHOLD = 0.0
CORR_THRESHOLD = 0.90
OHE_MIN_FREQUENCY = 3
MIN_ACTIVE_COUNT = 3

FORCED_NUMERIC_PREFIXES = ("DELTA_", "T12_")
FORCED_NUMERIC_COLUMNS = {
    "TXT_HABIT_COUNT",
    "TXTG_N_GROUPS",
}


def _ensure_dataframe(X, columns=None):
    """Return X as a defensive pandas DataFrame copy."""
    if isinstance(X, pd.DataFrame):
        return X.copy()

    X_arr = np.asarray(X)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if columns is None:
        columns = [
            f"x{i}"
            for i in range(X_arr.shape[1])
        ]

    return pd.DataFrame(
        X_arr,
        columns=columns,
    )


def _sanitize_value(value):
    """Normalize empty, boolean and string-coded values."""
    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return np.nan

        lower = value.lower()

        if lower == "true":
            return 1

        if lower == "false":
            return 0

        return lower

    return value


def round_categorical_matrix(X):
    """Round categorical numeric codes before one-hot encoding."""
    arr = np.asarray(
        X,
        dtype=float,
    )

    return np.rint(arr).astype(int)


def make_ohe(min_frequency=3):
    """Construct the OneHotEncoder used in the final pipeline."""
    kwargs = {
        "handle_unknown": "infrequent_if_exist",
        "min_frequency": min_frequency,
    }

    try:
        return OneHotEncoder(
            sparse_output=False,
            **kwargs,
        )
    except TypeError:
        # Compatibility with older scikit-learn releases.
        return OneHotEncoder(
            sparse=False,
            **kwargs,
        )


class BasicSanitizer(BaseEstimator, TransformerMixin):
    """Sanitize and coerce questionnaire predictors to numeric codes."""

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.columns_ = X_df.columns.tolist()
        return self

    def transform(self, X):
        X_df = _ensure_dataframe(
            X,
            getattr(self, "columns_", None),
        )

        X_df = X_df.apply(
            lambda col: col.map(_sanitize_value)
        )

        for col in X_df.columns:
            X_df[col] = pd.to_numeric(
                X_df[col],
                errors="coerce",
            )

        return X_df

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return np.array(
            getattr(
                self,
                "columns_",
                input_features,
            ),
            dtype=object,
        )


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Remove predictors with missingness strictly above threshold."""

    def __init__(self, threshold=0.20):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)

        missing_ratio = (
            X_df.isnull().mean()
        )

        self.missing_ratio_ = missing_ratio

        self.kept_columns_ = (
            missing_ratio[
                missing_ratio <= self.threshold
            ]
            .index
            .tolist()
        )

        self.dropped_columns_ = (
            missing_ratio[
                missing_ratio > self.threshold
            ]
            .index
            .tolist()
        )

        if len(self.kept_columns_) == 0:
            raise ValueError(
                "All columns were dropped by "
                "MissingnessFilter."
            )

        return self

    def transform(self, X):
        X_df = _ensure_dataframe(X)

        return X_df[
            self.kept_columns_
        ].copy()

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return np.array(
            self.kept_columns_,
            dtype=object,
        )


class SmartTypePreprocessor(
    BaseEstimator,
    TransformerMixin,
):
    """Apply type-aware imputation, scaling and encoding.

    Rules used in the final analysis:
    - <=2 unique values: binary
    - 3..15 unique values: categorical
    - >15 unique values: numeric
    - DELTA_* and T12_* are always numeric
    - TXT_HABIT_COUNT and TXTG_N_GROUPS are always numeric
    """

    def __init__(
        self,
        categorical_max_unique=15,
        ohe_min_frequency=3,
    ):
        self.categorical_max_unique = (
            categorical_max_unique
        )
        self.ohe_min_frequency = (
            ohe_min_frequency
        )

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)

        self.input_features_ = (
            X_df.columns.tolist()
        )

        nunique = X_df.nunique(
            dropna=True
        )

        self.binary_cols_ = []
        self.categorical_cols_ = []
        self.numeric_cols_ = []

        for col in X_df.columns:
            n_unique = int(
                nunique.get(col, 0)
            )

            forced_numeric = (
                col in FORCED_NUMERIC_COLUMNS
                or any(
                    str(col).startswith(prefix)
                    for prefix
                    in FORCED_NUMERIC_PREFIXES
                )
            )

            if forced_numeric:
                self.numeric_cols_.append(col)
                continue

            if n_unique <= 1:
                # Retained here so zero-variance/low-support
                # filters can remove it downstream.
                self.binary_cols_.append(col)

            elif n_unique <= 2:
                self.binary_cols_.append(col)

            elif n_unique <= (
                self.categorical_max_unique
            ):
                self.categorical_cols_.append(col)

            else:
                self.numeric_cols_.append(col)

        transformers = []

        if self.binary_cols_:
            bin_pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "bin",
                    bin_pipe,
                    self.binary_cols_,
                )
            )

        if self.numeric_cols_:
            num_pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            transformers.append(
                (
                    "num",
                    num_pipe,
                    self.numeric_cols_,
                )
            )

        if self.categorical_cols_:
            cat_pipe = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "rounder",
                        FunctionTransformer(
                            round_categorical_matrix,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    (
                        "ohe",
                        make_ohe(
                            min_frequency=(
                                self.ohe_min_frequency
                            )
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "cat",
                    cat_pipe,
                    self.categorical_cols_,
                )
            )

        if not transformers:
            raise ValueError(
                "No columns available for preprocessing."
            )

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0,
        )

        self.transformer_.fit(X_df)

        feature_names = []

        if self.binary_cols_:
            feature_names.extend(
                self.binary_cols_
            )

        if self.numeric_cols_:
            feature_names.extend(
                self.numeric_cols_
            )

        if self.categorical_cols_:
            ohe = (
                self.transformer_
                .named_transformers_["cat"]
                .named_steps["ohe"]
            )

            feature_names.extend(
                ohe.get_feature_names_out(
                    self.categorical_cols_
                )
            )

        self.feature_names_out_ = (
            np.array(
                feature_names,
                dtype=object,
            )
        )

        return self

    def transform(self, X):
        X_df = _ensure_dataframe(
            X,
            getattr(
                self,
                "input_features_",
                None,
            ),
        )

        return self.transformer_.transform(
            X_df
        )

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        return self.feature_names_out_


class LowSupportFilter(
    BaseEstimator,
    TransformerMixin,
):
    """Retain features with at least N non-zero observations."""

    def __init__(
        self,
        min_nonzero_count=3,
        eps=1e-12,
    ):
        self.min_nonzero_count = (
            min_nonzero_count
        )
        self.eps = eps

    def fit(self, X, y=None):
        X_arr = np.asarray(
            X,
            dtype=float,
        )

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(
                -1,
                1,
            )

        nonzero_count = np.sum(
            np.abs(X_arr) > self.eps,
            axis=0,
        )

        self.nonzero_count_ = (
            nonzero_count
        )

        self.keep_idx_ = np.where(
            nonzero_count
            >= self.min_nonzero_count
        )[0]

        self.drop_idx_ = np.where(
            nonzero_count
            < self.min_nonzero_count
        )[0]

        if self.keep_idx_.size == 0:
            self.keep_idx_ = np.array(
                [
                    int(
                        np.argmax(
                            nonzero_count
                        )
                    )
                ],
                dtype=int,
            )

            self.drop_idx_ = np.array(
                [
                    i
                    for i
                    in range(X_arr.shape[1])
                    if i not in self.keep_idx_
                ],
                dtype=int,
            )

        return self

    def transform(self, X):
        X_arr = np.asarray(X)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(
                -1,
                1,
            )

        return X_arr[
            :,
            self.keep_idx_,
        ]

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        if input_features is None:
            total = int(
                self.keep_idx_.size
                + self.drop_idx_.size
            )

            input_features = np.array(
                [
                    f"x{i}"
                    for i in range(total)
                ],
                dtype=object,
            )

        input_features = np.asarray(
            input_features,
            dtype=object,
        )

        return input_features[
            self.keep_idx_
        ]


class CorrelationFilter(
    BaseEstimator,
    TransformerMixin,
):
    """Remove later features correlated above the absolute threshold."""

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_arr = np.asarray(
            X,
            dtype=float,
        )

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(
                -1,
                1,
            )

        n_features = X_arr.shape[1]

        if n_features == 0:
            raise ValueError(
                "CorrelationFilter received zero features."
            )

        if n_features == 1:
            self.keep_idx_ = np.array(
                [0],
                dtype=int,
            )
            self.drop_idx_ = np.array(
                [],
                dtype=int,
            )
            return self

        corr = (
            pd.DataFrame(X_arr)
            .corr()
            .abs()
            .fillna(0.0)
        )

        upper = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1,
            ).astype(bool)
        )

        to_drop = [
            int(col)
            for col in upper.columns
            if (
                upper[col]
                > self.threshold
            ).any()
        ]

        drop_set = set(to_drop)

        self.drop_idx_ = np.array(
            sorted(drop_set),
            dtype=int,
        )

        self.keep_idx_ = np.array(
            [
                i
                for i
                in range(n_features)
                if i not in drop_set
            ],
            dtype=int,
        )

        if len(self.keep_idx_) == 0:
            self.keep_idx_ = np.array(
                [0],
                dtype=int,
            )

            self.drop_idx_ = np.array(
                [
                    i
                    for i
                    in range(
                        1,
                        n_features,
                    )
                ],
                dtype=int,
            )

        return self

    def transform(self, X):
        X_arr = np.asarray(X)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(
                -1,
                1,
            )

        return X_arr[
            :,
            self.keep_idx_,
        ]

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        if input_features is None:
            total = int(
                self.keep_idx_.size
                + self.drop_idx_.size
            )

            input_features = np.array(
                [
                    f"x{i}"
                    for i in range(total)
                ],
                dtype=object,
            )

        input_features = np.asarray(
            input_features,
            dtype=object,
        )

        return input_features[
            self.keep_idx_
        ]


def build_base_pipeline(
    model,
    missing_threshold=MISSING_THRESHOLD,
    categorical_max_unique=CATEGORICAL_MAX_UNIQUE,
    ohe_min_frequency=OHE_MIN_FREQUENCY,
    variance_threshold=NZV_THRESHOLD,
    min_active_count=MIN_ACTIVE_COUNT,
    correlation_threshold=CORR_THRESHOLD,
):
    """Build the complete train-only preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            (
                "sanitize",
                BasicSanitizer(),
            ),
            (
                "missing_filter",
                MissingnessFilter(
                    threshold=missing_threshold
                ),
            ),
            (
                "preprocess",
                SmartTypePreprocessor(
                    categorical_max_unique=(
                        categorical_max_unique
                    ),
                    ohe_min_frequency=(
                        ohe_min_frequency
                    ),
                ),
            ),
            (
                "variance_filter",
                VarianceThreshold(
                    threshold=variance_threshold
                ),
            ),
            (
                "low_support_filter",
                LowSupportFilter(
                    min_nonzero_count=(
                        min_active_count
                    )
                ),
            ),
            (
                "corr_filter",
                CorrelationFilter(
                    threshold=(
                        correlation_threshold
                    )
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )
