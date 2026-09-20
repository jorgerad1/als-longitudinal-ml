"""Repeated holdout and cross-validation evaluation.

This module reproduces the final evaluation strategy used for the
Pool1, Pool2 and Pool3 analyses:

- 10 fixed seeds: 42..51
- stratified 75/25 holdout for each seed
- shuffled stratified 5-fold CV for each seed (50 outer folds)
- inner 3-fold stratified hyperparameter tuning
- train-only preprocessing through the estimator pipeline
- aggregate performance metrics and confusion matrices
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from .models import make_search


SPLIT_SEEDS = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
]

CV_SEEDS = SPLIT_SEEDS

TEST_SIZE = 0.25
N_SPLITS = 5


def _get_positive_scores(
    model,
    X,
    ela_label,
):
    """Return scores corresponding to the ALS positive class."""

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        ela_col = list(
            model.classes_
        ).index(ela_label)

        return proba[:, ela_col]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)

        if (
            scores.ndim == 1
            and len(
                getattr(
                    model,
                    "classes_",
                    [],
                )
            )
            == 2
        ):
            pos_class = model.classes_[1]

            return (
                scores
                if pos_class == ela_label
                else -scores
            )

        if (
            scores.ndim == 2
            and hasattr(
                model,
                "classes_",
            )
        ):
            ela_col = list(
                model.classes_
            ).index(ela_label)

            return scores[:, ela_col]

    return None


def compute_metrics(
    y_true,
    y_pred,
    y_score,
    ela_label,
    labels_order,
):
    """Compute the metrics recorded in the final analysis."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_order,
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "f1_ela": f1_score(
            y_true,
            y_pred,
            pos_label=ela_label,
        ),
        "precision_ela": precision_score(
            y_true,
            y_pred,
            pos_label=ela_label,
            zero_division=0,
        ),
        "recall_ela": recall_score(
            y_true,
            y_pred,
            pos_label=ela_label,
            zero_division=0,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        "mcc": matthews_corrcoef(
            y_true,
            y_pred,
        ),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp": tp,
    }

    if y_score is None:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    else:
        try:
            metrics["roc_auc"] = (
                roc_auc_score(
                    y_true,
                    y_score,
                )
            )
        except Exception:
            metrics["roc_auc"] = np.nan

        try:
            metrics["pr_auc"] = (
                average_precision_score(
                    y_true,
                    y_score,
                    pos_label=ela_label,
                )
            )
        except Exception:
            metrics["pr_auc"] = np.nan

    return metrics, cm


def summarise_results(df):
    """Calculate mean and sample SD across folds/splits."""

    metric_cols = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "f1_ela",
        "precision_ela",
        "recall_ela",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
    ]

    rows = []

    for metric in metric_cols:
        rows.append(
            {
                "metric": metric,
                "mean": df[metric].mean(),
                "std": df[metric].std(
                    ddof=1
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["mean±std"] = (
        summary.apply(
            lambda row:
                f'{row["mean"]:.3f}'
                f'±{row["std"]:.3f}',
            axis=1,
        )
    )

    return summary


def extract_feature_snapshot(
    best_estimator,
    original_n_features,
    model_name,
):
    """Record dimensionality after each preprocessing stage."""

    missing_filter = (
        best_estimator
        .named_steps["missing_filter"]
    )

    preprocessor = (
        best_estimator
        .named_steps["preprocess"]
    )

    variance_filter = (
        best_estimator
        .named_steps["variance_filter"]
    )

    low_support_filter = (
        best_estimator
        .named_steps["low_support_filter"]
    )

    corr_filter = (
        best_estimator
        .named_steps["corr_filter"]
    )

    model = (
        best_estimator
        .named_steps["model"]
    )

    n_after_missing = len(
        missing_filter.kept_columns_
    )

    pre_features = (
        preprocessor
        .get_feature_names_out()
    )

    n_after_preprocess = len(
        pre_features
    )

    variance_mask = (
        variance_filter.get_support()
    )

    variance_features = (
        pre_features[variance_mask]
    )

    n_after_variance = int(
        np.sum(variance_mask)
    )

    low_support_features = (
        low_support_filter
        .get_feature_names_out(
            variance_features
        )
    )

    n_after_low_support = len(
        low_support_features
    )

    corr_features = (
        corr_filter
        .get_feature_names_out(
            low_support_features
        )
    )

    n_after_correlation = len(
        corr_features
    )

    snapshot = {
        "n_original_raw":
            int(original_n_features),

        "n_after_missing":
            int(n_after_missing),

        "n_after_preprocess":
            int(n_after_preprocess),

        "n_after_variance":
            int(n_after_variance),

        "n_after_low_support":
            int(n_after_low_support),

        "n_after_correlation":
            int(n_after_correlation),

        "dropped_missing":
            int(
                original_n_features
                - n_after_missing
            ),

        "dropped_variance":
            int(
                n_after_preprocess
                - n_after_variance
            ),

        "dropped_low_support":
            int(
                n_after_variance
                - n_after_low_support
            ),

        "dropped_correlation":
            int(
                n_after_low_support
                - n_after_correlation
            ),

        "n_binary":
            int(
                len(
                    getattr(
                        preprocessor,
                        "binary_cols_",
                        [],
                    )
                )
            ),

        "n_numeric":
            int(
                len(
                    getattr(
                        preprocessor,
                        "numeric_cols_",
                        [],
                    )
                )
            ),

        "n_categorical":
            int(
                len(
                    getattr(
                        preprocessor,
                        "categorical_cols_",
                        [],
                    )
                )
            ),
    }

    final_feature_names = np.array(
        corr_features,
        dtype=object,
    )

    if model_name == "LogisticRegression":
        coef = np.asarray(
            model.coef_
        )

        if coef.ndim == 2:
            coef = (
                np.abs(coef)
                .max(axis=0)
            )
        else:
            coef = np.abs(coef)

        nonzero_mask = coef > 1e-8

        n_model_selected = int(
            nonzero_mask.sum()
        )

        snapshot[
            "n_model_selected"
        ] = n_model_selected

        snapshot[
            "n_model_discarded"
        ] = int(
            len(nonzero_mask)
            - n_model_selected
        )

        snapshot[
            "selected_feature_names"
        ] = (
            final_feature_names[
                nonzero_mask
            ]
            .tolist()
        )

    else:
        snapshot[
            "n_model_selected"
        ] = np.nan

        snapshot[
            "n_model_discarded"
        ] = np.nan

        snapshot[
            "selected_feature_names"
        ] = None

    return snapshot


def evaluate_model(
    model_name,
    X_raw,
    encoded_y,
    ela_label,
    labels_order,
):
    """Run the complete repeated CV and holdout evaluation."""

    cv_rows = []
    holdout_rows = []

    cv_confusions = []
    holdout_confusions = []

    cv_feature_rows = []
    holdout_feature_rows = []

    cv_best_params = []
    holdout_best_params = []

    if len(X_raw) != len(encoded_y):
        raise ValueError(
            "X_raw and encoded_y length mismatch "
            "inside evaluate_model: "
            f"X_raw has {len(X_raw)} rows "
            f"and y has {len(encoded_y)} labels."
        )

    idx = np.arange(
        len(X_raw)
    )

    # ----------------------------------------------------------
    # Repeated shuffled stratified 5-fold CV:
    # 10 seeds x 5 folds = 50 outer test folds
    # ----------------------------------------------------------
    for seed in CV_SEEDS:

        outer_cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for (
            fold_id,
            (train_idx, test_idx),
        ) in enumerate(
            outer_cv.split(
                idx,
                encoded_y,
            ),
            start=1,
        ):

            X_train = (
                X_raw
                .iloc[train_idx]
                .copy()
            )

            X_test = (
                X_raw
                .iloc[test_idx]
                .copy()
            )

            y_train = (
                encoded_y[
                    train_idx
                ]
            )

            y_test = (
                encoded_y[
                    test_idx
                ]
            )

            search = make_search(
                model_name,
                seed,
            )

            search.fit(
                X_train,
                y_train,
            )

            best_estimator = (
                search.best_estimator_
            )

            y_pred = (
                best_estimator
                .predict(X_test)
            )

            y_score = (
                _get_positive_scores(
                    best_estimator,
                    X_test,
                    ela_label,
                )
            )

            metrics, cm = (
                compute_metrics(
                    y_test,
                    y_pred,
                    y_score,
                    ela_label,
                    labels_order,
                )
            )

            metrics.update(
                {
                    "seed": seed,
                    "fold": fold_id,
                }
            )

            cv_rows.append(
                metrics
            )

            cv_confusions.append(
                cm
            )

            snapshot = (
                extract_feature_snapshot(
                    best_estimator,
                    X_raw.shape[1],
                    model_name,
                )
            )

            snapshot.update(
                {
                    "seed": seed,
                    "fold": fold_id,
                }
            )

            cv_feature_rows.append(
                snapshot
            )

            params_row = {
                "seed": seed,
                "fold": fold_id,
            }

            params_row.update(
                search.best_params_
            )

            cv_best_params.append(
                params_row
            )

    # ----------------------------------------------------------
    # Repeated stratified 75/25 holdout:
    # one split for each of the same 10 seeds
    # ----------------------------------------------------------
    for seed in SPLIT_SEEDS:

        (
            train_idx,
            test_idx,
        ) = train_test_split(
            idx,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=encoded_y,
        )

        X_train = (
            X_raw
            .iloc[train_idx]
            .copy()
        )

        X_test = (
            X_raw
            .iloc[test_idx]
            .copy()
        )

        y_train = (
            encoded_y[
                train_idx
            ]
        )

        y_test = (
            encoded_y[
                test_idx
            ]
        )

        search = make_search(
            model_name,
            seed,
        )

        search.fit(
            X_train,
            y_train,
        )

        best_estimator = (
            search.best_estimator_
        )

        y_pred = (
            best_estimator
            .predict(X_test)
        )

        y_score = (
            _get_positive_scores(
                best_estimator,
                X_test,
                ela_label,
            )
        )

        metrics, cm = (
            compute_metrics(
                y_test,
                y_pred,
                y_score,
                ela_label,
                labels_order,
            )
        )

        metrics.update(
            {
                "seed": seed,
            }
        )

        holdout_rows.append(
            metrics
        )

        holdout_confusions.append(
            cm
        )

        snapshot = (
            extract_feature_snapshot(
                best_estimator,
                X_raw.shape[1],
                model_name,
            )
        )

        snapshot.update(
            {
                "seed": seed,
            }
        )

        holdout_feature_rows.append(
            snapshot
        )

        params_row = {
            "seed": seed,
        }

        params_row.update(
            search.best_params_
        )

        holdout_best_params.append(
            params_row
        )

    cv_df = pd.DataFrame(
        cv_rows
    )

    holdout_df = pd.DataFrame(
        holdout_rows
    )

    cv_summary = (
        summarise_results(
            cv_df
        )
    )

    holdout_summary = (
        summarise_results(
            holdout_df
        )
    )

    cv_confusion_sum = np.sum(
        cv_confusions,
        axis=0,
    )

    holdout_confusion_sum = np.sum(
        holdout_confusions,
        axis=0,
    )

    cv_confusion_mean = np.mean(
        cv_confusions,
        axis=0,
    )

    holdout_confusion_mean = np.mean(
        holdout_confusions,
        axis=0,
    )

    cv_features_df = pd.DataFrame(
        cv_feature_rows
    )

    holdout_features_df = pd.DataFrame(
        holdout_feature_rows
    )

    feature_exclude = [
        "selected_feature_names",
        "seed",
        "fold",
    ]

    cv_feature_numeric = (
        cv_features_df.drop(
            columns=[
                c
                for c in feature_exclude
                if c
                in cv_features_df.columns
            ]
        )
    )

    holdout_feature_numeric = (
        holdout_features_df.drop(
            columns=[
                c
                for c in feature_exclude
                if c
                in holdout_features_df.columns
            ]
        )
    )

    cv_feature_summary = (
        cv_feature_numeric
        .mean(
            numeric_only=True
        )
        .to_frame("mean")
        .join(
            cv_feature_numeric
            .std(
                ddof=1,
                numeric_only=True,
            )
            .to_frame("std")
        )
    )

    holdout_feature_summary = (
        holdout_feature_numeric
        .mean(
            numeric_only=True
        )
        .to_frame("mean")
        .join(
            holdout_feature_numeric
            .std(
                ddof=1,
                numeric_only=True,
            )
            .to_frame("std")
        )
    )

    cv_best_params_df = (
        pd.DataFrame(
            cv_best_params
        )
    )

    holdout_best_params_df = (
        pd.DataFrame(
            holdout_best_params
        )
    )

    return {
        "cv_df":
            cv_df,

        "holdout_df":
            holdout_df,

        "cv_summary":
            cv_summary,

        "holdout_summary":
            holdout_summary,

        "cv_confusion_sum":
            cv_confusion_sum,

        "holdout_confusion_sum":
            holdout_confusion_sum,

        "cv_confusion_mean":
            cv_confusion_mean,

        "holdout_confusion_mean":
            holdout_confusion_mean,

        "cv_features_df":
            cv_features_df,

        "holdout_features_df":
            holdout_features_df,

        "cv_feature_summary":
            cv_feature_summary,

        "holdout_feature_summary":
            holdout_feature_summary,

        "cv_best_params_df":
            cv_best_params_df,

        "holdout_best_params_df":
            holdout_best_params_df,
    }


def format_param_counts(
    params_df,
):
    """Summarize frequency of selected hyperparameters."""

    if params_df.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in params_df.columns
        if c not in {
            "seed",
            "fold",
        }
    ]

    blocks = []

    for col in cols:
        counts = (
            params_df[col]
            .astype(str)
            .value_counts()
            .rename_axis("value")
            .reset_index(name="count")
        )

        counts.insert(
            0,
            "parameter",
            col,
        )

        blocks.append(
            counts
        )

    if not blocks:
        return pd.DataFrame()

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def compact_holdout_table(
    results_dict,
):
    """Create the compact final holdout results table."""

    rows = []

    for (
        model_name,
        obj,
    ) in results_dict.items():

        summary = (
            obj["holdout_summary"]
            .set_index("metric")[
                "mean±std"
            ]
        )

        rows.append(
            {
                "model":
                    model_name,

                "accuracy":
                    summary[
                        "accuracy"
                    ],

                "f1_weighted":
                    summary[
                        "f1_weighted"
                    ],

                "f1_ela":
                    summary[
                        "f1_ela"
                    ],

                "recall_ela":
                    summary[
                        "recall_ela"
                    ],

                "balanced_accuracy":
                    summary[
                        "balanced_accuracy"
                    ],

                "mcc":
                    summary[
                        "mcc"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def compact_cv_table(
    results_dict,
):
    """Create the compact CV results table."""

    rows = []

    for (
        model_name,
        obj,
    ) in results_dict.items():

        summary = (
            obj["cv_summary"]
            .set_index("metric")[
                "mean±std"
            ]
        )

        rows.append(
            {
                "model":
                    model_name,

                "accuracy":
                    summary[
                        "accuracy"
                    ],

                "f1_weighted":
                    summary[
                        "f1_weighted"
                    ],

                "f1_ela":
                    summary[
                        "f1_ela"
                    ],

                "recall_ela":
                    summary[
                        "recall_ela"
                    ],

                "balanced_accuracy":
                    summary[
                        "balanced_accuracy"
                    ],

                "mcc":
                    summary[
                        "mcc"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )
