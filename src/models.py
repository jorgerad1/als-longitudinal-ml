"""Model definitions and hyperparameter tuning for the final analysis.

The search spaces and random-state settings in this module reproduce
the final Pool1/Pool2/Pool3 model-selection workflow.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.svm import LinearSVC

from .preprocessing import build_base_pipeline


INNER_CV_SPLITS = 3
TUNING_SCORING = "f1_macro"
MODEL_RANDOM_STATE = 42
RF_N_ITER = 12


LOGISTIC_REGRESSION_PARAM_GRID = {
    "model__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    ],
    "model__l1_ratio": [
        0.5,
        0.8,
        1.0,
    ],
    "model__class_weight": [
        None,
        "balanced",
    ],
}


LINEAR_SVC_PARAM_GRID = {
    "model__C": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    ],
    "model__class_weight": [
        None,
        "balanced",
    ],
}


RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [
        100,
        200,
        300,
    ],
    "model__max_depth": [
        None,
        3,
        5,
        8,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
        6,
    ],
    "model__max_features": [
        "sqrt",
        0.2,
        0.5,
    ],
    "model__class_weight": [
        None,
        "balanced",
        "balanced_subsample",
    ],
}


def make_search(model_name, seed):
    """Build the nested hyperparameter search used in the final study.

    Parameters
    ----------
    model_name : str
        One of "LogisticRegression", "SVC", or "RandomForest".
    seed : int
        Seed used for the shuffled inner StratifiedKFold and, for
        Random Forest, RandomizedSearchCV parameter sampling.

    Returns
    -------
    GridSearchCV or RandomizedSearchCV
        Search object containing the complete leakage-free pipeline.
    """
    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    if model_name == "LogisticRegression":
        pipe = build_base_pipeline(
            LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=5000,
                random_state=MODEL_RANDOM_STATE,
            )
        )

        return GridSearchCV(
            estimator=pipe,
            param_grid=LOGISTIC_REGRESSION_PARAM_GRID,
            scoring=TUNING_SCORING,
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )

    if model_name == "SVC":
        pipe = build_base_pipeline(
            LinearSVC(
                dual="auto",
                random_state=MODEL_RANDOM_STATE,
                max_iter=5000,
            )
        )

        return GridSearchCV(
            estimator=pipe,
            param_grid=LINEAR_SVC_PARAM_GRID,
            scoring=TUNING_SCORING,
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )

    if model_name == "RandomForest":
        pipe = build_base_pipeline(
            RandomForestClassifier(
                random_state=MODEL_RANDOM_STATE,
                n_jobs=-1,
            )
        )

        return RandomizedSearchCV(
            estimator=pipe,
            param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS,
            n_iter=RF_N_ITER,
            scoring=TUNING_SCORING,
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            random_state=seed,
            error_score="raise",
        )

    raise ValueError(
        f"Unknown model_name: {model_name}"
    )
