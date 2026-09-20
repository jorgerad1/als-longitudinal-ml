from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from src.models import (
    make_search,
    LOGISTIC_REGRESSION_PARAM_GRID,
    LINEAR_SVC_PARAM_GRID,
    RANDOM_FOREST_PARAM_DISTRIBUTIONS,
)


def test_search_types():
    lr = make_search(
        "LogisticRegression",
        42,
    )
    svc = make_search(
        "SVC",
        42,
    )
    rf = make_search(
        "RandomForest",
        42,
    )

    assert isinstance(lr, GridSearchCV)
    assert isinstance(svc, GridSearchCV)
    assert isinstance(
        rf,
        RandomizedSearchCV,
    )


def test_inner_cv_and_scoring():
    for model_name in [
        "LogisticRegression",
        "SVC",
        "RandomForest",
    ]:
        search = make_search(
            model_name,
            47,
        )

        assert search.scoring == "f1_macro"
        assert search.cv.n_splits == 3
        assert search.cv.shuffle is True
        assert search.cv.random_state == 47


def test_logistic_regression_configuration():
    search = make_search(
        "LogisticRegression",
        42,
    )

    model = (
        search.estimator
        .named_steps["model"]
    )

    assert model.solver == "saga"
    assert model.penalty == "elasticnet"
    assert model.max_iter == 5000
    assert model.random_state == 42

    assert (
        LOGISTIC_REGRESSION_PARAM_GRID[
            "model__C"
        ]
        == [0.001, 0.01, 0.1, 1.0, 10.0]
    )

    assert (
        LOGISTIC_REGRESSION_PARAM_GRID[
            "model__l1_ratio"
        ]
        == [0.5, 0.8, 1.0]
    )

    assert (
        LOGISTIC_REGRESSION_PARAM_GRID[
            "model__class_weight"
        ]
        == [None, "balanced"]
    )


def test_linear_svc_configuration():
    search = make_search(
        "SVC",
        42,
    )

    model = (
        search.estimator
        .named_steps["model"]
    )

    assert model.dual == "auto"
    assert model.max_iter == 5000
    assert model.random_state == 42

    assert (
        LINEAR_SVC_PARAM_GRID[
            "model__C"
        ]
        == [0.001, 0.01, 0.1, 1.0, 10.0]
    )


def test_random_forest_configuration():
    search = make_search(
        "RandomForest",
        51,
    )

    model = (
        search.estimator
        .named_steps["model"]
    )

    assert search.n_iter == 12
    assert search.random_state == 51

    assert model.random_state == 42
    assert model.n_jobs == -1

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__n_estimators"
        ]
        == [100, 200, 300]
    )

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__max_depth"
        ]
        == [None, 3, 5, 8]
    )

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__min_samples_split"
        ]
        == [2, 5, 10]
    )

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__min_samples_leaf"
        ]
        == [1, 2, 4, 6]
    )

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__max_features"
        ]
        == ["sqrt", 0.2, 0.5]
    )

    assert (
        RANDOM_FOREST_PARAM_DISTRIBUTIONS[
            "model__class_weight"
        ]
        == [
            None,
            "balanced",
            "balanced_subsample",
        ]
    )
