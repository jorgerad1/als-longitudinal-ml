import numpy as np

from src.evaluation import (
    SPLIT_SEEDS,
    CV_SEEDS,
    TEST_SIZE,
    N_SPLITS,
    compute_metrics,
)


def test_fixed_seeds():
    assert SPLIT_SEEDS == [
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

    assert CV_SEEDS == SPLIT_SEEDS


def test_evaluation_design():
    assert TEST_SIZE == 0.25
    assert N_SPLITS == 5

    # Ten seeds times five folds.
    assert (
        len(CV_SEEDS)
        * N_SPLITS
        == 50
    )


def test_binary_metrics_and_confusion_matrix():

    y_true = np.array([
        0, 0, 0,
        1, 1, 1,
    ])

    y_pred = np.array([
        0, 1, 0,
        1, 0, 1,
    ])

    y_score = np.array([
        0.1,
        0.7,
        0.2,
        0.8,
        0.4,
        0.9,
    ])

    metrics, cm = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        ela_label=1,
        labels_order=[0, 1],
    )

    assert cm.tolist() == [
        [2, 1],
        [1, 2],
    ]

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 2

    assert np.isclose(
        metrics["accuracy"],
        4 / 6,
    )

    assert np.isclose(
        metrics["recall_ela"],
        2 / 3,
    )


def test_no_scores_returns_nan_auc():

    y_true = np.array([
        0, 0, 1, 1,
    ])

    y_pred = np.array([
        0, 1, 1, 1,
    ])

    metrics, _ = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        ela_label=1,
        labels_order=[0, 1],
    )

    assert np.isnan(
        metrics["roc_auc"]
    )

    assert np.isnan(
        metrics["pr_auc"]
    )
