"""Offline end-to-end demonstration using fully synthetic data.

This is a software smoke test, not an attempt to reproduce the
published clinical results.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.ablation import detect_ablation_blocks
from src.pool_construction import (
    build_pool1,
    build_pool2,
    build_pool3,
    make_model_matrix,
)
from src.preprocessing import (
    build_base_pipeline,
)


DATA_PATH = (
    ROOT
    / "synthetic_data"
    / "synthetic_questionnaire.xlsx"
)


BASELINE_COLUMNS = [
    "ID",
    "SEXO",
    "EDAD_CAT",
    "FAMILIA_NEURO",
    "TRABAJO_BASE",
]


T1_COLUMNS = [
    "TIPOCASA1",
    "ESTUDIOS1",
    "SOCIOECO1",
    "TIPOTRABAJO1",
    "VIVESOLO1",
    "DEPORTE1",
    "DEPORTEFREC1",
    "PESO1",
    "ALTURA1",
    "FORMA1",
    "TABACO1",
    "ALCOHOL1",
    "HORASSUEÑO1",
    "PREOCUPAC1",
    "PERSONALIDAD1",
    "SOCIALIZA1",
    "ALIMENTACION1",
    "BEBIDASAZUCAR1",
    "DULCES1",
    "FASTFOOD1",
    "COMIDAS_DIA1",
    "CANT_COMIDA1",
    "SACIEDAD1",
]


T2_COLUMNS = [
    col[:-1] + "2"
    for col in T1_COLUMNS
]


TEXT_COLUMNS = [
    "TXT_NATACION",
    "TXT_CAMINAR",
    "TXT_GIMNASIO",
    "TXT_CICLISMO",
    "TXT_YOGA",
    "TXT_TENIS",
]


def build_demo_pools(df):
    """Construct the three synthetic pool representations."""

    pool1 = build_pool1(
        df[
            BASELINE_COLUMNS
        ].copy()
    )

    pool2_structured = (
        BASELINE_COLUMNS
        + T1_COLUMNS
    )

    pool2_input = df[
        pool2_structured
        + TEXT_COLUMNS
    ].copy()

    pool2 = build_pool2(
        pool2_input,
        starting_cols=pool2_structured,
    )

    pool3_structured = (
        BASELINE_COLUMNS
        + T1_COLUMNS
        + T2_COLUMNS
    )

    pool3_input = df[
        pool3_structured
        + TEXT_COLUMNS
    ].copy()

    pool3 = build_pool3(
        pool3_input,
        starting_cols=pool3_structured,
    )

    return {
        "Pool1": pool1,
        "Pool2": pool2,
        "Pool3": pool3,
    }


def fit_smoke_model(
    X,
    encoded_y,
):
    """Fit one simple leakage-free pipeline for execution testing."""

    indices = list(
        range(len(X))
    )

    (
        train_idx,
        test_idx,
    ) = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=encoded_y,
    )

    X_train = (
        X.iloc[train_idx]
        .copy()
    )

    X_test = (
        X.iloc[test_idx]
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

    estimator = (
        LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            C=0.1,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42,
        )
    )

    pipeline = (
        build_base_pipeline(
            estimator
        )
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = (
        pipeline.predict(
            X_test
        )
    )

    return {
        "n_train":
            len(train_idx),

        "n_test":
            len(test_idx),

        "accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),
    }


def main():

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="synthetic",
    )

    encoder = LabelEncoder()

    encoded_y = (
        encoder.fit_transform(
            df[
                "DIAGNOSTICO"
            ]
        )
    )

    pools = build_demo_pools(
        df
    )

    print(
        "Synthetic demonstration only"
    )

    print(
        "Classes:",
        list(
            encoder.classes_
        ),
    )

    print()

    for (
        pool_name,
        pool_df,
    ) in pools.items():

        X = make_model_matrix(
            pool_df
        )

        result = fit_smoke_model(
            X,
            encoded_y,
        )

        print(
            f"{pool_name}: "
            f"{X.shape[0]} rows x "
            f"{X.shape[1]} predictors"
        )

        print(
            "  smoke-test train/test: "
            f"{result['n_train']}/"
            f"{result['n_test']}"
        )

        print(
            "  synthetic accuracy "
            "(not scientifically meaningful): "
            f"{result['accuracy']:.3f}"
        )

        if pool_name == "Pool3":
            (
                text_block,
                temporal_block,
            ) = detect_ablation_blocks(
                X
            )

            print(
                "  compact text block:",
                len(text_block),
                "features",
            )

            print(
                "  temporal block:",
                len(temporal_block),
                "features",
            )

        print()

    print(
        "OK - offline synthetic "
        "end-to-end demonstration completed."
    )


if __name__ == "__main__":
    main()
