"""Ablation analysis for the final Pool2 and Pool3 configurations."""

import pandas as pd

from .evaluation import evaluate_model


ABLATION_CORE_METRICS = [
    "accuracy",
    "f1_weighted",
    "f1_ela",
    "recall_ela",
    "balanced_accuracy",
    "mcc",
]

ABLATION_DELTA_METRICS = (
    ABLATION_CORE_METRICS
    + [
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
    ]
)


def detect_ablation_blocks(X_df):
    """Identify the explicit compact text and temporal blocks."""

    text_block = sorted(
        [
            c
            for c in X_df.columns
            if (
                c.startswith("TXTG_")
                or c
                in {
                    "TXT_ANY_HABIT",
                    "TXT_HABIT_COUNT",
                }
            )
        ]
    )

    temporal_block = sorted(
        [
            c
            for c in X_df.columns
            if (
                c.startswith("DELTA_")
                or c.startswith("T12_")
            )
        ]
    )

    return (
        text_block,
        temporal_block,
    )


def make_pool2_ablation_designs(X_df):
    """Return feature matrices used in the Pool2 ablation."""

    text_block, _ = (
        detect_ablation_blocks(
            X_df
        )
    )

    designs = {
        "full_pool2":
            X_df.copy(),
    }

    if text_block:
        designs[
            "pool2_without_text_block"
        ] = (
            X_df
            .drop(
                columns=text_block,
                errors="ignore",
            )
            .copy()
        )

    return designs


def make_pool3_ablation_designs(X_df):
    """Return the four final Pool3 ablation feature matrices."""

    (
        text_block,
        temporal_block,
    ) = detect_ablation_blocks(
        X_df
    )

    designs = {
        "full_pool3":
            X_df.copy(),
    }

    if text_block:
        designs[
            "pool3_without_text_block"
        ] = (
            X_df
            .drop(
                columns=text_block,
                errors="ignore",
            )
            .copy()
        )

    if temporal_block:
        designs[
            "pool3_without_temporal_block"
        ] = (
            X_df
            .drop(
                columns=temporal_block,
                errors="ignore",
            )
            .copy()
        )

    if (
        text_block
        or temporal_block
    ):
        drop_both = sorted(
            set(
                text_block
                + temporal_block
            )
        )

        designs[
            "pool3_without_text_and_temporal"
        ] = (
            X_df
            .drop(
                columns=drop_both,
                errors="ignore",
            )
            .copy()
        )

    return designs


def run_pool2_ablation(
    X_raw,
    encoded_y,
    ela_label,
    labels_order,
    full_result=None,
):
    """Run final Pool2 ablation using Logistic Regression."""

    designs = (
        make_pool2_ablation_designs(
            X_raw
        )
    )

    results = {}

    for (
        design_name,
        X_design,
    ) in designs.items():

        if (
            design_name == "full_pool2"
            and full_result is not None
        ):
            results[
                design_name
            ] = full_result

        else:
            results[
                design_name
            ] = evaluate_model(
                "LogisticRegression",
                X_design,
                encoded_y,
                ela_label,
                labels_order,
            )

    return results


def run_pool3_ablation(
    X_raw,
    encoded_y,
    ela_label,
    labels_order,
    full_result=None,
):
    """Run final Pool3 ablation using Random Forest."""

    designs = (
        make_pool3_ablation_designs(
            X_raw
        )
    )

    results = {}

    for (
        design_name,
        X_design,
    ) in designs.items():

        if (
            design_name == "full_pool3"
            and full_result is not None
        ):
            results[
                design_name
            ] = full_result

        else:
            results[
                design_name
            ] = evaluate_model(
                "RandomForest",
                X_design,
                encoded_y,
                ela_label,
                labels_order,
            )

    return results


def ablation_compact_table(
    results_dict,
    split="holdout",
):
    """Create compact mean±SD ablation table."""

    rows = []

    for (
        design_name,
        obj,
    ) in results_dict.items():

        summary = (
            obj[
                f"{split}_summary"
            ]
            .set_index("metric")[
                "mean±std"
            ]
        )

        row = {
            "design":
                design_name
        }

        for metric in (
            ABLATION_CORE_METRICS
        ):
            if metric in summary.index:
                row[metric] = (
                    summary[metric]
                )

        rows.append(row)

    return pd.DataFrame(rows)


def ablation_delta_table(
    results_dict,
    reference_name,
    split="holdout",
    decimals=3,
):
    """Calculate mean metric differences versus the full design."""

    ref_df = (
        results_dict[
            reference_name
        ][f"{split}_df"]
    )

    rows = []

    for (
        design_name,
        obj,
    ) in results_dict.items():

        if (
            design_name
            == reference_name
        ):
            continue

        cur_df = obj[
            f"{split}_df"
        ]

        row = {
            "comparison":
                f"{design_name} - "
                f"{reference_name}"
        }

        for metric in (
            ABLATION_DELTA_METRICS
        ):
            if (
                metric in cur_df.columns
                and metric
                in ref_df.columns
            ):
                row[metric] = round(
                    float(
                        cur_df[
                            metric
                        ].mean()
                        - ref_df[
                            metric
                        ].mean()
                    ),
                    decimals,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def ablation_confusion_table(
    results_dict,
):
    """Summarize aggregated confusion matrices."""

    return pd.DataFrame(
        {
            "design":
                list(
                    results_dict.keys()
                ),

            "holdout_confusion_sum":
                [
                    str(
                        obj[
                            "holdout_confusion_sum"
                        ].tolist()
                    )
                    for obj
                    in results_dict.values()
                ],

            "cv_confusion_sum":
                [
                    str(
                        obj[
                            "cv_confusion_sum"
                        ].tolist()
                    )
                    for obj
                    in results_dict.values()
                ],
        }
    )
