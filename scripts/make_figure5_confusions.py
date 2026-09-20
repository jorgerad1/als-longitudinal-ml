"""Reproduce Figure 5 from published aggregate confusion matrices.

This script uses aggregate counts only. No participant-level data are
required.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "published_confusion_matrices.json"
)

OUTPUT_PATH = (
    ROOT
    / "figures"
    / "Figure_5.pdf"
)


def main():

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    matrices = {
        "A. Pool1 baseline\nRandom Forest":
            np.asarray(
                data[
                    "pool1_baseline_random_forest"
                ]
            ),

        "B. Pool2 full\nLogistic Regression":
            np.asarray(
                data[
                    "pool2_full_logistic_regression"
                ]
            ),

        "C. Pool3 full\nRandom Forest":
            np.asarray(
                data[
                    "pool3_full_random_forest"
                ]
            ),

        "D. Pool3 without temporal block\nRandom Forest":
            np.asarray(
                data[
                    "pool3_without_temporal_block_random_forest"
                ]
            ),
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    class_labels = [
        "Control",
        "ALS",
    ]

    vmax = max(
        matrix.max()
        for matrix
        in matrices.values()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    axes = axes.flatten()

    for (
        ax,
        (title, cm),
    ) in zip(
        axes,
        matrices.items(),
    ):

        im = ax.imshow(
            cm,
            interpolation="nearest",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(
            title,
            pad=10,
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            class_labels
        )

        ax.set_yticklabels(
            class_labels
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        threshold = (
            vmax / 2
        )

        for i in range(
            cm.shape[0]
        ):
            for j in range(
                cm.shape[1]
            ):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[i, j]
                        > threshold
                        else "black"
                    ),
                    fontweight="bold",
                )

    fig.subplots_adjust(
        right=0.88
    )

    cbar_ax = fig.add_axes(
        [
            0.82,
            0.24,
            0.02,
            0.45,
        ]
    )

    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
    )

    cbar.set_label(
        "Count"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            0.85,
            0.95,
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
