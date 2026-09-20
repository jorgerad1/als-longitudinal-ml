"""Reproduce Figure 6 from published aggregate feature counts.

No participant-level data are required.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "feature_counts_by_stage.csv"
)

OUTPUT_PATH = (
    ROOT
    / "figures"
    / "Figure_6.pdf"
)


def main():

    df = pd.read_csv(
        INPUT_PATH
    )

    stages = [
        "Stage 1 Leakage-free pipeline",
        "Stage 2 Text cleaning + row realignment",
        "Stage 3 Compact text + \ncompact T1→T2 change",
    ]

    pool1 = (
        df["pool1"]
        .astype(float)
        .tolist()
    )

    pool2 = (
        df["pool2"]
        .astype(float)
        .tolist()
    )

    pool3 = (
        df["pool3"]
        .astype(float)
        .tolist()
    )

    reference_series = (
        df[
            "pool3_precompaction_reference"
        ]
        .dropna()
    )

    if reference_series.empty:
        raise ValueError(
            "Pool3 pre-compaction reference "
            "was not found."
        )

    pool3_reference_stage3 = (
        float(
            reference_series.iloc[0]
        )
    )

    x = np.arange(
        len(stages)
    )

    width = 0.14
    offset = 0.20

    fig, ax = plt.subplots(
        figsize=(12, 5.2)
    )

    bars1 = ax.bar(
        x - offset,
        pool1,
        width,
        label="Pool1",
    )

    bars2 = ax.bar(
        x,
        pool2,
        width,
        label="Pool2",
    )

    bars3 = ax.bar(
        x + offset,
        pool3,
        width,
        label="Pool3",
    )

    ref_x = (
        x[2]
        + offset
        + 0.09
    )

    ax.bar(
        ref_x,
        pool3_reference_stage3,
        width,
        facecolor="none",
        edgecolor="green",
        hatch="///",
        linewidth=0.8,
        label=(
            "Pool3 reference\n"
            "(pre-compaction)"
        ),
        zorder=0,
    )

    def add_labels(
        bars,
        y_shift=3,
    ):
        for bar in bars:
            height = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                height + y_shift,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.text(
        ref_x,
        pool3_reference_stage3
        + 3,
        f"{pool3_reference_stage3:.1f}",
        ha="center",
        va="bottom",
        fontsize=13,
        color="dimgray",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        stages,
        fontsize=13,
    )

    ax.set_ylabel(
        "Mean number of retained features",
        fontsize=13,
    )

    ax.set_xlabel(
        "Optimization stage",
        fontsize=13,
    )

    ax.legend(
        frameon=False,
        fontsize=12,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ymax = max(
        pool1
        + pool2
        + pool3
        + [
            pool3_reference_stage3
        ]
    ) * 1.1

    ax.set_ylim(
        0,
        ymax,
    )

    plt.tight_layout()

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
