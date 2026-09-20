import pandas as pd

from src.ablation import (
    detect_ablation_blocks,
    make_pool3_ablation_designs,
)


def test_ablation_block_detection():

    X = pd.DataFrame({
        "BASE":
            [0, 1],

        "TXTG_ENDURANCE":
            [1, 0],

        "TXT_ANY_HABIT":
            [1, 0],

        "TXT_HABIT_COUNT":
            [2, 0],

        "DELTA_PESO":
            [2, -1],

        "T12_N_CHANGED_ALL":
            [3, 1],
    })

    (
        text_block,
        temporal_block,
    ) = detect_ablation_blocks(X)

    assert (
        "TXTG_ENDURANCE"
        in text_block
    )

    assert (
        "TXT_ANY_HABIT"
        in text_block
    )

    assert (
        "DELTA_PESO"
        in temporal_block
    )

    assert (
        "T12_N_CHANGED_ALL"
        in temporal_block
    )


def test_pool3_ablation_designs():

    X = pd.DataFrame({
        "BASE":
            [0, 1],

        "TXTG_ENDURANCE":
            [1, 0],

        "DELTA_PESO":
            [2, -1],
    })

    designs = (
        make_pool3_ablation_designs(
            X
        )
    )

    assert set(designs) == {
        "full_pool3",
        "pool3_without_text_block",
        "pool3_without_temporal_block",
        "pool3_without_text_and_temporal",
    }

    assert (
        "TXTG_ENDURANCE"
        not in designs[
            "pool3_without_text_block"
        ].columns
    )

    assert (
        "DELTA_PESO"
        not in designs[
            "pool3_without_temporal_block"
        ].columns
    )

    assert (
        list(
            designs[
                "pool3_without_text_and_temporal"
            ].columns
        )
        == ["BASE"]
    )
