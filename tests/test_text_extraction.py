import pandas as pd

from src.text_extraction import (
    align_chunk_output_to_input,
    generate_chunk_prompt,
)


def test_prompt_contains_required_rules():

    prompt = generate_chunk_prompt(
        '{"pacientes": []}',
        {"TXT_NATACION"},
    )

    assert (
        "No modifiques los valores "
        "ya presentes"
        in prompt
    )

    assert (
        "pacientes_actualizados"
        in prompt
    )

    assert (
        "habitos_comunes_nuevos"
        in prompt
    )

    assert (
        "NATACION"
        in prompt
    )


def test_row_realign_and_preserve_original_values():

    original = pd.DataFrame({
        "ID": [101, 102, 103],
        "DEPORTE1": [
            1,
            None,
            0,
        ],
        "PESO1": [
            70,
            80,
            90,
        ],
    })

    llm_output = {
        "pacientes_actualizados": [
            {
                "datos": {
                    "ID": 101,
                    "DEPORTE1": 0,
                    "PESO1": 999,
                    "NATACION": 1,
                }
            },
            {
                "datos": {
                    "ID": 102,
                    "DEPORTE1": 1,
                    "PESO1": 80,
                }
            },
        ],
        "habitos_comunes_nuevos": [
            "NATACION"
        ],
    }

    rows = (
        align_chunk_output_to_input(
            original,
            llm_output,
            starting_cols=list(
                original.columns
            ),
            id_col="ID",
        )
    )

    out = pd.DataFrame(rows)

    # Existing structured values are preserved.
    assert (
        out.loc[
            out["ID"] == 101,
            "DEPORTE1",
        ].iloc[0]
        == 1
    )

    assert (
        out.loc[
            out["ID"] == 101,
            "PESO1",
        ].iloc[0]
        == 70
    )

    # Missing structured value may be completed.
    assert (
        out.loc[
            out["ID"] == 102,
            "DEPORTE1",
        ].iloc[0]
        == 1
    )

    # Omitted LLM record is reconstructed.
    assert len(out) == 3
    assert (
        103
        in out["ID"].tolist()
    )

    # Derived habit survives canonicalization.
    assert (
        out.loc[
            out["ID"] == 101,
            "TXT_NATACION",
        ].iloc[0]
        == 1
    )
