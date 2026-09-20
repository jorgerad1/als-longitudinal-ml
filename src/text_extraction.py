"""Schema-guided LLM extraction and row realignment.

This module implements the GPT-4o mini text-to-table workflow used in
the final Pool2 and Pool3 analyses.

No patient data or credentials are stored in this module.
"""

import json
import math
import os
import time

import pandas as pd
from openai import OpenAI

from .text_features import (
    canonical_habit_name,
    normalize_patient_record,
    rebuild_dataframe,
)


MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
CHUNK_SIZE = 10

SYSTEM_PROMPT = (
    "Eres un asistente experto en datos médicos"
)


def split_chunks(
    patients,
    surveys,
    chunk_size=CHUNK_SIZE,
):
    """Yield aligned dataframe/text chunks."""
    for i in range(
        0,
        len(patients),
        chunk_size,
    ):
        yield (
            patients.iloc[
                i:i + chunk_size
            ],
            surveys[
                i:i + chunk_size
            ],
        )


def generate_chunk_prompt(
    chunk_data,
    new_columns,
):
    """Generate the literal user-prompt logic used in the study."""

    habit_names = (
        sorted(
            [
                col.replace(
                    "TXT_",
                    "",
                )
                for col
                in new_columns
            ]
        )
        if len(new_columns) > 0
        else []
    )

    return f"""
        Eres un asistente de datos que ayuda a completar un conjunto de registros de pacientes.

        Cada paciente tiene:
        - Un campo "datos" con valores estructurados (algunos pueden ser nulos o vacíos).
        - Un campo "encuesta" con texto libre que puede contener información útil sobre hábitos del paciente.

        También recibes una lista de hábitos ya detectados en otros grupos de pacientes:
        {habit_names}

        Instrucciones:
        1. Completa valores nulos o vacíos en "datos" usando el texto libre solo cuando la información sea clara.
        2. No modifiques los valores ya presentes en variables estructuradas.
        3. Si detectas un hábito nuevo claramente mencionado, añádelo en `habitos_comunes_nuevos`.
        4. Los nombres de hábitos deben devolverse en MAYÚSCULAS, sin acentos y con guiones bajos (por ejemplo: NATACION, CICLISMO, GIMNASIO).
        5. Para todos los hábitos conocidos (y los nuevos que marques en un paciente), incluye el campo en "datos" con:
           - 1 si el hábito se menciona claramente
           - 0 si no se menciona
        6. No dejes vacíos los campos de hábitos. Usa solo 1 o 0.
        7. Devuelve SOLO un JSON plano sin comentarios ni texto adicional, con el formato:
        {{
          "pacientes_actualizados": [{{ "datos": ... }}, ...],
          "habitos_comunes_nuevos": ["..."]
        }}

        Aquí tienes los datos:
        {chunk_data}
        """


def make_openai_client():
    """Create a client from the OPENAI_API_KEY environment variable."""

    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment "
            "variable is not set."
        )

    return OpenAI(
        api_key=api_key
    )


def request_llm(
    prompt,
    client=None,
):
    """Call the same Chat Completions endpoint used in the study.

    The historical analysis did not configure automatic retries,
    seed, top_p, max_tokens or response_format explicitly.
    """

    if client is None:
        client = make_openai_client()

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content":
                        SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        prompt,
                },
            ],
        )
    )

    try:
        return json.loads(
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:
        # The original notebook printed the complete API response.
        # The public version deliberately avoids doing so because an
        # API response may contain sensitive patient-derived content.
        print(
            "Could not parse LLM "
            f"response as JSON: {exc}"
        )
        return None


def is_missing_scalar(value):
    """Return True for missing/blank scalar values."""

    if isinstance(value, str):
        return (
            value.strip() == ""
        )

    try:
        return bool(
            pd.isna(value)
        )
    except Exception:
        return False


def normalize_id_key(value):
    """Normalize patient identifiers for deterministic alignment."""

    if is_missing_scalar(value):
        return None

    try:
        if (
            isinstance(
                value,
                float,
            )
            and float(
                value
            ).is_integer()
        ):
            return str(
                int(value)
            )
    except Exception:
        pass

    return str(
        value
    ).strip()


def merge_original_and_updated_record(
    original_record,
    updated_record,
    starting_cols,
):
    """Preserve known structured values and fill only missing ones."""

    merged = {}

    updated_record = (
        updated_record
        or {}
    )

    for col in starting_cols:

        original_value = (
            original_record.get(
                col,
                None,
            )
        )

        updated_value = (
            updated_record.get(
                col,
                None,
            )
        )

        if not is_missing_scalar(
            original_value
        ):
            # Never overwrite known structured values.
            merged[col] = (
                original_value
            )

        elif not is_missing_scalar(
            updated_value
        ):
            # Accept LLM completion only when original was missing.
            merged[col] = (
                updated_value
            )

        else:
            merged[col] = None

    # Preserve newly generated habit fields.
    for (
        key,
        value,
    ) in updated_record.items():

        if key not in starting_cols:
            merged[key] = value

    return merged


def align_chunk_output_to_input(
    df_chunk,
    llm_output,
    starting_cols,
    id_col="ID",
):
    """Realign LLM output to the original records by participant ID.

    Missing LLM rows are reconstructed from the original structured
    input instead of being discarded.
    """

    original_records = (
        df_chunk.to_dict(
            orient="records"
        )
    )

    if not isinstance(
        llm_output,
        dict,
    ):
        llm_output = {}

    returned_patients = (
        llm_output.get(
            "pacientes_actualizados",
            [],
        )
    )

    if not isinstance(
        returned_patients,
        list,
    ):
        returned_patients = []

    normalized_updates = []

    for patient in returned_patients:

        if not isinstance(
            patient,
            dict,
        ):
            continue

        patient_data = (
            patient.get(
                "datos",
                {},
            )
        )

        if not isinstance(
            patient_data,
            dict,
        ):
            patient_data = {}

        normalized_updates.append(
            normalize_patient_record(
                patient_data,
                starting_cols,
            )
        )

    aligned_rows = []

    if id_col in starting_cols:

        updates_by_id = {}
        overflow_updates = []

        for row in normalized_updates:

            row_key = (
                normalize_id_key(
                    row.get(
                        id_col
                    )
                )
            )

            if (
                row_key is None
                or row_key
                in updates_by_id
            ):
                overflow_updates.append(
                    row
                )

            else:
                updates_by_id[
                    row_key
                ] = row

        overflow_iter = iter(
            overflow_updates
        )

        for original_row in original_records:

            original_key = (
                normalize_id_key(
                    original_row.get(
                        id_col
                    )
                )
            )

            updated_row = (
                updates_by_id.pop(
                    original_key,
                    None,
                )
            )

            if updated_row is None:
                try:
                    updated_row = next(
                        overflow_iter
                    )
                except StopIteration:
                    updated_row = None

            aligned_rows.append(
                merge_original_and_updated_record(
                    original_row,
                    updated_row,
                    starting_cols,
                )
            )

    else:

        for (
            pos,
            original_row,
        ) in enumerate(
            original_records
        ):

            updated_row = (
                normalized_updates[pos]
                if pos
                < len(
                    normalized_updates
                )
                else None
            )

            aligned_rows.append(
                merge_original_and_updated_record(
                    original_row,
                    updated_row,
                    starting_cols,
                )
            )

    return aligned_rows


def process_with_llm(
    structured_df,
    survey_rows,
    client=None,
    chunk_size=CHUNK_SIZE,
    id_col="ID",
    verbose=True,
):
    """Run the complete sequential chunked LLM extraction workflow."""

    if len(
        structured_df
    ) != len(
        survey_rows
    ):
        raise ValueError(
            "structured_df and survey_rows "
            "must contain the same number "
            "of records."
        )

    starting_cols = list(
        structured_df.columns
    )

    new_columns = set()
    updated_rows = []

    n_chunks = math.ceil(
        len(structured_df)
        / chunk_size
    )

    for (
        chunk_idx,
        (
            df_chunk,
            survey_chunk,
        ),
    ) in enumerate(
        split_chunks(
            structured_df,
            survey_rows,
            chunk_size,
        ),
        start=1,
    ):

        records = (
            df_chunk.to_dict(
                orient="records"
            )
        )

        patients_json = [
            {
                "datos": record,
                "encuesta":
                    survey_chunk[idx],
            }
            for (
                idx,
                record,
            ) in enumerate(
                records
            )
        ]

        json_input = (
            json.dumps(
                {
                    "pacientes":
                        patients_json
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        prompt = (
            generate_chunk_prompt(
                json_input,
                new_columns,
            )
        )

        if verbose:
            print(
                "Processing chunk "
                f"{chunk_idx} "
                f"of {n_chunks} >>> ",
                end="",
            )

        start_time = time.time()

        output = request_llm(
            prompt,
            client=client,
        )

        elapsed = (
            time.time()
            - start_time
        )

        if verbose:
            print(
                "Finished. Time elapsed: "
                + time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        elapsed
                    ),
                )
            )

        if not isinstance(
            output,
            dict,
        ):
            if verbose:
                print(
                    "Invalid LLM output; "
                    "falling back to "
                    "original chunk values."
                )

            output = {
                "pacientes_actualizados":
                    [],
                "habitos_comunes_nuevos":
                    [],
            }

        returned = output.get(
            "pacientes_actualizados",
            [],
        )

        returned_n = (
            len(returned)
            if isinstance(
                returned,
                list,
            )
            else 0
        )

        if (
            returned_n
            != len(df_chunk)
            and verbose
        ):
            print(
                "LLM returned "
                f"{returned_n} rows "
                "for a chunk of "
                f"{len(df_chunk)}. "
                "Realigning by ID and "
                "reconstructing omitted rows "
                "from the original structured data."
            )

        normalized_rows = (
            align_chunk_output_to_input(
                df_chunk,
                output,
                starting_cols,
                id_col=id_col,
            )
        )

        habits = output.get(
            "habitos_comunes_nuevos",
            [],
        )

        if not isinstance(
            habits,
            list,
        ):
            habits = []

        normalized_new_columns = {
            canonical_habit_name(
                col
            )
            for col in habits
            if canonical_habit_name(
                col
            )
            is not None
        }

        updated_rows.extend(
            normalized_rows
        )

        # Newly discovered habits are supplied to later chunks.
        new_columns.update(
            normalized_new_columns
        )

    if len(updated_rows) != len(
        structured_df
    ):
        raise ValueError(
            "Chunk alignment failed: "
            f"expected {len(structured_df)} "
            "rows after LLM processing, "
            f"got {len(updated_rows)}."
        )

    df_updated = (
        rebuild_dataframe(
            updated_rows,
            starting_cols,
            new_columns,
        )
    )

    return (
        df_updated,
        new_columns,
    )
