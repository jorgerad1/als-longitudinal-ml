"""Generate a fully synthetic questionnaire dataset.

The records generated here are artificial examples created solely for
testing the public reproducibility workflow. They are not sampled from,
derived from, fitted to, or intended to reproduce the distributions of
the restricted clinical cohort.
"""

from pathlib import Path

import pandas as pd


N_SYNTHETIC = 40

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "synthetic_questionnaire.xlsx"
)


def make_record(i):
    """Construct one deterministic fictitious participant."""

    participant_id = f"SYN{i + 1:04d}"

    # Deliberately balanced and artificial outcome labels.
    diagnosis = (
        "CONTROL"
        if i % 2 == 0
        else "ELA"
    )

    # Completely artificial baseline variables.
    sex = i % 2
    age_cat = (i % 3) + 1
    family_neuro = 1 if i % 7 == 0 else 0
    baseline_work = (i % 4) + 1

    # Synthetic T1 values.
    tipocasa1 = (i % 3) + 1
    estudios1 = (i % 4) + 1
    socioeco1 = (i % 3) + 1
    tipotrabajo1 = (i % 4) + 1
    vivesolo1 = i % 2

    deporte1 = 1 if i % 3 != 0 else 0
    deportefrec1 = i % 7

    peso1 = 60 + (i % 17)
    altura1 = 155 + (i % 25)
    forma1 = (i % 3) + 1

    tabaco1 = 1 if i % 8 == 0 else 0
    alcohol1 = i % 4
    horas1 = 6 + (i % 3)

    preocupac1 = (i % 5) + 1
    personalidad1 = ((i + 1) % 5) + 1
    socializa1 = ((i + 2) % 5) + 1

    alimentacion1 = (i % 5) + 1
    bebida1 = i % 4
    dulces1 = (i + 1) % 4
    fastfood1 = (i + 2) % 4
    comidas1 = 2 + (i % 4)
    cant1 = (i % 5) + 1
    saciedad1 = ((i + 3) % 5) + 1

    # Artificial longitudinal changes.
    peso2 = peso1 + ((i % 5) - 2)
    altura2 = altura1

    deporte2 = (
        deporte1
        if i % 4 != 0
        else 1 - deporte1
    )

    deportefrec2 = max(
        0,
        deportefrec1
        + (
            1
            if i % 4 == 0
            else 0
        )
        - (
            1
            if i % 6 == 0
            else 0
        ),
    )

    tipocasa2 = (
        tipocasa1
        if i % 6
        else ((tipocasa1 % 3) + 1)
    )

    estudios2 = estudios1

    socioeco2 = (
        socioeco1
        if i % 5
        else ((socioeco1 % 3) + 1)
    )

    tipotrabajo2 = (
        tipotrabajo1
        if i % 4
        else ((tipotrabajo1 % 4) + 1)
    )

    vivesolo2 = (
        vivesolo1
        if i % 9
        else 1 - vivesolo1
    )

    forma2 = (
        forma1
        if i % 5
        else ((forma1 % 3) + 1)
    )

    tabaco2 = tabaco1

    alcohol2 = max(
        0,
        alcohol1
        - (
            1
            if i % 7 == 0
            else 0
        ),
    )

    horas2 = max(
        4,
        horas1
        + (
            1
            if i % 6 == 0
            else 0
        )
        - (
            1
            if i % 4 == 0
            else 0
        ),
    )

    preocupac2 = max(
        1,
        min(
            5,
            preocupac1
            + (
                1
                if i % 5 == 0
                else 0
            ),
        ),
    )

    personalidad2 = personalidad1

    socializa2 = max(
        1,
        min(
            5,
            socializa1
            - (
                1
                if i % 6 == 0
                else 0
            ),
        ),
    )

    alimentacion2 = max(
        1,
        min(
            5,
            alimentacion1
            + (
                1
                if i % 7 == 0
                else 0
            ),
        ),
    )

    bebida2 = max(
        0,
        bebida1
        - (
            1
            if i % 5 == 0
            else 0
        ),
    )

    dulces2 = max(
        0,
        dulces1
        - (
            1
            if i % 6 == 0
            else 0
        ),
    )

    fastfood2 = max(
        0,
        fastfood1
        - (
            1
            if i % 4 == 0
            else 0
        ),
    )

    comidas2 = comidas1

    cant2 = max(
        1,
        min(
            5,
            cant1
            + (
                1
                if i % 8 == 0
                else 0
            ),
        ),
    )

    saciedad2 = max(
        1,
        min(
            5,
            saciedad1
            + (
                1
                if i % 9 == 0
                else 0
            ),
        ),
    )

    # Entirely invented free-text sentences.
    habit_index = i % 6

    t1_texts = [
        "Nada dos veces por semana y pasea los fines de semana.",
        "Va al gimnasio y realiza ejercicios de fuerza.",
        "Sale en bicicleta de forma recreativa.",
        "Practica yoga ocasionalmente.",
        "Juega al tenis algunos fines de semana.",
        "No refiere una actividad deportiva habitual.",
    ]

    t2_texts = [
        "Actualmente camina y nada de forma ocasional.",
        "Mantiene actividad de gimnasio.",
        "Continua utilizando la bicicleta los fines de semana.",
        "Ha reducido el yoga y camina con frecuencia.",
        "Continua jugando al tenis ocasionalmente.",
        "No describe nuevos habitos deportivos.",
    ]

    free_text_t1 = t1_texts[
        habit_index
    ]

    free_text_t2 = t2_texts[
        habit_index
    ]

    # Deterministic synthetic "LLM-derived" habit indicators.
    # These permit an offline demonstration without calling an API.
    txt_natacion = 1 if habit_index == 0 else 0
    txt_caminar = 1 if habit_index == 0 else 0
    txt_gimnasio = 1 if habit_index == 1 else 0
    txt_ciclismo = 1 if habit_index == 2 else 0
    txt_yoga = 1 if habit_index == 3 else 0
    txt_tenis = 1 if habit_index == 4 else 0

    record = {
        "ID": participant_id,
        "DIAGNOSTICO": diagnosis,

        # Baseline
        "SEXO": sex,
        "EDAD_CAT": age_cat,
        "FAMILIA_NEURO": family_neuro,
        "TRABAJO_BASE": baseline_work,

        # T1
        "TIPOCASA1": tipocasa1,
        "ESTUDIOS1": estudios1,
        "SOCIOECO1": socioeco1,
        "TIPOTRABAJO1": tipotrabajo1,
        "VIVESOLO1": vivesolo1,
        "DEPORTE1": deporte1,
        "DEPORTEFREC1": deportefrec1,
        "PESO1": peso1,
        "ALTURA1": altura1,
        "FORMA1": forma1,
        "TABACO1": tabaco1,
        "ALCOHOL1": alcohol1,
        "HORASSUEÑO1": horas1,
        "PREOCUPAC1": preocupac1,
        "PERSONALIDAD1": personalidad1,
        "SOCIALIZA1": socializa1,
        "ALIMENTACION1": alimentacion1,
        "BEBIDASAZUCAR1": bebida1,
        "DULCES1": dulces1,
        "FASTFOOD1": fastfood1,
        "COMIDAS_DIA1": comidas1,
        "CANT_COMIDA1": cant1,
        "SACIEDAD1": saciedad1,

        # T2
        "TIPOCASA2": tipocasa2,
        "ESTUDIOS2": estudios2,
        "SOCIOECO2": socioeco2,
        "TIPOTRABAJO2": tipotrabajo2,
        "VIVESOLO2": vivesolo2,
        "DEPORTE2": deporte2,
        "DEPORTEFREC2": deportefrec2,
        "PESO2": peso2,
        "ALTURA2": altura2,
        "FORMA2": forma2,
        "TABACO2": tabaco2,
        "ALCOHOL2": alcohol2,
        "HORASSUEÑO2": horas2,
        "PREOCUPAC2": preocupac2,
        "PERSONALIDAD2": personalidad2,
        "SOCIALIZA2": socializa2,
        "ALIMENTACION2": alimentacion2,
        "BEBIDASAZUCAR2": bebida2,
        "DULCES2": dulces2,
        "FASTFOOD2": fastfood2,
        "COMIDAS_DIA2": comidas2,
        "CANT_COMIDA2": cant2,
        "SACIEDAD2": saciedad2,

        # Invented free text
        "FREE_TEXT_T1": free_text_t1,
        "FREE_TEXT_T2": free_text_t2,

        # Deterministic offline text-derived indicators
        "TXT_NATACION": txt_natacion,
        "TXT_CAMINAR": txt_caminar,
        "TXT_GIMNASIO": txt_gimnasio,
        "TXT_CICLISMO": txt_ciclismo,
        "TXT_YOGA": txt_yoga,
        "TXT_TENIS": txt_tenis,
    }

    # Add a few artificial missing values to exercise imputation.
    if i % 13 == 0:
        record["SOCIOECO1"] = None

    if i % 17 == 0:
        record["HORASSUEÑO1"] = None

    return record


def main():
    records = [
        make_record(i)
        for i in range(
            N_SYNTHETIC
        )
    ]

    df = pd.DataFrame(
        records
    )

    metadata = pd.DataFrame(
        [
            {
                "item":
                    "purpose",
                "value":
                    "Public end-to-end software demonstration only",
            },
            {
                "item":
                    "origin",
                "value":
                    "Fully artificial deterministic records",
            },
            {
                "item":
                    "clinical_data_used",
                "value":
                    "No",
            },
            {
                "item":
                    "distribution_fitted_to_private_cohort",
                "value":
                    "No",
            },
            {
                "item":
                    "n_records",
                "value":
                    N_SYNTHETIC,
            },
            {
                "item":
                    "outcome",
                "value":
                    "Artificial balanced CONTROL/ELA labels",
            },
        ]
    )

    with pd.ExcelWriter(
        OUTPUT_PATH,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="synthetic",
            index=False,
        )

        metadata.to_excel(
            writer,
            sheet_name="metadata",
            index=False,
        )

    print(
        f"Wrote {len(df)} fully "
        f"synthetic records to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
