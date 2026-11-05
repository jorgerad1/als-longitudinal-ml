"""Pure data transformation: corrected T1/T2 temporal features for paperNLP v2.

No patient data, APIs, filesystem I/O, classification, or model fitting.

Input uses generic long-format records with four columns: ``ID, base, T1, T2``.
``ID`` is for row alignment only and is never a model predictor. The actual
private workbook's T1/T2 column harmonization is deliberately not included;
the input contract requires one value per source-defined pair. This isolates
and demonstrates the exact arithmetic of the final temporal descriptors.

This independently authored public implementation follows audited Stage25/34
semantic definitions; it is not a verbatim copy of the private source scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# Audited Stage25/Stage34 schema. Six disjoint specific domains cover 42 pairs;
# IMC_CAT is the 43rd and is included ONLY in ALL.
DOMAINS: dict[str, tuple[str, ...]] = {
    "SOCIO": ("TIPOCASA", "ESTUDIOS", "FORM_CONT", "SOCIOECO", "TIPOTRABAJO", "VIVESOLO"),
    "ACTIVITY": ("DEPORTE", "DEPORTEFREC"),
    "BODY": ("DAR", "TIPOPESO", "PESO", "ALTURA", "FORMA", "ACUMULOGRASO", "CAMBIOBRUSCOPESO"),
    "SUBSTANCE_SLEEP": ("DROGAS", "CANNABIS", "COCAINA", "OPIACEOS", "DISEÑO", "OTRASDROGAS", "TABACO", "ALCOHOL", "HORASSUEÑO"),
    "PSYCHOSOCIAL": ("PREOCUPAC", "PERSONALIDAD", "SOCIALIZA"),
    "DIET": ("ALIMENTACION", "BEBIDASAZUCAR", "DULCES", "TIPOCOMIDA", "FASTFOOD", "FRITO", "PLANCHA", "EMPANADO", "HERVIDO", "GUISADO", "CRUDO", "OTRACOCINA", "COMIDAS_DIA", "CANT_COMIDA", "SACIEDAD"),
}
CHANGED_BASES = (
    "PESO", "ALTURA", "DEPORTEFREC", "HORASSUEÑO", "PREOCUPAC",
    "PERSONALIDAD", "SOCIALIZA", "ALIMENTACION", "BEBIDASAZUCAR",
    "DULCES", "FASTFOOD", "COMIDAS_DIA", "CANT_COMIDA", "SACIEDAD",
)
METRIC_BASES = ("PESO", "ALTURA", "COMIDAS_DIA")
ALL_BASES = tuple(base for members in DOMAINS.values() for base in members) + ("IMC_CAT",)
DOMAIN_MEMBERS = {"ALL": ALL_BASES, **DOMAINS}
FEATURE_COLUMNS = (
    tuple("T12_CHANGED_" + b for b in CHANGED_BASES)
    + tuple("T12_RATE_CHANGED_" + d for d in DOMAIN_MEMBERS)
    + tuple("T12_METRIC_DELTA_" + b for b in METRIC_BASES)
)
COVERAGE_COLUMNS = tuple("T12_N_COMPARABLE_" + d for d in DOMAIN_MEMBERS)
INPUT_COLUMNS = ("ID", "base", "T1", "T2")


class TemporalInputError(ValueError):
    """Input does not conform to the audited 43-pair schema."""


@dataclass(frozen=True)
class TemporalOutput:
    features: pd.DataFrame
    coverage_audit: pd.DataFrame


def validate_schema() -> None:
    names = [b for v in DOMAINS.values() for b in v]
    if len(names) != 42 or len(set(names)) != 42:
        raise RuntimeError("Six domains must cover 42 distinct pairs")
    if len(ALL_BASES) != 43 or len(set(ALL_BASES)) != 43:
        raise RuntimeError("ALL must cover 43 distinct pairs")
    if set(ALL_BASES) - set(names) != {"IMC_CAT"}:
        raise RuntimeError("IMC_CAT must occur only in ALL")
    if len(CHANGED_BASES) != 14 or len(set(CHANGED_BASES)) != 14:
        raise RuntimeError("Expected 14 unique change bases")
    if len(METRIC_BASES) != 3 or len(FEATURE_COLUMNS) != 24 or len(COVERAGE_COLUMNS) != 7:
        raise RuntimeError("Feature inventory must be 14+7+3; audit 7")
    if not set(CHANGED_BASES + METRIC_BASES).issubset(ALL_BASES):
        raise RuntimeError("Individual indicators refer to unknown pairs")
    if list(DOMAIN_MEMBERS) != ["ALL", "SOCIO", "ACTIVITY", "BODY", "SUBSTANCE_SLEEP", "PSYCHOSOCIAL", "DIET"]:
        raise RuntimeError("Domain order does not match audited source definitions")


def _clean(values: pd.Series, column: str) -> pd.Series:
    # Numeric codes (categories) must already be harmonized. Unparseable source
    # values cause an error instead of being silently converted to missing.
    converted = pd.to_numeric(values, errors="coerce")
    if (values.notna() & converted.isna()).any():
        raise TemporalInputError(f"Non-numeric nonmissing input in {column}")
    arr = converted.astype(float)
    if np.isinf(arr).any():
        raise TemporalInputError(f"Infinite input in {column}")
    return arr


def build_temporal_features(
    pairs: pd.DataFrame, participant_order: Sequence[str] | None = None
) -> TemporalOutput:
    """Generate exactly 24 revised features and seven audit-only coverage counts.

    Every participant must have precisely one row for each of 43 audited bases.
    A missing T1 or T2 means that pair is NOT comparable; no change or delta
    is inferred. A zero-comparable-pair rate is NaN, never 0. The source's
    coverage denominators are provided separately and not used as predictors.

    Values must be numeric or missing (nominal categories as their recorded
    numeric codes). The function does not annualize differences or interpret
    changes as causal. It is independent of any outcome or clinical labels.
    """
    validate_schema()
    if not isinstance(pairs, pd.DataFrame):
        raise TemporalInputError("Expected a pandas DataFrame")
    if set(pairs.columns) != set(INPUT_COLUMNS) or len(pairs.columns) != len(INPUT_COLUMNS):
        raise TemporalInputError("Input must have exactly ID, base, T1, T2 (no labels/metadata)")
    if pairs.empty or pairs["ID"].isna().any() or pairs["base"].isna().any():
        raise TemporalInputError("Empty input, null ID or null base")
    if not all(isinstance(x, str) and x.strip() for x in pairs["ID"]):
        raise TemporalInputError("Participant IDs must be nonblank strings")
    if not all(isinstance(x, str) and x.strip() for x in pairs["base"]):
        raise TemporalInputError("Pair names must be nonblank strings")
    if pairs.duplicated(["ID", "base"]).any():
        raise TemporalInputError("Duplicate participant/base pair")
    observed_bases = set(pairs["base"])
    if observed_bases != set(ALL_BASES):
        raise TemporalInputError("Pair map is not the audited 43-base schema")

    discovered_order = list(dict.fromkeys(pairs["ID"].tolist()))
    if participant_order is None:
        order = discovered_order
    else:
        order = list(participant_order)
        if len(order) != len(discovered_order) or len(set(order)) != len(order) or set(order) != set(discovered_order):
            raise TemporalInputError("Explicit participant_order does not match provided IDs")
    if len(pairs) != len(order) * 43:
        raise TemporalInputError("Expected exactly 43 source pairs per participant")
    counts = pairs.groupby("ID", sort=False)["base"].nunique()
    if not counts.eq(43).all():
        raise TemporalInputError("Every participant must have all 43 source pairs")

    src = pairs.copy()
    src["T1"] = _clean(src["T1"], "T1")
    src["T2"] = _clean(src["T2"], "T2")
    src = src.set_index(["ID", "base"])
    sorted_index = pd.MultiIndex.from_product([order, ALL_BASES], names=["ID", "base"])
    src = src.reindex(sorted_index)
    if len(src) != len(order) * 43:
        raise TemporalInputError("Pair alignment failed")
    t1 = src["T1"].to_numpy(dtype=float).reshape(len(order),43)
    t2 = src["T2"].to_numpy(dtype=float).reshape(len(order),43)
    obs = np.isfinite(t1) & np.isfinite(t2)
    changed = obs & (t1 != t2)
    pos = {b: i for i,b in enumerate(ALL_BASES)}

    out = {"ID":order}
    audit = {"ID":order}
    for base in CHANGED_BASES:
        idx=pos[base]
        out["T12_CHANGED_" + base] = np.where(obs[:,idx], changed[:,idx].astype(float), np.nan)
    for domain, members in DOMAIN_MEMBERS.items():
        indices=[pos[b] for b in members]
        denominator=obs[:,indices].sum(axis=1).astype(int)
        numerator=changed[:,indices].sum(axis=1).astype(int)
        ratio=np.divide(numerator,denominator,
                        out=np.full(len(order),np.nan,dtype=float),
                        where=denominator>0)
        out["T12_RATE_CHANGED_" + domain] = ratio
        audit["T12_N_COMPARABLE_" + domain] = denominator
    for base in METRIC_BASES:
        idx=pos[base]
        out["T12_METRIC_DELTA_"+base] = np.where(obs[:,idx],t2[:,idx]-t1[:,idx],np.nan)

    feat = pd.DataFrame(out,columns=("ID",)+FEATURE_COLUMNS)
    cov = pd.DataFrame(audit,columns=("ID",)+COVERAGE_COLUMNS)
    if list(feat.columns) != ["ID",*FEATURE_COLUMNS] or len(feat) != len(order):
        raise RuntimeError("Internal schema validation failed")
    if feat.drop(columns="ID").shape[1] != 24 or cov.drop(columns="ID").shape[1] != 7:
        raise RuntimeError("Unexpected output dimensionality")
    for domain in DOMAIN_MEMBERS:
        col="T12_RATE_CHANGED_"+domain
        den=cov["T12_N_COMPARABLE_"+domain]
        if not feat.loc[den.eq(0),col].isna().all():
            raise RuntimeError("Zero observed pairs must produce NaN, not 0")
    return TemporalOutput(features=feat,coverage_audit=cov)


validate_schema()
