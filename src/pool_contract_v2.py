"""Synthetic-only Pool1/Pool2/Pool3 assembly contract for paperNLP v2.

Inputs are explicit, independent feature matrices. No labels, demographics,
private workbook parsing, model fitting or inferential calculations are used.
The demonstration uses 24 invented baseline and 45 invented T1 columns merely
to reach the published input dimensions; these are NOT the real questionnaire
feature names nor a verified port of Stage52 pool selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from extraction_contract_v2 import COMPACT_COLUMNS, FORBIDDEN_COLS
from temporal_v2 import FEATURE_COLUMNS, COVERAGE_COLUMNS


class PoolContractError(ValueError):
    """Mismatched IDs, outcomes, or invalid pool feature contracts."""


@dataclass(frozen=True)
class FeaturePools:
    pool1: pd.DataFrame
    pool2: pd.DataFrame
    pool3: pd.DataFrame
    pool3_without_temporal: pd.DataFrame


def _checked(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or "ID" not in frame or frame.empty:
        raise PoolContractError(name + ": expected nonempty ID-indexed DataFrame")
    if frame.columns.duplicated().any() or frame["ID"].isna().any():
        raise PoolContractError(name + ": duplicated columns or missing ID")
    ids=frame["ID"].tolist()
    if len(ids)!=len(set(ids)) or any(not isinstance(x,str) or not x.strip() for x in ids):
        raise PoolContractError(name + ": invalid ID sequence")
    others = [c for c in frame.columns if c != "ID"]
    if not others or any(not isinstance(c,str) or c.lower() in FORBIDDEN_COLS for c in others):
        raise PoolContractError(name + ": outcome or malformed predictor name")
    if frame[others].shape[1] != len(others):
        raise PoolContractError(name + ": duplicated predictor names")
    if any(isinstance(v, (list, dict)) for col in others for v in frame[col].tolist()):
        raise PoolContractError(name + ": non-scalar predictor value")
    for col in others:
        raw = frame[col]
        numbers = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numbers.isna()).any():
            raise PoolContractError(name + ": nonnumeric predictor field")
    return frame.set_index("ID").copy()


def build_pools(
    baseline: pd.DataFrame,
    t1_structured: pd.DataFrame,
    t1_text: pd.DataFrame,
    t12_text: pd.DataFrame,
    temporal: pd.DataFrame,
    *,
    expected_dimensions: Sequence[int] = (24,77,101),
) -> FeaturePools:
    """Assemble pools, preserving T1-only and T1+T2 extraction as separate data.

    True temporal predictor block has exactly the 24 verified feature names;
    the seven audit denominators are forbidden. The 77-column non-temporal
    Pool3 variant differs from Pool2 because it uses its own T1+T2 text pass.
    """
    if tuple(expected_dimensions) != (24,77,101):
        raise PoolContractError("This documented demonstration targets 24/77/101")
    frames = {
        "baseline":_checked(baseline,"baseline"),
        "t1_structured":_checked(t1_structured,"t1_structured"),
        "t1_text":_checked(t1_text,"t1_text"),
        "t12_text":_checked(t12_text,"t12_text"),
        "temporal":_checked(temporal,"temporal"),
    }
    order = baseline["ID"].tolist()
    for name,frame in frames.items():
        if set(frame.index) != set(order):
            raise PoolContractError(name + ": participant mismatch")
        frames[name] = frame.loc[order]
    b,s,t1,t12,tempor = (frames[k] for k in frames)
    if tuple(t1.columns) != COMPACT_COLUMNS or tuple(t12.columns) != COMPACT_COLUMNS:
        raise PoolContractError("Text summaries must use the same eight compact fields")
    if tuple(tempor.columns) != FEATURE_COLUMNS:
        raise PoolContractError("Temporal block must match the 24 audited predictor columns")
    if set(COVERAGE_COLUMNS) & set().union(*(set(frame.columns) for frame in frames.values())):
        raise PoolContractError("Coverage counts cannot enter predictive matrices")
    if (set(b.columns)&set(s.columns)) or (set(b.columns)&set(t1.columns)) or (set(s.columns)&set(t1.columns)):
        raise PoolContractError("Overlapping baseline/T1/text predictor column names")
    if set(tempor.columns)&(set(b.columns)|set(s.columns)|set(t1.columns)):
        raise PoolContractError("Overlapping temporal/other predictor names")
    pool1 = b.copy()
    pool2 = pd.concat([b,s,t1],axis=1)
    no_temporal = pd.concat([b,s,t12],axis=1)
    pool3 = pd.concat([no_temporal,tempor],axis=1)
    if (pool1.shape[1],pool2.shape[1],pool3.shape[1]) != tuple(expected_dimensions):
        raise PoolContractError("Input counts do not match 24/77/101; inspect schema")
    if no_temporal.shape[1]!=77 or not pool2.columns.equals(no_temporal.columns):
        raise PoolContractError("Non-temporal arm must share 77 predictor names")
    if not pool1.index.equals(pool2.index) or not pool2.index.equals(pool3.index):
        raise PoolContractError("Pool participant alignment failed")
    def show(frame):
        return frame.reset_index(drop=False).rename(columns={"index":"ID"})
    return FeaturePools(show(pool1),show(pool2),show(pool3),show(no_temporal))
