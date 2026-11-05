"""Offline, synthetic-first extraction output contract for paperNLP v2.

Models are NOT called here. Source values are preserved over LLM proposals;
outputs are joined by explicit ID and compact text features are derived from
reported *mentions* rather than clinical assertions of behavior absence.

This public demonstration uses illustrative alias/group mappings. It does not
claim the mappings are identical to the private Stage41/Stage52 catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import json
import re
import unicodedata

import numpy as np
import pandas as pd

COMPACT_COLUMNS = (
    "TXTG_ENDURANCE", "TXTG_GYM_STRENGTH", "TXTG_MIND_BODY",
    "TXTG_TEAM_RACKET", "TXTG_OUTDOOR", "TXT_ANY_HABIT",
    "TXT_HABIT_COUNT", "TXTG_N_GROUPS",
)
GROUP_NAMES = tuple(col.removeprefix("TXTG_") for col in COMPACT_COLUMNS[:5])
FORBIDDEN_COLS = frozenset({
    "label", "target", "outcome", "diagnosis", "diagnostico",
    "diagnóstico", "als", "is_als", "ela", "clase", "y_true",
})


class ExtractionContractError(ValueError):
    """Malformed or ambiguous input; reject rather than align by row index."""


@dataclass(frozen=True)
class ExtractionOutput:
    structured: pd.DataFrame
    compact_text: pd.DataFrame
    individual_mentions: pd.DataFrame
    accepted_structured_proposals: int


def _missing(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _canonical(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionContractError("Empty or invalid habit name")
    norm = unicodedata.normalize("NFKD", raw)
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    norm = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_").upper()
    if norm.startswith("TXT_"):
        norm = norm[4:]
    if not norm or not re.fullmatch(r"[A-Z0-9_]+", norm):
        raise ExtractionContractError("Invalid normalized habit name")
    return norm


def _validate_ids(values: Sequence[Any], label: str) -> list[str]:
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ExtractionContractError(label + ": IDs must be nonempty strings")
    if len(values) != len(set(values)):
        raise ExtractionContractError(label + ": duplicate IDs")
    return list(values)


def _parse_json(payload: str | Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(payload, str):
        try:
            item = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExtractionContractError("Invalid JSON response") from exc
    else:
        item = payload
    if not isinstance(item, Mapping) or set(item) != {"pacientes_actualizados", "habitos_comunes_nuevos"}:
        raise ExtractionContractError("Unexpected top-level response schema")
    rows = item["pacientes_actualizados"]
    new = item["habitos_comunes_nuevos"]
    if not isinstance(rows, list) or not rows or not isinstance(new, list):
        raise ExtractionContractError("Invalid response arrays")
    if any(not isinstance(x, Mapping) or set(x) != {"datos"} or not isinstance(x["datos"], Mapping) for x in rows):
        raise ExtractionContractError("Every response row requires a datos dictionary")
    if any(not isinstance(x, str) for x in new):
        raise ExtractionContractError("Habit names must be strings")
    return [dict(x["datos"]) for x in rows], new


def merge_llm_response(
    original: pd.DataFrame,
    payload: str | Mapping[str, Any],
    *,
    vocabulary: Sequence[str],
    alias_map: Mapping[str, str],
    group_map: Mapping[str, str | Sequence[str]],
) -> ExtractionOutput:
    """Merge an offline saved API-shaped response into a structured synthetic input.

    - Match every response record by `datos.ID`, never response position.
    - Preserve every original nonmissing structured value.
    - Accept proposals only for empty structured fields (not evidence validation).
    - Count binary *mentions* only; 0 means not explicitly mentioned in this
      synthetic response, NOT proof that the person never performed the activity.
    - Newly proposed vocabulary names require explicit inclusion in
      habitos_comunes_nuevos; unknown structured columns are rejected.
    """
    if not isinstance(original, pd.DataFrame) or "ID" not in original.columns:
        raise ExtractionContractError("Original data must be a DataFrame with ID")
    if original.columns.duplicated().any():
        raise ExtractionContractError("Duplicated original column names")
    if any(str(c).strip().lower() in FORBIDDEN_COLS for c in original.columns):
        raise ExtractionContractError("Outcomes/diagnoses are forbidden in extraction input")
    ids = _validate_ids(original["ID"].tolist(), "original")
    structured_cols = [c for c in original.columns if c != "ID"]
    if not structured_cols or not all(isinstance(c, str) and c.strip() for c in structured_cols):
        raise ExtractionContractError("Expected nonempty structured feature names")
    rows, new_habits = _parse_json(payload)
    returned_ids = _validate_ids([x.get("ID") for x in rows], "response")
    if set(returned_ids) != set(ids):
        raise ExtractionContractError("Missing or unexpected response participant ID")
    if not isinstance(vocabulary, Sequence) or isinstance(vocabulary, str):
        raise ExtractionContractError("Vocabulary must be a list of names")
    vocab = {_canonical(v) for v in vocabulary}
    vocab.update(_canonical(v) for v in new_habits)
    aliases = {_canonical(k): _canonical(v) for k, v in alias_map.items()}
    # A canonical habit may legitimately be in several original public groups.
    # Accept the historical single-group mapping for backwards compatibility,
    # and a tuple/list/set of groups for the canonical public mapping.
    collected: dict[str, set[str]] = {}
    for raw_habit, raw_memberships in group_map.items():
        habit = _canonical(raw_habit)
        if isinstance(raw_memberships, str):
            memberships = (raw_memberships,)
        elif isinstance(raw_memberships, (tuple, list, set, frozenset)) and raw_memberships:
            memberships = tuple(raw_memberships)
        else:
            raise ExtractionContractError("Invalid group memberships")
        if any(not isinstance(g, str) or not g.strip() for g in memberships):
            raise ExtractionContractError("Invalid compact group name")
        names = {g.upper().removeprefix("TXTG_") for g in memberships}
        if any(g not in GROUP_NAMES for g in names):
            raise ExtractionContractError("Unknown compact group")
        collected.setdefault(habit, set()).update(names)
    groups = {habit: tuple(g for g in GROUP_NAMES if g in memberships)
              for habit, memberships in collected.items()}

    response_by_id = dict(zip(returned_ids, rows))
    original_by_id = original.set_index("ID", drop=False)
    final_rows = []
    individual = []
    accepted = 0
    for pid in ids:
        baseline = original_by_id.loc[pid]
        returned = response_by_id[pid]
        unexpected = []
        raw_mentions: dict[str, int] = {}
        for raw_key, proposal in returned.items():
            if raw_key == "ID":
                continue
            if not isinstance(raw_key, str):
                raise ExtractionContractError("Non-string response key")
            if raw_key in structured_cols:
                continue
            normalized = _canonical(raw_key)
            if normalized not in vocab:
                unexpected.append(raw_key)
                continue
            if isinstance(proposal, (bool, np.bool_)) or proposal not in (0, 1):
                raise ExtractionContractError("Habit mentions must be binary 0/1")
            canonical = aliases.get(normalized, normalized)
            raw_mentions[canonical] = max(raw_mentions.get(canonical, 0), int(proposal))
        if unexpected:
            raise ExtractionContractError("Response includes unapproved structured/habit fields")
        merged = {"ID": pid}
        for col in structured_cols:
            previous = baseline[col]
            proposed = returned.get(col)
            if _missing(previous) and not _missing(proposed):
                merged[col] = proposed
                accepted += 1
            else:
                merged[col] = previous
        final_rows.append(merged)
        individual.append({"ID": pid, **raw_mentions})
    mention_frame = pd.DataFrame(individual).fillna(0)
    mention_cols = sorted(c for c in mention_frame if c != "ID")
    mention_frame = mention_frame[["ID", *mention_cols]]
    compact_rows = []
    for _, rec in mention_frame.iterrows():
        matched = {group: 0 for group in GROUP_NAMES}
        total = 0
        for col in mention_cols:
            if rec[col] == 1:
                total += 1
                if col in groups:
                    for group in groups[col]:
                        matched[group] = 1
        compact_rows.append({
            "ID": rec["ID"],
            **{"TXTG_" + g: matched[g] for g in GROUP_NAMES},
            "TXT_ANY_HABIT": int(total > 0),
            "TXT_HABIT_COUNT": total,
            "TXTG_N_GROUPS": sum(matched.values()),
        })
    compact = pd.DataFrame(compact_rows, columns=["ID", *COMPACT_COLUMNS])
    return ExtractionOutput(
        structured=pd.DataFrame(final_rows, columns=["ID", *structured_cols]),
        compact_text=compact,
        individual_mentions=mention_frame,
        accepted_structured_proposals=accepted,
    )
