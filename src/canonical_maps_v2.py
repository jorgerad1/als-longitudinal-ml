"""Adapt the already public 36-alias/5-group tables to the offline v2 contract.

No OpenAI client, patient dataset, dynamic source import, or external network.
The source JSON is reconstructed exclusively from the public text_features.py
literal dictionaries during the separate Stage71 staging operation.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import unicodedata
from typing import Mapping

GROUP_ORDER = ('ENDURANCE', 'GYM_STRENGTH', 'MIND_BODY', 'TEAM_RACKET', 'OUTDOOR')
EXPECTED_GROUPS = frozenset(GROUP_ORDER)

class CanonicalMapError(ValueError):
    pass


def normalise(raw: str, *, group: bool=False) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CanonicalMapError('A map name is empty or non-text')
    raw=unicodedata.normalize('NFKD',raw)
    raw=''.join(c for c in raw if not unicodedata.combining(c))
    key=re.sub(r'[^A-Za-z0-9]+','_',raw).strip('_').upper()
    if group and key.startswith('TXTG_'):
        key=key[5:]
    elif not group and key.startswith('TXT_'):
        key=key[4:]
    if not key:
        raise CanonicalMapError('Empty normalised key')
    return key


def adapt_maps(original_aliases: Mapping[str,str], original_groups: Mapping[str,list[str]]):
    """Return (alias->canonical, canonical->tuple of all groups).

    A habit can belong to multiple original groups. Do not collapse this
    many-to-many relation: each mentioned habit activates *every* group
    containing it, while TXT_HABIT_COUNT counts that habit once.
    This does not prove equivalence to private Stage52 feature values.
    """
    if not isinstance(original_aliases, Mapping) or len(original_aliases)!=36:
        raise CanonicalMapError('Expected 36 original public aliases')
    if not isinstance(original_groups,Mapping) or len(original_groups)!=5:
        raise CanonicalMapError('Expected 5 original public groups')
    aliases={}
    for raw_key,raw_value in original_aliases.items():
        key=normalise(raw_key)
        value=normalise(raw_value)
        if key in aliases and aliases[key]!=value:
            raise CanonicalMapError('Conflicting normalised alias')
        aliases[key]=value
    if len(aliases)!=36:
        raise CanonicalMapError('Distinct original aliases collapse under normalisation')
    groups={}
    recognised=set()
    for raw_group,members in original_groups.items():
        group=normalise(raw_group,group=True)
        if group not in EXPECTED_GROUPS or group in recognised:
            raise CanonicalMapError('Unexpected/duplicate public group')
        recognised.add(group)
        if not isinstance(members,list) or not members or any(not isinstance(x,str) for x in members):
            raise CanonicalMapError('Public group must list habit labels')
        for raw in members:
            habit=normalise(raw)
            canonical=aliases.get(habit,habit)
            groups.setdefault(canonical, set()).add(group)
    if recognised!=EXPECTED_GROUPS:
        raise CanonicalMapError('Not all expected groups were present')
    return aliases, {habit: tuple(g for g in GROUP_ORDER if g in memberships)
                     for habit, memberships in groups.items()}


def read_public_config(path:Path):
    p=Path(path)
    obj=json.loads(p.read_text(encoding='utf8'))
    if not isinstance(obj,dict) or set(obj)!= {'provenance','HABIT_ALIAS_MAP','TEXT_GROUP_MAP'}:
        raise CanonicalMapError('Unexpected config keys')
    provenance=obj['provenance']
    if provenance.get('source')!='src/text_features.py':
        raise CanonicalMapError('Unexpected public origin')
    return obj,adapt_maps(obj['HABIT_ALIAS_MAP'],obj['TEXT_GROUP_MAP'])
