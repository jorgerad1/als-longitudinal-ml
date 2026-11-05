#!/usr/bin/env python3
"""Exercise all five original public groups on wholly invented JSON responses."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from canonical_maps_v2 import normalise,read_public_config
from extraction_contract_v2 import merge_llm_response


def demonstrate():
    raw,(aliases,groups)=read_public_config(ROOT/'config/v2_CANONICAL_alias_groups.json')
    chosen=[]
    for group in ('ENDURANCE','GYM_STRENGTH','MIND_BODY','TEAM_RACKET','OUTDOOR'):
        chosen.append(next(k for k,v in groups.items() if group in v))
    chosen=list(dict.fromkeys(chosen)) # one habit may activate multiple groups
    invented={'ID':'SYNTHETIC_V2_001','PESO1':60}
    response={'ID':'SYNTHETIC_V2_001','PESO1':999,**{k:1 for k in chosen}}
    payload={'pacientes_actualizados':[{'datos':response}],'habitos_comunes_nuevos':[]}
    result=merge_llm_response(pd.DataFrame([invented]),payload,
        vocabulary=chosen,alias_map=aliases,group_map=groups)
    row=result.compact_text.iloc[0]
    assert result.structured.iloc[0]['PESO1']==60
    assert all(row['TXTG_'+g]==1 for g in ('ENDURANCE','GYM_STRENGTH','MIND_BODY','TEAM_RACKET','OUTDOOR'))
    assert row['TXTG_N_GROUPS']==5 and row['TXT_HABIT_COUNT']==len(chosen)
    print(f'V2_CANONICAL_MAPS_DEMO_OK: 36 public aliases, 5 original groups, {len(chosen)} distinct synthetic mentions')
    print('SOURCE_VALUE_PRECEDENCE_OK: True; NO_API=1 NO_ML=1 NO_CLINICAL_DATA=1 NO_OUTPUT_FILES=1')

if __name__=='__main__':demonstrate()
