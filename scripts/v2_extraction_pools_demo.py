#!/usr/bin/env python3
"""Demonstrate JSON-contract replay and pool assembly from invented data ONLY.

The actual GPT-6 Sol model is not called. No remote requests, training,
clinical data, inference, files written, or Git activity occur.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from extraction_contract_v2 import merge_llm_response,COMPACT_COLUMNS
from pool_contract_v2 import build_pools
from temporal_v2 import build_temporal_features


def synthetic_demo():
    fixture=json.loads((ROOT/'synthetic_data/v2_extraction_cases.json').read_text(encoding='utf-8'))
    mapping=json.loads((ROOT/'config/v2_ILLUSTRATIVE_alias_groups.json').read_text(encoding='utf-8'))
    original=pd.DataFrame(fixture['original_structured'])
    vocabulary=['NATACION','NADAR','CICLISMO','YOGA','PILATES','GIMNASIO','FUTBOL','SENDERISMO','TENIS','CAMINAR']
    args=dict(vocabulary=vocabulary,alias_map=mapping['alias_map'],group_map=mapping['group_map'])
    first=merge_llm_response(original,fixture['t1_response'],**args)
    second=merge_llm_response(original,fixture['t12_response'],**args)
    ids=original.ID.tolist()
    baseline=pd.DataFrame({'ID':ids,**{f'DEMO_BASE_{i:02d}':[i*100+j for j in range(4)] for i in range(1,25)}})
    t1_struct=pd.DataFrame({'ID':ids,**{f'DEMO_T1_{i:02d}':[i*2+j for j in range(4)] for i in range(1,46)}})
    paired=pd.DataFrame(json.loads((ROOT/'synthetic_data/v2_temporal_pairs.json').read_text(encoding='utf-8')))
    temporal=build_temporal_features(paired,participant_order=ids)
    pools=build_pools(baseline,t1_struct,first.compact_text,second.compact_text,temporal.features)
    assert all(frame.ID.tolist()==ids for frame in (pools.pool1,pools.pool2,pools.pool3,pools.pool3_without_temporal))
    assert [frame.shape[1]-1 for frame in (pools.pool1,pools.pool2,pools.pool3)] == [24,77,101]
    assert list(first.compact_text.columns)==['ID',*COMPACT_COLUMNS]
    assert list(second.compact_text.columns)==['ID',*COMPACT_COLUMNS]
    assert not pools.pool2.equals(pools.pool3_without_temporal)
    assert all('LABEL' not in col for col in pools.pool3.columns)
    print('=== V2_SYNTHETIC_EXTRACTION_POOLS_DEMO ===')
    print('PARTICIPANTS: 4 invented only, IDs explicitly realigned')
    print('EXTRACTION: offline API-shaped JSON replay, two independent synthetic passes')
    print('ORIGINAL_STRUCTURED: nonmissing values preserved; missing proposals merged')
    print('TEXT: 5 activity groups + 3 compact summary fields (8 total)')
    print('POOLS: 24 / 77 / 101 input features; plus ID (not a predictor)')
    print('POOL3_NO_TEMPORAL: 77 predictors; uses T1T2 text summaries, not Pool2 T1-only')
    print('TEMPORAL: 24 audited predictors; seven audit counts excluded')
    print('CAUTION: 45 DEMO_T1 columns and alias/group map are illustrative, not clinical parity')
    print('NO_API=1 NO_ML=1 NO_CLINICAL_DATA=1 NO_OUTPUT_FILES=1')
    print('=== END_V2_SYNTHETIC_EXTRACTION_POOLS_DEMO ===')
    return pools,first,second


if __name__ == '__main__':synthetic_demo()
