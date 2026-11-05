"""Synthetic-only tests. No network, external services, or clinical data."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from extraction_contract_v2 import merge_llm_response, ExtractionContractError, COMPACT_COLUMNS
from pool_contract_v2 import build_pools, PoolContractError
from temporal_v2 import build_temporal_features, COVERAGE_COLUMNS, FEATURE_COLUMNS
from v2_extraction_pools_demo import synthetic_demo

FIXTURE=json.loads((ROOT/'synthetic_data/v2_extraction_cases.json').read_text(encoding='utf-8'))
MAPPING=json.loads((ROOT/'config/v2_ILLUSTRATIVE_alias_groups.json').read_text(encoding='utf-8'))
VOCAB=['NATACION','NADAR','CICLISMO','YOGA','PILATES','GIMNASIO','FUTBOL','SENDERISMO','TENIS','CAMINAR']
ARGS=dict(vocabulary=VOCAB,alias_map=MAPPING['alias_map'],group_map=MAPPING['group_map'])


class TestOfflineExtractionAndPools(unittest.TestCase):
    def setUp(self):
        self.source=pd.DataFrame(FIXTURE['original_structured'])
        self.t1=FIXTURE['t1_response']
        self.t12=FIXTURE['t12_response']

    def merge(self,payload=None,source=None):
        return merge_llm_response(self.source if source is None else source,
                                  self.t1 if payload is None else payload,**ARGS)

    def test_api_contract_and_realignment(self):
        got=self.merge()
        self.assertEqual(got.structured.ID.tolist(),['SYN_A','SYN_B','SYN_C','SYN_ZERO'])
        self.assertEqual(got.compact_text.ID.tolist(),got.structured.ID.tolist())
        self.assertEqual(len(got.individual_mentions),4)

    def test_source_precedence_and_missing_only(self):
        got=self.merge().structured.set_index('ID')
        self.assertEqual(got.loc['SYN_A','PESO1'],60) # do not overwrite with 999
        self.assertEqual(got.loc['SYN_B','DEPORTE1'],1) # do not overwrite with 0
        self.assertEqual(got.loc['SYN_A','DEPORTE1'],1) # previously missing
        self.assertEqual(got.loc['SYN_C','PESO1'],65) # previously missing
        self.assertEqual(got.loc['SYN_B','SOCIALIZA1'],3) # previously missing
        self.assertEqual(self.merge().accepted_structured_proposals,4) # includes SYN_ZERO

    def test_alias_dedup_compact_8(self):
        got=self.merge()
        row=got.compact_text.set_index('ID').loc['SYN_A']
        self.assertEqual(row.TXT_HABIT_COUNT,2) # NATACION+NADAR same concept; CICLISMO
        self.assertEqual(row.TXT_ANY_HABIT,1)
        self.assertEqual(row.TXTG_ENDURANCE,1)
        self.assertEqual(row.TXTG_N_GROUPS,1)
        self.assertEqual(got.compact_text.shape,(4,9))

    def test_no_mention_is_not_diagnosis(self):
        got=self.merge().compact_text.set_index('ID')
        self.assertEqual(got.loc['SYN_ZERO','TXT_ANY_HABIT'],0)
        self.assertTrue(all(col.startswith('TXT') for col in got.columns))

    def test_separate_t1_t12_passes(self):
        a=self.merge().compact_text.set_index('ID')
        b=self.merge(self.t12).compact_text.set_index('ID')
        self.assertFalse(a.equals(b))
        self.assertEqual(a.loc['SYN_A','TXTG_TEAM_RACKET'],0)
        self.assertEqual(b.loc['SYN_A','TXTG_TEAM_RACKET'],1)

    def test_reject_missing_extra_duplicate_ids(self):
        for replacement in (
            self.t1['pacientes_actualizados'][1:],
            self.t1['pacientes_actualizados']+[self.t1['pacientes_actualizados'][0]],
        ):
            obj={**self.t1,'pacientes_actualizados':replacement}
            with self.subTest(length=len(replacement)),self.assertRaises(ExtractionContractError):self.merge(obj)

    def test_reject_json_and_unexpected_field(self):
        with self.assertRaises(ExtractionContractError):self.merge('{invalid json')
        copy=json.loads(json.dumps(self.t1))
        copy['pacientes_actualizados'][0]['datos']['UNKNOWN_COLUMN']=1
        with self.assertRaises(ExtractionContractError):self.merge(copy)

    def test_reject_nonbinary_habit(self):
        copy=json.loads(json.dumps(self.t1))
        copy['pacientes_actualizados'][1]['datos']['NATACION']=2
        with self.assertRaises(ExtractionContractError):self.merge(copy)

    def test_reject_outcome_in_source(self):
        with self.assertRaises(ExtractionContractError):self.merge(source=self.source.assign(diagnosis=1))

    def test_full_dimension_contract_and_distinct_arm(self):
        pools,_,_=synthetic_demo()
        self.assertEqual((pools.pool1.shape,pools.pool2.shape,pools.pool3.shape,pools.pool3_without_temporal.shape),
                         ((4,25),(4,78),(4,102),(4,78)))
        self.assertNotEqual(pools.pool2.loc[0,'TXTG_TEAM_RACKET'],pools.pool3_without_temporal.loc[0,'TXTG_TEAM_RACKET'])
        self.assertEqual(list(pools.pool3.columns[-24:]),list(FEATURE_COLUMNS))
        self.assertFalse(set(COVERAGE_COLUMNS)&set(pools.pool3.columns))

    def test_reject_temporal_coverage_columns(self):
        paired=pd.DataFrame(json.loads((ROOT/'synthetic_data/v2_temporal_pairs.json').read_text(encoding='utf8')))
        temporal=build_temporal_features(paired)
        pools,first,second=synthetic_demo()
        baseline=pools.pool1
        t1_struct=pools.pool2[['ID',*[x for x in pools.pool2 if x.startswith('DEMO_T1_')]]]
        bad=temporal.features.merge(temporal.coverage_audit,on='ID')
        with self.assertRaises(PoolContractError):
            build_pools(baseline,t1_struct,first.compact_text,second.compact_text,bad)

    def test_reject_participant_drift_and_duplicate(self):
        pools,first,second=synthetic_demo()
        baseline=pools.pool1
        t1_struct=pools.pool2[['ID',*[x for x in pools.pool2 if x.startswith('DEMO_T1_')]]]
        temp=pools.pool3[['ID',*FEATURE_COLUMNS]]
        drift=t1_struct.copy();drift.loc[0,'ID']='SYN_DIFFERENT'
        with self.assertRaises(PoolContractError):
            build_pools(baseline,drift,first.compact_text,second.compact_text,temp)
        dup=baseline.copy();dup.loc[0,'ID']=dup.loc[1,'ID']
        with self.assertRaises(PoolContractError):
            build_pools(dup,t1_struct,first.compact_text,second.compact_text,temp)

    def test_reject_outcome_predictor_in_pool(self):
        pools,first,second=synthetic_demo()
        baseline=pools.pool1.assign(ALS=1)
        t1_struct=pools.pool2[['ID',*[x for x in pools.pool2 if x.startswith('DEMO_T1_')]]]
        temp=pools.pool3[['ID',*FEATURE_COLUMNS]]
        with self.assertRaises(PoolContractError):
            build_pools(baseline,t1_struct,first.compact_text,second.compact_text,temp)


if __name__=='__main__': unittest.main()
