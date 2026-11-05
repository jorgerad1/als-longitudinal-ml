"""No clinical inputs; compare only already-public map literals and invented labels."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from canonical_maps_v2 import (read_public_config,adapt_maps,normalise,CanonicalMapError)
from v2_canonical_maps_demo import demonstrate

P=ROOT/'config/v2_CANONICAL_alias_groups.json'
PUBLIC_SOURCE=ROOT/'src/text_features.py'

def get_public_literals():
    tree=ast.parse(PUBLIC_SOURCE.read_text(encoding='utf8'))
    result={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('HABIT_ALIAS_MAP','TEXT_GROUP_MAP'):
                    result[target.id]=ast.literal_eval(node.value)
    return result

class CanonicalMapsTests(unittest.TestCase):
    def test_public_source_is_exact_origin(self):
        raw,_=read_public_config(P)
        original=get_public_literals()
        self.assertEqual(raw['HABIT_ALIAS_MAP'],original['HABIT_ALIAS_MAP'])
        self.assertEqual(raw['TEXT_GROUP_MAP'],original['TEXT_GROUP_MAP'])
    def test_author_verified_map_digest_prefixes(self):
        raw,_=read_public_config(P)
        for symbol,expected in [('HABIT_ALIAS_MAP','32a95e9693da'),('TEXT_GROUP_MAP','a5d081b1ac05')]:
            value=json.dumps(raw[symbol],sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str).encode('utf8')
            self.assertTrue(hashlib.sha256(value).hexdigest().startswith(expected))
    def test_map_cardinalities_and_group_coverage(self):
        _,(a,g)=read_public_config(P)
        self.assertEqual(len(a),36)
        self.assertTrue(len(g)>=5)
        self.assertEqual({g for memberships in g.values() for g in memberships},
                         {'ENDURANCE','GYM_STRENGTH','MIND_BODY','TEAM_RACKET','OUTDOOR'})
        self.assertTrue(all(isinstance(v,tuple) and v for v in g.values()))
    def test_normalise_prefix_and_accents(self):
        self.assertEqual(normalise('TXT_natación'),'NATACION')
        self.assertEqual(normalise('TXTG_TEAM_RACKET',group=True),'TEAM_RACKET')
    def test_group_overlap_preserved(self):
        _,(a,g)=read_public_config(P)
        raw=json.loads(P.read_text('utf8'))
        original_groups=raw['TEXT_GROUP_MAP']
        keys=list(original_groups)
        altered={k:list(v) for k,v in original_groups.items()}
        altered[keys[1]].append(altered[keys[0]][0])
        alias_map, modified = adapt_maps(raw['HABIT_ALIAS_MAP'],altered)
        canonical = alias_map.get(normalise(altered[keys[0]][0]), normalise(altered[keys[0]][0]))
        self.assertIn(normalise(keys[0],group=True),modified[canonical])
        self.assertIn(normalise(keys[1],group=True),modified[canonical])
    def test_map_tampering_detected(self):
        raw=json.loads(P.read_text('utf8'))
        raw['HABIT_ALIAS_MAP']={}
        with self.assertRaises(CanonicalMapError):
            adapt_maps(raw['HABIT_ALIAS_MAP'],raw['TEXT_GROUP_MAP'])
    def test_public_synthetic_integration(self):
        demonstrate()

    def test_overlap_activates_all_groups_but_counts_habit_once(self):
        import pandas as pd
        from extraction_contract_v2 import merge_llm_response
        aliases, groups = read_public_config(P)[1]
        shared = next((habit for habit, memberships in groups.items()
                       if len(memberships)>1), None)
        self.assertIsNotNone(shared, 'Actual public mapping must retain its overlap')
        item={'ID':'SYN_OVERLAP_001','PESO1':70}
        payload={'pacientes_actualizados':[{'datos':{**item,shared:1}}],
                 'habitos_comunes_nuevos':[]}
        result=merge_llm_response(pd.DataFrame([item]),payload,
            vocabulary=[shared],alias_map=aliases,group_map=groups)
        row=result.compact_text.iloc[0]
        self.assertEqual(row['TXT_HABIT_COUNT'],1)
        self.assertEqual(row['TXTG_N_GROUPS'],len(groups[shared]))
        for name in groups[shared]:
            self.assertEqual(row['TXTG_'+name],1)

if __name__=='__main__':unittest.main()
