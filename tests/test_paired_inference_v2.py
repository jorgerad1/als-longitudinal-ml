"""Unit tests exclusively on invented binary prediction matrices."""
from __future__ import annotations
import sys,unittest
from pathlib import Path
from dataclasses import dataclass
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from paired_inference_v2 import (compare_pool_predictions,from_model_results,
    InferenceV2Error,holm_adjust,_counts_by_subject,_f1_macro_from_counts,_resampling_units)

FAMILIES=('LogReg','SVC','RF')
COMPARISONS=(('Pool2','Pool1'),('Pool3','Pool1'),('Pool3','Pool2'))


def make_data(n=30,k=4):
    rng=np.random.default_rng(123)
    y=np.array(([0,1]*(n//2))+([0] if n%2 else []),dtype=int)
    out={}
    for f in FAMILIES:
        p1=np.zeros((k,n),dtype=int)
        p2=np.broadcast_to(y,(k,n)).copy()
        p2[:,::5]=1-p2[:,::5]
        p3=np.broadcast_to(y,(k,n)).copy()
        p3[:,::11]=1-p3[:,::11]
        out[f,'Pool1']=p1.copy();out[f,'Pool2']=p2.copy();out[f,'Pool3']=p3.copy()
    return y,out

class InferenceTests(unittest.TestCase):
    def test_confusion_macro_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score
        y,p=make_data()
        counts=_counts_by_subject(y,p['RF','Pool2'])
        val=_f1_macro_from_counts(counts.sum(axis=0))
        self.assertAlmostEqual(float(val),f1_score(np.tile(y,4),p['RF','Pool2'].ravel(),average='macro'))
    def test_holm_adjust_and_monotonic(self):
        x=holm_adjust([.04,.01,.03,1.0])
        np.testing.assert_allclose(x,[.09,.04,.09,1.0])
    def test_nine_contrasts(self):
        y,p=make_data()
        rows=compare_pool_predictions(y,p,COMPARISONS,bootstrap_resamples=50,permutation_resamples=50)
        self.assertEqual(len(rows),9)
        self.assertEqual({r.classifier for r in rows},set(FAMILIES))
        self.assertEqual({r.unique_participants for r in rows},{30})
        self.assertTrue(all(0<=r.paired_permutation_p<=r.holm_adjusted_p<=1 for r in rows))
        self.assertTrue(all(r.conditional_ci_lower<=r.conditional_ci_upper for r in rows))
    def test_reproducibility(self):
        y,p=make_data()
        args=dict(bootstrap_resamples=55,permutation_resamples=55)
        a=compare_pool_predictions(y,p,COMPARISONS,**args)
        b=compare_pool_predictions(y,p,COMPARISONS,**args)
        self.assertEqual(a,b)
    def test_identical_models_zero_delta(self):
        y,p=make_data()
        for f in FAMILIES:p[f,'Pool2']=p[f,'Pool1'].copy()
        a=compare_pool_predictions(y,p,COMPARISONS,bootstrap_resamples=35,permutation_resamples=35)
        for r in a:
            if r.first=='Pool2' and r.second=='Pool1':
                self.assertEqual(r.pooled_delta_macro_f1,0)
                self.assertEqual(r.paired_permutation_p,1)
    def test_pair_units_are_atomic(self):
        y,p=make_data()
        grp=np.repeat(np.arange(15),2)
        _,units=_resampling_units(y,grp)
        self.assertEqual(len(units),15)
        self.assertTrue(all(len(unit)==2 for unit in units))
        contrasts=compare_pool_predictions(y,p,COMPARISONS,groups=grp,
            bootstrap_resamples=50,permutation_resamples=50)
        self.assertEqual(len(contrasts),9)
    def test_bad_pair_rejected(self):
        y,p=make_data()
        grp=np.repeat(np.arange(15),2);grp[0]=100
        with self.assertRaises(InferenceV2Error):
            compare_pool_predictions(y,p,COMPARISONS,groups=grp,bootstrap_resamples=40,permutation_resamples=40)
    def test_misaligned_seed_counts_rejected(self):
        y,p=make_data();p['RF','Pool1']=p['RF','Pool1'][:2]
        with self.assertRaises(InferenceV2Error):
            compare_pool_predictions(y,p,COMPARISONS,bootstrap_resamples=40,permutation_resamples=40)
    def test_nonbinary_predictions_rejected(self):
        y,p=make_data();p['RF','Pool1'][0,0]=2
        with self.assertRaises(InferenceV2Error):
            compare_pool_predictions(y,p,COMPARISONS,bootstrap_resamples=40,permutation_resamples=40)
    def test_wrong_comparison_count_rejected(self):
        y,p=make_data()
        with self.assertRaises(InferenceV2Error):
            compare_pool_predictions(y,p,COMPARISONS[:1],bootstrap_resamples=40,permutation_resamples=40)
    def test_adapter_splits_agreement(self):
        @dataclass
        class Fake:
            family:str;pool:str;seed:int;oof_predictions:np.ndarray;splits:tuple
        y,p=make_data();rs=[]
        for (fam,pool),matrix in p.items():
            for k in range(matrix.shape[0]):
                rs.append(Fake(fam,pool,42+k,matrix[k],(((0,1),(2,3)),)))
        a=from_model_results(rs,y,comparisons=COMPARISONS,bootstrap_resamples=50,permutation_resamples=50)
        self.assertEqual(len(a),9)
        rs[0].splits=(((0,2),(1,3)),)
        with self.assertRaises(InferenceV2Error):
            from_model_results(rs,y,comparisons=COMPARISONS,bootstrap_resamples=50,permutation_resamples=50)
    def test_sd_bootstrap_not_fold_independent(self):
        y,p=make_data();a=compare_pool_predictions(y,p,COMPARISONS,bootstrap_resamples=60,permutation_resamples=60)
        self.assertTrue(all(r.repetitions==4 for r in a))
        self.assertTrue(all(r.unique_participants==30 for r in a))
    def test_matched_comparisons(self):
        y,p=make_data()
        new={}
        for (f,pool),mat in p.items():
            name={'Pool1':'Pool2_T1','Pool2':'Pool3_without_temporal','Pool3':'Pool3_full'}[pool]
            new[f,name]=mat
        c=(('Pool3_full','Pool2_T1'),('Pool3_full','Pool3_without_temporal'),('Pool3_without_temporal','Pool2_T1'))
        a=compare_pool_predictions(y,new,c,groups=np.repeat(np.arange(15),2),
            bootstrap_resamples=40,permutation_resamples=40)
        self.assertEqual(len(a),9)

if __name__=='__main__':unittest.main()
