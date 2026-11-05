"""Synthetic-only tests. Never import restricted clinical files."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
from nested_cv_v2 import (TrainOnlyPreprocessing,model_search,run_nested_cv,
                           summarize_results,V2ValidationError)

class NestedCVTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(37)
        self.y=np.array([0,1]*12)
        self.groups=np.repeat(np.arange(12),2)
        self.df=pd.DataFrame({
            'age':rng.normal(55,9,size=24),
            'TXT_HABIT_COUNT':rng.integers(0,4,size=24),
            'T12_CHANGED_PESO':rng.integers(0,2,size=24),
            'cat':np.tile(['a','b','c'],8),
            'hidden_missing':np.r_[np.ones(2),np.repeat(np.nan,22)],
        })

    def test_train_only_missingness(self):
        pre=TrainOnlyPreprocessing(support_min=2)
        pre.fit(self.df)
        self.assertNotIn('hidden_missing',pre.keep_columns_)
        self.assertEqual(pre.transform(self.df).shape[0],24)
    def test_train_partition_values_only(self):
        a=self.df.copy();a.loc[:8,'age']=np.nan
        pre=TrainOnlyPreprocessing(support_min=2).fit(a.iloc[:12])
        self.assertNotIn('age',pre.keep_columns_)
        # Original dataset age is mostly observed; this verifies fold-only selection.
        self.assertIn('age',TrainOnlyPreprocessing(support_min=2).fit(self.df).keep_columns_)
    def test_correlation_filter(self):
        df=pd.DataFrame({'a':np.arange(20,dtype=float),
                         'b':np.arange(20,dtype=float)*2,
                         'c':np.tile([0.0,1.0],10)})
        with_corr=TrainOnlyPreprocessing(support_min=2,correlation_filter=True).fit(df)
        without=TrainOnlyPreprocessing(support_min=2,correlation_filter=False).fit(df)
        self.assertGreater(without.transform(df).shape[1],with_corr.transform(df).shape[1])
        self.assertEqual(with_corr.correlation_removed_count_,1)
    def test_reject_identifier_in_predictors(self):
        with self.assertRaises(V2ValidationError):
            run_nested_cv({'Pool1':self.df.assign(ID=range(24))},self.y,fast=True)
    def test_classifiers_and_params(self):
        for family in ['LogReg','SVC','RF']:
            s=model_search(family,inner_cv=2,seed=42)
            self.assertEqual(s.scoring,'f1_macro')
            self.assertEqual(s.estimator.named_steps['preprocess'].correlation_threshold,0.9)
    def test_nest_and_oof(self):
        res=run_nested_cv({'Pool1':self.df.drop(columns='hidden_missing')},self.y,
                          families=['SVC'],seeds=[42],outer_folds=3,inner_folds=2,fast=True)
        self.assertEqual(len(res),1)
        self.assertEqual(int(np.sum(res[0].confusion)),24)
        self.assertTrue((res[0].oof_predictions>=0).all())
        self.assertTrue(0<=res[0].macro_f1<=1)
    def test_pair_preserving(self):
        res=run_nested_cv({'Pool1':self.df.drop(columns='hidden_missing')},self.y,
                          families=['SVC'],seeds=[42],outer_folds=3,inner_folds=2,
                          groups=self.groups,fast=True)
        for tr,te in res[0].splits:
            self.assertFalse(set(self.groups[list(tr)]) & set(self.groups[list(te)]))
    def test_paired_outer_splits(self):
        res=run_nested_cv({'Pool1':self.df.drop(columns='hidden_missing'),
                           'Pool2':self.df.drop(columns='hidden_missing').assign(extra=np.arange(24))},self.y,
                          families=['SVC'],seeds=[42],outer_folds=3,inner_folds=2,fast=True)
        self.assertEqual(res[0].splits,res[1].splits)
        self.assertEqual(len(summarize_results(res)),2)
    def test_one_seed_sd_nan(self):
        res=run_nested_cv({'Pool1':self.df.drop(columns='hidden_missing')},self.y,
                          families=['SVC'],seeds=[42],outer_folds=3,inner_folds=2,fast=True)
        self.assertTrue(np.isnan(summarize_results(res)['macro_f1_sd'].iloc[0]))
    def test_invalid_pair(self):
        groups=self.groups.copy();groups[0]=999
        with self.assertRaises(V2ValidationError):
            run_nested_cv({'Pool1':self.df},self.y,groups=groups,fast=True)

if __name__=='__main__':unittest.main()
