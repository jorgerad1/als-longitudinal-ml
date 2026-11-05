"""Runs a strictly fictitious, small nested-CV smoke test. Never reads clinical files."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from nested_cv_v2 import run_nested_cv, summarize_results

def main():
    rng=np.random.default_rng(42)
    n=96
    y=np.array([0,1]*(n//2))
    p1=pd.DataFrame({'synthetic_age':rng.normal(55,10,n),'synthetic_marker':rng.normal(0,1,n)+y*0.2})
    p2=p1.assign(synthetic_additional=rng.normal(0,1,n)+y*.1)
    p3=p2.assign(T12_CHANGED_PESO=rng.integers(0,2,size=n))
    r=run_nested_cv({'Pool1':p1,'Pool2':p2,'Pool3':p3},y,
      families=('LogReg','SVC','RF'),seeds=[42],outer_folds=3,inner_folds=2,fast=True)
    s=summarize_results(r)
    assert len(s)==9 and all(s.seeds==1)
    assert all(int(np.sum(x.confusion))==n for x in r)
    print('STAGE79_SYNTHETIC_NESTED_CV_DEMO_OK: 3 families x 3 pools; 1 seed; 3x2 CV; 0 clinical subjects')
    print('NO_API=1 NO_CLINICAL_DATA=1 PUBLICATION_READY=False')

if __name__=='__main__':main()
