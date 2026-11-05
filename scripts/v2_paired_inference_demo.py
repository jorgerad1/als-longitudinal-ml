#!/usr/bin/env python3
"""Demonstrate aggregate conditional inference using wholly invented predictions."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from paired_inference_v2 import compare_pool_predictions

def main():
    # These synthetic predictions are not learned or sampled from real patients.
    labels=np.tile(np.array([0,1],dtype=int),18)
    p={}
    for fam in ('LogReg','SVC','RF'):
        for pool,period in [('Pool1',3),('Pool2',5),('Pool3',11)]:
            preds=np.broadcast_to(labels,(3,len(labels))).copy()
            preds[:,::period]=1-preds[:,::period]
            p[fam,pool]=preds
    cs=compare_pool_predictions(labels,p,[('Pool2','Pool1'),('Pool3','Pool1'),('Pool3','Pool2')],
             bootstrap_resamples=100,permutation_resamples=100)
    assert len(cs)==9
    print('STAGE80_SYNTHETIC_INFERENCE_DEMO_OK: 9 paired contrasts; 100 bootstrap + 100 paired swaps; no clinical results')
if __name__=='__main__':main()
