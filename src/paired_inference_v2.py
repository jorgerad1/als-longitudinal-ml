"""Experimental publication v2.0.0 paired inference on *in-memory* OOF predictions.

Synthetic-first implementation. The *exact* algorithmic parity with private
stage56B/53/54/55 inference code is NOT established. No filesystem, Git,
network or OpenAI I/O. Never serialize clinical OOF predictions.

The repeated outer CV creates repeated *predictions* for each participant, not
independent participants. All resampling/permutation operations operate at the
person level (or matched-pair level), preserving all repeated predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import numpy as np


class InferenceV2Error(ValueError):
    pass


@dataclass(frozen=True)
class ContrastV2:
    classifier: str
    first: str
    second: str
    unique_participants: int
    repetitions: int
    pooled_delta_macro_f1: float
    conditional_ci_lower: float
    conditional_ci_upper: float
    paired_permutation_p: float
    holm_adjusted_p: float
    bootstrap_resamples: int
    permutation_resamples: int


def _binary_array(x, name):
    arr = np.asarray(x)
    if arr.ndim != 1 or not len(arr) or not np.isin(arr,[0,1]).all():
        raise InferenceV2Error(f'{name} must be 1D binary 0/1')
    return arr.astype(np.int8, copy=False)


def _matrix(raw, n, name):
    arr=np.asarray(raw)
    if arr.ndim!=2 or arr.shape[1]!=n or arr.shape[0]<1 or not np.isin(arr,[0,1]).all():
        raise InferenceV2Error(f'{name} must be [seeds, participants] binary 0/1')
    return arr.astype(np.int8,copy=False)


def _counts_by_subject(labels:np.ndarray, predictions:np.ndarray):
    """Each row sums TN,FP,FN,TP over *all seeds* for one person."""
    positive=predictions.sum(axis=0,dtype=np.int64)
    zero=predictions.shape[0]-positive
    return np.column_stack((np.where(labels==0,zero,0),
                            np.where(labels==0,positive,0),
                            np.where(labels==1,zero,0),
                            np.where(labels==1,positive,0))).astype(np.int64)


def _f1_macro_from_counts(c):
    """Vectorized binary macro-F1; follow sklearn's zero_division=0 convention."""
    c=np.asarray(c)
    tn,fp,fn,tp=(c[...,i].astype(float) for i in range(4))
    d0=2*tn+fp+fn
    d1=2*tp+fp+fn
    score0=np.divide(2*tn,d0,out=np.zeros_like(tn),where=d0!=0)
    score1=np.divide(2*tp,d1,out=np.zeros_like(tp),where=d1!=0)
    return 0.5*(score0+score1)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values for one prespecified comparison family."""
    p=np.asarray(p_values,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise InferenceV2Error('Invalid permutation p-values')
    order=np.argsort(p,kind='stable')
    ordered=(len(p)-np.arange(len(p)))*p[order]
    adjusted=np.minimum(1.0,np.maximum.accumulate(ordered))
    out=np.empty(len(p),dtype=float);out[order]=adjusted
    return out


def _resampling_units(y,groups=None):
    n=len(y)
    if groups is None:
        # Patient-stratified bootstrap by outcome; paired permutation per patient.
        strat=[np.flatnonzero(y==value) for value in (0,1)]
        if any(len(ix)==0 for ix in strat):
            raise InferenceV2Error('Both outcome classes required')
        perm_units=[[int(i)] for i in range(n)]
        return strat,perm_units
    arr=np.asarray(groups)
    if arr.ndim!=1 or len(arr)!=n:
        raise InferenceV2Error('Matched pair IDs must align with participants')
    # Avoid printing group IDs, which can be sensitive in some data sources.
    from collections import defaultdict
    by_group=defaultdict(list)
    for i,g in enumerate(arr.tolist()):
        try:
            if g is None or g!=g:
                raise InferenceV2Error('Missing group identifiers')
            by_group[g].append(i)
        except TypeError as exc:
            raise InferenceV2Error('Unhashable group identifier') from exc
    units=list(by_group.values())
    if any(len(u)!=2 or sorted(y[u].tolist())!=[0,1] for u in units):
        raise InferenceV2Error('Each matched pair must contain one control and one case')
    return [np.arange(len(units),dtype=int)],units


def _single_contrast(y,pa,pb,*,groups,bootstrap_resamples,permutation_resamples,seed):
    n=len(y)
    ac=_counts_by_subject(y,pa);bc=_counts_by_subject(y,pb)
    observed=float(_f1_macro_from_counts(ac.sum(axis=0))-_f1_macro_from_counts(bc.sum(axis=0)))
    strata,units=_resampling_units(y,groups)
    rng=np.random.default_rng(seed)
    if groups is None:
        sampled=np.concatenate([
            strata_ids[rng.integers(0,len(strata_ids),size=(bootstrap_resamples,len(strata_ids)))]
            for strata_ids in strata],axis=1)
        chosen_a=ac[sampled].sum(axis=1);chosen_b=bc[sampled].sum(axis=1)
    else:
        unit_a=np.stack([ac[ix].sum(axis=0) for ix in units])
        unit_b=np.stack([bc[ix].sum(axis=0) for ix in units])
        sampled=rng.integers(0,len(units),size=(bootstrap_resamples,len(units)))
        chosen_a=unit_a[sampled].sum(axis=1);chosen_b=unit_b[sampled].sum(axis=1)
    deltas=_f1_macro_from_counts(chosen_a)-_f1_macro_from_counts(chosen_b)
    lo,hi=map(float,np.quantile(deltas,[0.025,0.975]))

    # Paired permutation: the entire prediction trajectory (all seeds) for each
    # participant, or both members of a matched pair, is swapped as one unit.
    unit_a=np.stack([ac[ix].sum(axis=0) for ix in units])
    unit_b=np.stack([bc[ix].sum(axis=0) for ix in units])
    swap=rng.integers(0,2,size=(permutation_resamples,len(units)),dtype=np.int8)
    # A is replaced by B on swapped units, and vice versa.
    base_a=unit_a.sum(axis=0);base_b=unit_b.sum(axis=0)
    shift=(unit_b-unit_a)
    summed_shift=swap @ shift
    null_delta=_f1_macro_from_counts(base_a+summed_shift)-_f1_macro_from_counts(base_b-summed_shift)
    p=float((1+np.count_nonzero(np.abs(null_delta)>=abs(observed)-1e-12))/(permutation_resamples+1))
    return observed,lo,hi,p


def compare_pool_predictions(y, predictions:Mapping[tuple[str,str],np.ndarray],
                             comparisons:Sequence[tuple[str,str]],*,
                             groups=None, bootstrap_resamples:int=2000,
                             permutation_resamples:int=2000,random_state:int=20261105,
                             require_nine:bool=True):
    """Run within-family paired pool comparisons with patient-/pair-level resampling.

    `predictions[(classifier,pool)]` = binary [n_seed,n_participant] array,
    *same individuals in the same order and same outer split* across pools.
    Caller must ensure input alignment and split identity; use `from_model_results`
    for a checked adapter from Stage79 objects.

    Returns aggregate contrast records. No data are stored in a file.
    """
    y=_binary_array(y,'labels');n=len(y)
    if len(np.unique(y))!=2:raise InferenceV2Error('Both classes required')
    if type(bootstrap_resamples) is not int or bootstrap_resamples<20:
        raise InferenceV2Error('bootstrap_resamples must be an integer >= 20')
    if type(permutation_resamples) is not int or permutation_resamples<20:
        raise InferenceV2Error('permutation_resamples must be an integer >= 20')
    if not comparisons or len(set(comparisons))!=len(comparisons) or any(a==b for a,b in comparisons):
        raise InferenceV2Error('Expected unique nonidentical comparison pairs')
    if not predictions:raise InferenceV2Error('Empty predictions')
    if any(not isinstance(k,tuple) or len(k)!=2 for k in predictions):
        raise InferenceV2Error('Prediction keys must be (family,pool) tuples')
    parsed={k:_matrix(v,n,str(k)) for k,v in predictions.items()}
    seed_counts={v.shape[0] for v in parsed.values()}
    if len(seed_counts)!=1:raise InferenceV2Error('Pools have different number of seeds')
    n_seeds=seed_counts.pop()
    families=sorted({fam for fam,_ in parsed})
    if require_nine and len(families)*len(comparisons)!=9:
        raise InferenceV2Error('Expected nine contrasts (3 families x 3 comparisons)')
    if groups is not None:_resampling_units(y,groups)
    observed=[];pvals=[]
    for family in families:
        for a,b in comparisons:
            if (family,a) not in parsed or (family,b) not in parsed:
                raise InferenceV2Error('Missing family/pool prediction for contrast')
            label=f'{random_state}:{family}:{a}:{b}'.encode('utf8')
            seed=int.from_bytes(sha256(label).digest()[:8],'big')
            delta,lo,hi,p=_single_contrast(y,parsed[(family,a)],parsed[(family,b)],
                groups=groups,bootstrap_resamples=bootstrap_resamples,
                permutation_resamples=permutation_resamples,seed=seed)
            observed.append((family,a,b,delta,lo,hi,p));pvals.append(p)
    adjusted=holm_adjust(pvals)
    return [ContrastV2(fam,a,b,n,n_seeds,delta,lo,hi,p,float(adj),bootstrap_resamples,permutation_resamples)
            for (fam,a,b,delta,lo,hi,p),adj in zip(observed,adjusted)]


def from_model_results(results,y,*,comparisons:Sequence[tuple[str,str]],groups=None,
                       bootstrap_resamples=2000,permutation_resamples=2000,random_state=20261105,
                       require_nine=True):
    """Adapter from Stage79 `ModelResult` sequences. Checks same outer splits.

    Each ModelResult contains an in-memory one-seed 1D OOF prediction vector;
    only aggregate ContrastV2 records are returned.
    """
    if not results:raise InferenceV2Error('No model results')
    y=_binary_array(y,'labels')
    grouped={};splits={}
    for r in results:
        key=(str(r.family),str(r.pool));seed=int(r.seed)
        if seed in grouped.setdefault(key,{}):
            raise InferenceV2Error('Duplicate seed for family/pool')
        grouped[key][seed]=_binary_array(r.oof_predictions,'model OOF')
        if len(grouped[key][seed])!=len(y):raise InferenceV2Error('OOF length mismatch')
        canonical=repr(r.splits)
        if seed in splits and splits[seed]!=canonical:
            raise InferenceV2Error('Outer splits differ across pools/families')
        splits[seed]=canonical
    expected_seeds=None;parsed={}
    for key,by_seed in grouped.items():
        these=tuple(sorted(by_seed))
        if expected_seeds is None:expected_seeds=these
        if these!=expected_seeds:raise InferenceV2Error('Seed sets do not match')
        parsed[key]=np.stack([by_seed[s] for s in these],axis=0)
    return compare_pool_predictions(y,parsed,comparisons,groups=groups,
          bootstrap_resamples=bootstrap_resamples,permutation_resamples=permutation_resamples,
          random_state=random_state,require_nine=require_nine)


def summary_rows(contrasts:Sequence[ContrastV2]):
    """Produce public-safe aggregate dictionary rows (not individual predictions)."""
    from dataclasses import asdict
    return [asdict(c) for c in contrasts]
