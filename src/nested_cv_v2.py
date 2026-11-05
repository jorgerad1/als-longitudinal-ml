"""Portable, synthetic-first nested-CV engine for the paperNLP v2 release draft.

This module encodes the documented training-only preprocessing, nested CV,
three estimator families and paired outer splits. It does NOT reproduce
previously reported clinical results or the historical exact tuning snapshots.
The latter require source-code parity verification on the private server.

No data are read, written, or sent to network by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FORCED_NUMERIC_PREFIXES = ('T12_CHANGED_', 'T12_RATE_CHANGED_', 'T12_METRIC_DELTA_')
FORCED_NUMERIC_EXACT = {'TXT_HABIT_COUNT', 'TXTG_N_GROUPS'}


class V2ValidationError(ValueError):
    pass


class TrainOnlyPreprocessing(BaseEstimator, TransformerMixin):
    """Learns every filter and encoding rule from the current training split."""
    def __init__(self, missingness_max=0.20, support_min=3,
                 correlation_threshold=0.90, correlation_filter=True):
        self.missingness_max = missingness_max
        self.support_min = support_min
        self.correlation_threshold = correlation_threshold
        self.correlation_filter = correlation_filter

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise V2ValidationError('Expected a pandas DataFrame (column names required)')
        if X.columns.has_duplicates:
            raise V2ValidationError('Duplicate predictor column names')
        if len(X) == 0:
            raise V2ValidationError('Empty training partition')
        self.input_columns_ = list(X.columns)
        self.keep_columns_ = list(X.columns[X.isna().mean().le(self.missingness_max)])
        if not self.keep_columns_:
            raise V2ValidationError('All features failed train-only missingness test')
        self.kinds_ = {}
        for col in self.keep_columns_:
            s = X[col].dropna()
            if col.startswith(FORCED_NUMERIC_PREFIXES) or col in FORCED_NUMERIC_EXACT:
                kind = 'numeric'
            elif not pd.api.types.is_numeric_dtype(s):
                kind = 'category'
            else:
                unique = s.nunique(dropna=True)
                kind = 'binary' if unique <= 2 else 'category' if unique <= 15 else 'numeric'
            self.kinds_[col] = kind
        num = [n for n in self.keep_columns_ if self.kinds_[n]=='numeric']
        binary = [n for n in self.keep_columns_ if self.kinds_[n]=='binary']
        cat = [n for n in self.keep_columns_ if self.kinds_[n]=='category']
        # sklearn's ColumnTransformer concatenates in the same deterministic order.
        transformers=[]
        if binary:
            transformers.append(('binary',SimpleImputer(strategy='most_frequent'),binary))
        if cat:
            transformers.append(('category',Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(sparse_output=False,handle_unknown='infrequent_if_exist',min_frequency=3)),
            ]),cat))
        if num:
            transformers.append(('numeric',Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler()),
            ]),num))
        self.encoder_ = ColumnTransformer(transformers, remainder='drop',sparse_threshold=0)
        train = self._clean(X)
        raw = np.asarray(self.encoder_.fit_transform(train),dtype=float)
        self.encoded_feature_names_ = list(self.encoder_.get_feature_names_out())
        if not np.isfinite(raw).all():
            raise V2ValidationError('Nonfinite numeric encodings')
        # Variance then low non-zero support. Keep deterministic input order.
        variance_ok = np.var(raw,axis=0)>0
        support_ok = np.count_nonzero(raw,axis=0)>=self.support_min
        retained = np.flatnonzero(variance_ok & support_ok)
        if not len(retained):
            raise V2ValidationError('No encoded features after variance/support filters')
        self.pre_correlation_count_ = len(retained)
        corr_removed=[]
        if self.correlation_filter and len(retained)>1:
            # Drop later encoded column of each highly correlated pair.
            survivor=[]
            for idx in retained:
                if not survivor:
                    survivor.append(idx)
                    continue
                a = raw[:,idx]
                sd_a = np.std(a)
                similar = False
                for prev in survivor:
                    b = raw[:,prev]
                    coeff = float(np.mean((a-a.mean())*(b-b.mean()))/(sd_a*np.std(b)))
                    if abs(coeff)>self.correlation_threshold:
                        similar=True
                        break
                if similar:
                    corr_removed.append(idx)
                else:
                    survivor.append(idx)
            retained=np.asarray(survivor,dtype=int)
        self.selected_indices_=retained.astype(int)
        self.correlation_removed_count_=len(corr_removed)
        self.selected_feature_names_=[self.encoded_feature_names_[i] for i in self.selected_indices_]
        return self

    def _clean(self,X):
        if not isinstance(X,pd.DataFrame):
            raise V2ValidationError('Expected DataFrame')
        missing=set(self.keep_columns_) - set(X.columns)
        if missing:
            raise V2ValidationError('Test partition missing training columns')
        out=X.loc[:,self.keep_columns_].copy()
        for name,kind in self.kinds_.items():
            if kind in {'numeric','binary'}:
                out[name]=pd.to_numeric(out[name],errors='coerce')
        return out

    def transform(self,X):
        if not hasattr(self,'selected_indices_'):
            raise V2ValidationError('Not yet fitted')
        raw=np.asarray(self.encoder_.transform(self._clean(X)),dtype=float)
        if not np.isfinite(raw).all():
            raise V2ValidationError('Nonfinite transformed values')
        return raw[:,self.selected_indices_]


@dataclass(frozen=True)
class ModelResult:
    family: str
    pool: str
    seed: int
    n_subjects: int
    macro_f1: float
    als_f1: float
    als_recall: float
    mcc: float
    confusion: tuple[tuple[int,int],tuple[int,int]]
    oof_predictions: np.ndarray
    splits: tuple[tuple[tuple[int,...],tuple[int,...]],...]


def model_search(family: str, *, inner_cv, seed: int, fast: bool=False,
                 no_correlation_filter: bool=False):
    """Return fitted-search specification; all preprocessing lives in pipeline."""
    common_prep=TrainOnlyPreprocessing(correlation_filter=not no_correlation_filter)
    if family=='LogReg':
        est=LogisticRegression(solver='saga',penalty='elasticnet',max_iter=5000,
                               random_state=42)
        params={'classifier__C':[1.0] if fast else [0.001,0.01,0.1,1,10],
                'classifier__l1_ratio':[0.5] if fast else [0.5,0.8,1.0],
                'classifier__class_weight':[None] if fast else [None,'balanced']}
    elif family=='SVC':
        from sklearn.svm import LinearSVC
        est=LinearSVC(dual='auto',max_iter=5000,random_state=42)
        params={'classifier__C':[1.0] if fast else [0.001,0.01,0.1,1,10],
                'classifier__class_weight':[None] if fast else [None,'balanced']}
    elif family=='RF':
        est=RandomForestClassifier(random_state=42,n_jobs=1)
        params={
            'classifier__n_estimators':[20] if fast else [100,200,300],
            'classifier__max_depth':[3] if fast else [None,3,5,8],
            'classifier__min_samples_split':[2] if fast else [2,5,10],
            'classifier__min_samples_leaf':[1] if fast else [1,2,4,6],
            'classifier__max_features':['sqrt'] if fast else ['sqrt',0.2,0.5],
            'classifier__class_weight':[None] if fast else [None,'balanced','balanced_subsample'],
        }
    else:
        raise V2ValidationError('Unsupported classifier: '+str(family))
    pipe=Pipeline([('preprocess',common_prep),('classifier',est)])
    if family=='RF':
        return RandomizedSearchCV(pipe,params,n_iter=1 if fast else 12,
                                  scoring='f1_macro',cv=inner_cv,random_state=seed,
                                  n_jobs=1,refit=True,error_score='raise')
    return GridSearchCV(pipe,params,scoring='f1_macro',cv=inner_cv,n_jobs=1,
                        refit=True,error_score='raise')


def run_nested_cv(pools: Mapping[str,pd.DataFrame], y,
                  *, families:Sequence[str]=('LogReg','SVC','RF'),
                  seeds:Sequence[int]=tuple(range(42,52)),
                  outer_folds:int=5,inner_folds:int=3,
                  groups=None, fast:bool=False, no_correlation_filter:bool=False):
    """Run paired outer splits, optionally preserving matched pairs as groups.

    Inputs must be pre-constructed feature matrices aligned by ROW POSITION.
    Do not supply identifiers or outcome columns as features. No files/API involved.
    Returns results with in-memory OOF arrays (do not print real patient arrays).
    """
    if not pools:raise V2ValidationError('No pools')
    ns={len(df) for df in pools.values()}
    if len(ns)!=1:raise V2ValidationError('Pool row counts differ')
    n=ns.pop()
    ys=np.asarray(y)
    if ys.ndim!=1 or len(ys)!=n or set(np.unique(ys))!={0,1}:
        raise V2ValidationError('Binary labels 0/1 and aligned row counts required')
    if any(not isinstance(df,pd.DataFrame) or df.columns.has_duplicates for df in pools.values()):
        raise V2ValidationError('Each pool must have unique named DataFrame predictors')
    for label,df in pools.items():
        forbidden={x for x in df.columns if x.casefold() in {'id','participant_id','patient_id','label','outcome','diagnosis','diagnostico','ela','als'}}
        if forbidden:raise V2ValidationError('Identifier/outcome-like column in '+label)
    if groups is not None:
        gs=np.asarray(groups)
        if len(gs)!=n or pd.isna(gs).any():raise V2ValidationError('Invalid group IDs')
        for g in np.unique(gs):
            # The paired ALS-control protocol has exactly one of each class per pair.
            if sorted(ys[gs==g].tolist())!=[0,1]:
                raise V2ValidationError('Group must contain exactly one control and one case')
    else:gs=None
    result=[]
    for seed in seeds:
        outer=(StratifiedGroupKFold(n_splits=outer_folds,shuffle=True,random_state=seed)
               if gs is not None else StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=seed))
        splits=tuple((tuple(map(int,tr)),tuple(map(int,te))) for tr,te in outer.split(np.zeros((n,1)),ys,gs))
        for pool,df in pools.items():
            for family in families:
                oof=np.full(n,-1,dtype=int)
                for tr,te in splits:
                    tr_idx=np.asarray(tr,dtype=int);te_idx=np.asarray(te,dtype=int)
                    inner=(StratifiedGroupKFold(n_splits=inner_folds,shuffle=True,random_state=seed)
                           if gs is not None else StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=seed))
                    search=model_search(family,inner_cv=inner,seed=seed,fast=fast,
                                        no_correlation_filter=no_correlation_filter)
                    kwargs={'groups':gs[tr_idx]} if gs is not None else {}
                    with warnings.catch_warnings():
                        from sklearn.exceptions import ConvergenceWarning
                        warnings.filterwarnings('ignore',category=ConvergenceWarning)
                        warnings.filterwarnings('ignore',category=FutureWarning,message="'penalty' was deprecated.*")
                        search.fit(df.iloc[tr_idx],ys[tr_idx],**kwargs)
                    pred=search.predict(df.iloc[te_idx])
                    if (oof[te_idx]!=-1).any():raise V2ValidationError('Duplicate OOF')
                    oof[te_idx]=pred
                if np.any(oof==-1):raise V2ValidationError('Incomplete OOF')
                confusion=confusion_matrix(ys,oof,labels=[0,1])
                result.append(ModelResult(family,pool,int(seed),n,
                                          float(f1_score(ys,oof,average='macro')),
                                          float(f1_score(ys,oof,pos_label=1,zero_division=0)),
                                          float(recall_score(ys,oof,pos_label=1,zero_division=0)),
                                          float(matthews_corrcoef(ys,oof)),
                                          tuple(map(tuple,confusion.tolist())),oof,splits))
    return result


def summarize_results(results:Sequence[ModelResult]):
    """Return only aggregate metrics; no participant-level prediction arrays."""
    rows=[]
    for (pool,family),sub in sorted(_group_results(results).items()):
        vals=np.asarray([[r.macro_f1,r.als_f1,r.als_recall,r.mcc] for r in sub])
        mean=vals.mean(axis=0);sd=vals.std(axis=0,ddof=1) if len(sub)>1 else np.full(4,np.nan)
        row={'pool':pool,'family':family,'seeds':len(sub)}
        for i,k in enumerate(['macro_f1','als_f1','als_recall','mcc']):
            row[k+'_mean']=float(mean[i]);row[k+'_sd']=float(sd[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _group_results(results):
    grouped={}
    for r in results:grouped.setdefault((r.pool,r.family),[]).append(r)
    return grouped
