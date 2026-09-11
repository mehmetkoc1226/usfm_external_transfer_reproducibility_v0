"""Deterministic RQ4 manuscript-audit reanalysis; no model training.

Run from the repository root:
    python scripts/recompute_rq4_adaptation.py

Uses the notebook's sequential seeds 101021..101028 for its adaptation block.
Accelerates AUROC via exact integer bootstrap multiplicities and tied-score groups.
Validates every score vector on the first ten draws against sklearn AUROC.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'results' / 'archived' / 'FULLSTUDY_FINAL_FULLFT_RESULTS'
OUT = ROOT / 'results' / 'recomputed'
OUT.mkdir(exist_ok=True)
SEEDS = [42, 43, 44]
SCENARIOS = [('BUSBRA','BREAST'), ('BUSBRA','BUSI'), ('BREAST','BUSBRA'), ('BREAST','BUSI')]
MODELS = ['URFM', 'UltraFedFM']
B = 3000
def read(name):
    return pd.read_csv(E/name, float_precision='round_trip', dtype={'unit_id': str})
scores = {'FT':read('FULLFT_EXTERNAL_TARGET_SCORES.csv'),
          'LP':read('LINEAR_PROBE_TARGET_SCORES_RECOMPUTED.csv')}
old = read('RQ4_ADAPTATION_CHANGE_IN_USFM_ADVANTAGE.csv')

def weighted_auc(counts, y, score):
    order = np.argsort(score, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]) != 0)+1]
    positive = np.add.reduceat(counts[:,order]*y[order], starts, axis=1)
    negative = np.add.reduceat(counts[:,order]*(1-y[order]), starts, axis=1)
    numerator = (positive*(np.cumsum(negative,axis=1)-negative/2)).sum(axis=1)
    return numerator / (positive.sum(axis=1)*negative.sum(axis=1))

rows, arrays, comparisons = [], [], []
max_validation_error = 0.0
counter = 20  # Eight FT contrasts and twelve within-model contrasts precede this block.
for model in MODELS:
    for source,target in SCENARIOS:
        vectors, ref_ids, y = [], None, None
        for seed in SEEDS:
            for method,m in [('FT',model),('FT','MAE'),('LP',model),('LP','MAE')]:
                d=scores[method]
                d=d[(d.model==m)&(d.source==source)&(d.target==target)&(d.seed==seed)].sort_values('unit_id')
                assert len(d) and not d.unit_id.duplicated().any()
                ids,labels=d.unit_id.to_numpy(),d.y.to_numpy(int)
                if ref_ids is None:
                    ref_ids,y=ids,labels
                assert np.array_equal(ref_ids,ids) and np.array_equal(y,labels)
                assert np.isfinite(d.score).all()
                vectors.append(d.score.to_numpy(float))
        points=np.array([roc_auc_score(y,s) for s in vectors]).reshape(3,4)
        seed_points=points[:,0]-points[:,1]-points[:,2]+points[:,3]
        # Preserve the notebook's arithmetic grouping for the point estimates.
        seed_points=(points[:,0]-points[:,1])-(points[:,2]-points[:,3])
        point=float(np.mean(seed_points))
        rng_seed=101001+counter
        counter+=1
        rng=np.random.default_rng(rng_seed)
        i0,i1=np.flatnonzero(y==0),np.flatnonzero(y==1)
        counts=np.empty((B,len(y)),dtype=np.int64)
        first=[]
        for b in range(B):
            idx=np.concatenate([rng.choice(i0,size=len(i0),replace=True),rng.choice(i1,size=len(i1),replace=True)])
            counts[b]=np.bincount(idx,minlength=len(y))
            if b<10:
                first.append(idx)
        all_auc=np.stack([weighted_auc(counts,y,s) for s in vectors],axis=1).reshape(B,3,4)
        direct=np.array([[roc_auc_score(y[idx],s[idx]) for s in vectors] for idx in first]).reshape(10,3,4)
        err=float(np.max(np.abs(all_auc[:10]-direct)))
        max_validation_error=max(max_validation_error,err)
        assert err<1e-12
        boot=np.mean((all_auc[:,:,0]-all_auc[:,:,1])-(all_auc[:,:,2]-all_auc[:,:,3]),axis=1)
        previous=old[(old.usfm==model)&(old.source==source)&(old.target==target)].iloc[0]
        assert abs(point-previous.point)<1e-12
        row=dict(usfm=model,source=source,target=target,point=float(previous.point),
                 seed42_change=float(previous.seed42_change),seed43_change=float(previous.seed43_change),
                 seed44_change=float(previous.seed44_change),bootstrap_median=float(np.median(boot)),
                 ci_lo=float(np.quantile(boot,.025)),ci_hi=float(np.quantile(boot,.975)),
                 p_change_positive=float(np.mean(boot>0)))
        rows.append(row)
        arrays.append(pd.DataFrame(dict(usfm=model,source=source,target=target,
                     bootstrap_rep=np.arange(B),adaptation_change_in_usfm_advantage=boot)))
        comparisons.append(dict(usfm=model,source=source,target=target,rng_seed=rng_seed,
            old_ci_lo=previous.ci_lo,new_ci_lo=row['ci_lo'],old_ci_hi=previous.ci_hi,new_ci_hi=row['ci_hi'],
            old_support=previous.p_change_positive,new_support=row['p_change_positive'],
            interval_sign_unchanged=bool((previous.ci_lo>0)==(row['ci_lo']>0) and (previous.ci_hi<0)==(row['ci_hi']<0))))
        print(model,source,target,'CI',row['ci_lo'],row['ci_hi'],'support',row['p_change_positive'],flush=True)
pd.DataFrame(rows).to_csv(OUT/'RQ4_ADAPTATION_CHANGE_IN_USFM_ADVANTAGE.csv',index=False)
pd.concat(arrays).to_csv(OUT/'RQ4_ADAPTATION_CHANGE_BOOTSTRAP.csv',index=False)
pd.DataFrame(comparisons).to_csv(OUT/'old_vs_recomputed.csv',index=False)
metadata=dict(purpose='Independent manuscript-audit reanalysis from archived target-unit scores',
    repetitions=B,rng='numpy.random.default_rng',seeds=list(range(101021,101029)),
    scenario_order=SCENARIOS,model_order=MODELS,target_order='unit_id string ascending',
    pairing='same pathology-stratified indices across two models, three seeds, LP and FT within each contrast',
    validation='first ten replicates of all 8 contrasts, 12 AUROCs per replicate, against sklearn',
    validated_auc_count=960,max_auc_validation_error=max_validation_error,
    numpy_version=np.__version__,pandas_version=pd.__version__,sklearn_version=sklearn.__version__,
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.glob('*.csv'))})
(OUT/'reanalysis_metadata.json').write_text(json.dumps(metadata,indent=2))
assert all(r['interval_sign_unchanged'] for r in comparisons)
print('PASS: all point estimates and interval sign classifications preserved; max AUROC validation error',max_validation_error)
