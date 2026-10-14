"""Recompute primary metrics, leave-one-seed summaries and LP history diagnostics.
No fitting, target-based model selection or changes to original result packages.
"""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/archived';OUT=ROOT/'results/submission_checks';OUT.mkdir(exist_ok=True)
E=A/'FULLSTUDY_FINAL_FULLFT_RESULTS'
metrics=[];hashes={}
for regime,name in [('LP','LINEAR_PROBE_TARGET_SCORES_RECOMPUTED.csv'),('FT','FULLFT_EXTERNAL_TARGET_SCORES.csv')]:
 d=pd.read_csv(E/name,dtype={'unit_id':str},float_precision='round_trip')
 hashes[name]=hashlib.sha256((E/name).read_bytes()).hexdigest()
 d=d[d.source!=d.target]
 for keys,g in d.groupby(['model','source','seed','target']):
  assert not g.unit_id.duplicated().any() and np.isfinite(g.score).all()
  metrics.append(dict(regime=regime,model=keys[0],source=keys[1],seed=keys[2],target=keys[3],
   auc=roc_auc_score(g.y,g.score),ap=average_precision_score(g.y,g.score),target_units=len(g)))
m=pd.DataFrame(metrics);assert len(m)==72
m.to_csv(OUT/'primary_metrics_recomputed.csv',index=False)
paired=[];loo=[];summary=[]
for model in ['URFM','UltraFedFM']:
 for (source,target),group in m.groupby(['source','target']):
  p=group.pivot(index='seed',columns=['regime','model'],values='auc')
  lp=p[('LP',model)]-p[('LP','MAE')];ft=p[('FT',model)]-p[('FT','MAE')]
  for seed in [42,43,44]:
   paired.append(dict(usfm=model,source=source,target=target,seed=seed,lp_margin=lp[seed],ft_margin=ft[seed],change=ft[seed]-lp[seed]))
   loo.append(dict(usfm=model,source=source,target=target,omitted_seed=seed,
       lp_margin=lp.drop(seed).mean(),ft_margin=ft.drop(seed).mean(),
       reversal=bool(lp.drop(seed).mean()>0 and ft.drop(seed).mean()<0)))
  slp=np.array([lp.drop(s).mean() for s in lp.index]);sft=np.array([ft.drop(s).mean() for s in ft.index])
  summary.append(dict(usfm=model,source=source,target=target,lp_full=lp.mean(),ft_full=ft.mean(),
     lp_loo_min=slp.min(),lp_loo_max=slp.max(),ft_loo_min=sft.min(),ft_loo_max=sft.max(),
     lp_positive_seed_count=int((lp>0).sum()),ft_positive_seed_count=int((ft>0).sum()),
     loo_reversal_count=int(((slp>0)&(sft<0)).sum())))
pd.DataFrame(paired).to_csv(OUT/'seed_specific_margins.csv',index=False)
pd.DataFrame(loo).to_csv(OUT/'leave_one_seed_out.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'leave_one_seed_out_summary.csv',index=False)
hist=[];status=[]
folders=['FM_EXTERNAL_TRANSFER_PHASE0_RESULTS','ULTRAFEDFM_PHASE0B_RESULTS']
for folder in folders:
 for path in sorted((A/folder).glob('*_history.csv')):
  if path.name.startswith('DINO'):continue
  d=pd.read_csv(path)
  assert len(d)==30 and d.epoch.tolist()==list(range(1,31))
  hist.append(d)
  b=d.loc[d.source_val_auc.idxmax()]
  status.append(dict(representation=b.representation,source=b.source,seed=int(b.seed),best_epoch=int(b.epoch),
    source_val_auc=float(b.source_val_auc),first_train_loss=float(d.train_loss.iloc[0]),
    last_train_loss=float(d.train_loss.iloc[-1]),last5_mean_train_loss=float(d.train_loss.iloc[-5:].mean()),
    preceding5_mean_train_loss=float(d.train_loss.iloc[-10:-5].mean())))
h=pd.concat(hist);s=pd.DataFrame(status);assert len(s)==18
h.to_csv(OUT/'lp_histories.csv',index=False);s.to_csv(OUT/'lp_training_budget_diagnostics.csv',index=False)
metadata={'primary_external_evaluations':72,'metric_scalars':144,
 'loo_diagnostic':'Post-hoc descriptive omission of each of 3 completed seeds; not a confidence interval or population jackknife',
 'lp_heads':18,'best_epoch_30':int((s.best_epoch==30).sum()),'best_epoch_28_to_30':int((s.best_epoch>=28).sum()),
 'same_source_split':True,'primary_scores_sha256':hashes,'new_encoder_training':False,
 'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
print(pd.DataFrame(summary).round(6).to_string(index=False))
