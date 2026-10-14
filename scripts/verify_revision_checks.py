"""Audit revision-stage numerical artifacts against archived scores and seals."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
CHECK=ROOT/'results/submission_checks'
CONTROL=ROOT/'results/lp_convergence_control'

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

def main():
    cohort_counts={}
    for ds,expected in [('BUSBRA',{0:722,1:342}),('BREAST',{0:154,1:98}),('BUSI',{0:437,1:210})]:
        mf=pd.read_csv(ROOT/'data/manifests'/f'{ds}.csv')
        if ds=='BUSBRA':
            assert mf.groupby('patient_key').y.nunique().max()==1
            mf=mf.drop_duplicates('patient_key')
        counts=mf.y.value_counts().to_dict();assert counts==expected
        cohort_counts[ds]={'benign':counts[0],'malignant':counts[1]}
    metrics=pd.read_csv(CHECK/'primary_metrics_recomputed.csv',float_precision='round_trip')
    count=0
    for regime,file in [('LP','LINEAR_PROBE_TARGET_SCORES_RECOMPUTED.csv'),('FT','FULLFT_EXTERNAL_TARGET_SCORES.csv')]:
        d=pd.read_csv(ROOT/'results/archived/FULLSTUDY_FINAL_FULLFT_RESULTS'/file,
                      dtype={'unit_id':str},float_precision='round_trip')
        for k,g in d[d.source!=d.target].groupby(['model','source','seed','target']):
            m=metrics[(metrics.regime==regime)&(metrics.model==k[0])&
                      (metrics.source==k[1])&(metrics.seed==k[2])&(metrics.target==k[3])]
            assert len(m)==1 and not g.unit_id.duplicated().any()
            close(m.iloc[0].auc,roc_auc_score(g.y,g.score))
            close(m.iloc[0].ap,average_precision_score(g.y,g.score));count+=1
    assert count==72 and len(metrics)==72

    loo=pd.read_csv(CHECK/'leave_one_seed_out.csv',float_precision='round_trip')
    for _,r in loo.iterrows():
        m=metrics[(metrics.source==r.source)&(metrics.target==r.target)&(metrics.seed!=r.omitted_seed)]
        p=m.pivot(index='seed',columns=['regime','model'],values='auc');assert len(p)==2
        close(r.lp_margin,(p['LP',r.usfm]-p['LP','MAE']).mean())
        close(r.ft_margin,(p['FT',r.usfm]-p['FT','MAE']).mean())
        assert bool(r.reversal)==bool(r.lp_margin>0 and r.ft_margin<0)
    assert len(loo)==24
    history=pd.read_csv(CHECK/'lp_histories.csv')
    assert len(history.groupby(['representation','source','seed']))==18
    for _,g in history.groupby(['representation','source','seed']):
        assert g.epoch.tolist()==list(range(1,31))
    hist=pd.read_csv(CHECK/'lp_training_budget_diagnostics.csv')
    assert (hist.best_epoch==30).sum()==11 and (hist.best_epoch>=28).sum()==15

    seal=json.loads((CONTROL/'selection_seal.json').read_text())
    for name,key in [('protocol_locked_before_fit.json','protocol_sha256'),
                     ('source_selected_head_parameters.json','selected_parameters_sha256'),
                     ('source_grid_results.csv','source_grid_sha256')]:
        assert hashlib.sha256((CONTROL/name).read_bytes()).hexdigest()==seal[key]
    grid=pd.read_csv(CONTROL/'source_grid_results.csv',float_precision='round_trip')
    assert len(grid)==42 and not grid.convergence_warning.any()
    assert grid.iterations.max()<5000 and grid.objective_gradient_inf.max()<1e-6
    selected=pd.read_csv(CONTROL/'source_selected_heads.csv',float_precision='round_trip')
    assert len(selected)==6
    for _,r in selected.iterrows():
        g=grid[(grid.model==r.model)&(grid.source==r.source)]
        best=g.sort_values(['source_validation_auc','C'],ascending=[False,True]).iloc[0]
        close(r.C,best.C);close(r.validation_auc,best.source_validation_auc)
    scores=pd.read_csv(CONTROL/'external_target_scores.csv',float_precision='round_trip',dtype={'unit_id':str})
    results=pd.read_csv(CONTROL/'external_results.csv',float_precision='round_trip')
    for _,r in results.iterrows():
        g=scores[(scores.model==r.model)&(scores.source==r.source)&(scores.target==r.target)]
        assert len(g)==r.target_units and not g.unit_id.duplicated().any()
        close(r.auc,roc_auc_score(g.y,g.score));close(r.ap,average_precision_score(g.y,g.score))
    assert len(results)==12
    margins=pd.read_csv(CONTROL/'external_margins.csv',float_precision='round_trip')
    for _,r in margins.iterrows():
        g=results[(results.source==r.source)&(results.target==r.target)].set_index('model')
        close(r.margin,g.loc[r.usfm,'auc']-g.loc['MAE','auc'])
    positives=margins.groupby('usfm').margin.apply(lambda x:int((x>0).sum())).to_dict()
    assert positives=={'URFM':4,'UltraFedFM':1}
    cache=pd.read_csv(CHECK/'feature_cache_head_correspondence.csv',float_precision='round_trip')
    assert len(cache)==12 and (cache.seed==42).all()
    assert cache.max_absolute_score_error.max()<2e-5
    close(cache.auc_difference,0);close(cache.ap_difference,0)
    for _,r in cache.iterrows():
        digest=hashlib.sha256((ROOT/'data/manifests'/f'{r.target}.csv').read_bytes()).hexdigest()
        assert digest==r.manifest_sha256
    report={'status':'PASS','cohort_unit_counts_verified':cohort_counts,
            'primary_evaluations_verified':count,'primary_metric_scalars':count*2,
            'seed_omissions_verified':len(loo),'complete_LP_histories':18,
            'source_candidate_fits_verified':42,'selected_source_heads':6,
            'control_external_evaluations_verified':12,'control_positive_margins':positives,
            'feature_cache_head_metric_pairs_verified':12,'source_selection_seal_verified':True}
    (ROOT/'audit/revision_verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
