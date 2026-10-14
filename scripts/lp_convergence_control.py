"""Post-hoc convergence-based frozen-head sensitivity analysis.

Run select first, then evaluate. The candidate grid and tie rule are fixed below.
Source selection never reads BUSI features or external scores. Raw feature arrays
are supplied separately; they are not redistributed with this repository.
"""
import argparse, hashlib, json, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
MODELS={'URFM':'URFM_224','UltraFedFM':'UltraFedFM_224','MAE':'ImageNet_MAE_224'}
SOURCES=['BUSBRA','BREAST']
SCENARIOS=[('BUSBRA','BREAST'),('BUSBRA','BUSI'),('BREAST','BUSBRA'),('BREAST','BUSI')]
C_GRID=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0]
CONFIG={'analysis':'post-hoc frozen-head sensitivity; not an original RQ or preregistration',
 'solver':'sklearn LogisticRegression lbfgs','penalty':'L2','C_grid':C_GRID,
 'max_iter':5000,'tol':1e-8,'feature_scaling':'StandardScaler fit on source-training units only',
 'selection':'maximum source-validation AUROC; exact ties prefer smallest C',
 'class_weights':'N/(2*n_c), normalized to mean one; source-training counts only',
 'analysis_units':'BUSBRA mean image features by patient; BREAST case; BUSI image',
 'source_splits':'unchanged archived seed-42 splits','training_random_state':42,
 'target_selection':False,'training_runs_per_model_source':1,
 'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
             'scipy':scipy.__version__,'sklearn':sklearn.__version__}}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_units(features,model,dataset):
    mf=ROOT/'data/manifests'/f'{dataset}.csv'
    meta=pd.read_csv(mf)
    assert np.array_equal(meta.manifest_row,np.arange(len(meta))), 'Manifest row order invalid'
    stem=MODELS[model]+'_'+dataset
    path=features/(stem+'_features.npy');done=features/(stem+'_done.npy')
    X=np.load(path,allow_pickle=False);complete=np.load(done,allow_pickle=False)
    assert X.dtype==np.float32 and X.shape==(len(meta),768)
    assert complete.shape==(len(meta),) and complete.dtype==bool and complete.all()
    assert np.isfinite(X).all()
    # Original phase-0 notebook concatenated BUSI train rows then val rows
    # before feature extraction. The public manifest preserves lexical image
    # order; do not attach it directly to the concatenated feature cache.
    if dataset=='BUSI':
        assert set(meta['split'])=={'train','val'}
        meta=pd.concat([meta[meta['split']=='train'],meta[meta['split']=='val']],ignore_index=True)
    if dataset=='BUSBRA':
        rows=[];vectors=[]
        for (pid,y),inds in meta.groupby(['patient_key','y'],sort=True).indices.items():
            rows.append({'unit_id':str(pid),'y':int(y)})
            vectors.append(X[np.asarray(inds,dtype=int)].mean(axis=0))
        m=pd.DataFrame(rows);X=np.stack(vectors).astype(np.float32)
    else:
        key='patient_key' if dataset=='BREAST' else 'image_id'
        m=pd.DataFrame({'unit_id':meta[key].astype(str),'y':meta.y.astype(int)})
    assert not m.unit_id.duplicated().any()
    return m,X,{'feature':sha(path),'done':sha(done),'manifest':sha(mf)}

def select(features,out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'protocol_locked_before_fit.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    heads=[];fits=[]
    with threadpool_limits(limits=1):
      for model in MODELS:
        for source in SOURCES:
          meta,X,input_hashes=load_units(features,model,source)
          split_path=ROOT/'data/splits'/f'{source}_PHASE0_SOURCE_SPLIT.csv'
          sp=pd.read_csv(split_path,dtype={'patient_key':str}).set_index('patient_key')
          assert set(sp.index)==set(meta.unit_id)
          aligned=sp.loc[meta.unit_id]
          assert np.array_equal(aligned.y.to_numpy(),meta.y.to_numpy())
          tr=np.flatnonzero(aligned.source_split.to_numpy()=='train')
          va=np.flatnonzero(aligned.source_split.to_numpy()=='val')
          assert len(tr)+len(va)==len(meta) and len(set(tr)&set(va))==0
          assert (len(tr),len(va))==({'BUSBRA':(851,213),'BREAST':(201,51)}[source])
          y=meta.y.to_numpy();scaler=StandardScaler().fit(X[tr])
          Xtr=scaler.transform(X[tr]);Xva=scaler.transform(X[va])
          cnt=np.bincount(y[tr],minlength=2);w=len(tr)/(2*cnt);w=w/w.mean()
          best_auc=-np.inf;best=None
          for C in C_GRID:
            with warnings.catch_warnings(record=True) as caught:
              warnings.simplefilter('always',ConvergenceWarning)
              clf=LogisticRegression(C=C,solver='lbfgs',max_iter=CONFIG['max_iter'],tol=CONFIG['tol'],
                         class_weight={0:float(w[0]),1:float(w[1])},random_state=42)
              clf.fit(Xtr,y[tr])
            if any(issubclass(c.category,ConvergenceWarning) for c in caught):
              raise RuntimeError(f'Convergence warning: {model}/{source}/C={C}')
            pred=clf.decision_function(Xva);auc=float(roc_auc_score(y[va],pred))
            sw=w[y[tr]];res=sw*(expit(clf.decision_function(Xtr))-y[tr])/sw.sum()
            grad=Xtr.T@res+clf.coef_[0]/(C*sw.sum())
            grad_inf=float(max(np.abs(grad).max(),abs(res.sum())))
            fits.append({'model':model,'source':source,'C':C,'iterations':int(clf.n_iter_[0]),
                'source_validation_auc':auc,'source_validation_ap':float(average_precision_score(y[va],pred)),
                'objective_gradient_inf':grad_inf,'convergence_warning':False})
            if auc>best_auc:
              best_auc=auc
              best={'model':model,'source':source,'C':C,'validation_auc':auc,
                 'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),
                 'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                 'train_units':len(tr),'validation_units':len(va),'input_sha256':input_hashes,
                 'split_sha256':sha(split_path)}
          heads.append(best)
          print('Selected',model,source,'C',best['C'],'validation AUROC',best_auc,flush=True)
    pd.DataFrame(fits).to_csv(out/'source_grid_results.csv',index=False)
    pd.DataFrame([{k:h[k] for k in ['model','source','C','validation_auc','train_units','validation_units']} for h in heads]).to_csv(out/'source_selected_heads.csv',index=False)
    (out/'source_selected_head_parameters.json').write_text(json.dumps(heads,indent=2)+'\n')
    (out/'selection_seal.json').write_text(json.dumps({'protocol_sha256':sha(out/'protocol_locked_before_fit.json'),
        'selected_parameters_sha256':sha(out/'source_selected_head_parameters.json'),
        'source_grid_sha256':sha(out/'source_grid_results.csv'),'external_evaluation_started':False},indent=2)+'\n')

def evaluate(features,out):
    seal=json.loads((out/'selection_seal.json').read_text())
    assert seal['selected_parameters_sha256']==sha(out/'source_selected_head_parameters.json')
    assert seal['protocol_sha256']==sha(out/'protocol_locked_before_fit.json')
    assert seal['source_grid_sha256']==sha(out/'source_grid_results.csv')
    heads=json.loads((out/'source_selected_head_parameters.json').read_text())
    rows=[];scores=[];hashes={}
    for h in heads:
      targets=[t for s,t in SCENARIOS if s==h['source']]
      for target in targets:
        meta,X,ih=load_units(features,h['model'],target)
        scaler=StandardScaler()
        scaler.mean_=np.array(h['mean']);scaler.scale_=np.array(h['scale'])
        scaler.var_=scaler.scale_**2;scaler.n_features_in_=768
        z=scaler.transform(X)@np.array(h['coef'])+h['intercept']
        rows.append({'model':h['model'],'source':h['source'],'target':target,'selected_C':h['C'],
                'auc':float(roc_auc_score(meta.y,z)),'ap':float(average_precision_score(meta.y,z)),
                'target_units':len(meta)})
        scores.append(meta.assign(model=h['model'],source=h['source'],target=target,score=z))
        hashes[h['model']+'_'+target]=ih
    d=pd.DataFrame(rows);d.to_csv(out/'external_results.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(out/'external_target_scores.csv',index=False)
    margins=[]
    for (source,target),g in d.groupby(['source','target']):
      g=g.set_index('model')
      for model in ['URFM','UltraFedFM']:
        margins.append({'usfm':model,'source':source,'target':target,
            'auc_usfm':g.loc[model,'auc'],'auc_mae':g.loc['MAE','auc'],
            'margin':float(g.loc[model,'auc']-g.loc['MAE','auc'])})
    m=pd.DataFrame(margins);m.to_csv(out/'external_margins.csv',index=False)
    (out/'evaluation_metadata.json').write_text(json.dumps({'selection_verified':True,
        'selection_seal_sha256':sha(out/'selection_seal.json'),'feature_input_sha256':hashes,
        'external_rows':len(d),'new_encoder_training':False,'primary_results_replaced':False,
        'BUSI_cache_order':'manifest train rows followed by val rows, matching the original notebook'},indent=2)+'\n')
    print(m[['usfm','source','target','margin']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--stage',choices=['select','evaluate'],required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results/lp_convergence_control')
    args=ap.parse_args()
    (select if args.stage=='select' else evaluate)(args.features,args.output)
