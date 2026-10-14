"""Verify feature-cache row correspondence using six archived seed-42 LP heads.

No arbitrary torch/pickle classes are imported. Only contiguous float32 tensors
and OrderedDict are accepted. Head checkpoints and feature caches are supplied
separately; neither is redistributed by this script.
"""
import argparse, collections, hashlib, io, json, pickle, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from lp_convergence_control import ROOT, MODELS, SOURCES, SCENARIOS, load_units

class FloatStorage:
    pass

def rebuild(storage, offset, shape, stride, requires_grad, hooks):
    expected=[];step=1
    for n in reversed(shape):
        expected.insert(0,step);step*=n
    if tuple(stride)!=tuple(expected) or offset<0 or offset+step>len(storage):
        raise ValueError('Only bounded contiguous tensors are accepted')
    return storage[offset:offset+step].reshape(shape).copy()

def read_head(path):
    with zipfile.ZipFile(path) as z:
        roots={n.split('/')[0] for n in z.namelist()};assert len(roots)==1
        root=roots.pop()
        assert z.read(root+'/byteorder')==b'little'
        class SafeUnpickler(pickle.Unpickler):
            def find_class(self,module,name):
                allowed={('torch','FloatStorage'):FloatStorage,
                         ('torch._utils','_rebuild_tensor_v2'):rebuild,
                         ('collections','OrderedDict'):collections.OrderedDict}
                if (module,name) not in allowed:
                    raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
                return allowed[module,name]
            def persistent_load(self,pid):
                tag,kind,key,location,count=pid
                assert tag=='storage' and kind is FloatStorage and location=='cpu'
                a=np.frombuffer(z.read(root+'/data/'+key),dtype='<f4')
                assert len(a)==count
                return a
        head=SafeUnpickler(io.BytesIO(z.read(root+'/data.pkl'))).load()
    assert head['seed']==42
    assert set(head['state_dict'])=={'weight','bias'}
    assert head['state_dict']['weight'].shape==(2,768)
    assert head['state_dict']['bias'].shape==(2,)
    return head

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--heads',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results/submission_checks')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    p=ROOT/'results/archived/FULLSTUDY_FINAL_FULLFT_RESULTS/LINEAR_PROBE_TARGET_SCORES_RECOMPUTED.csv'
    archived=pd.read_csv(p,dtype={'unit_id':str},float_precision='round_trip')
    rows=[]
    for model,stem in MODELS.items():
        for source in SOURCES:
            hp=args.heads/f'{stem}_{source}_seed42_best.pt'
            head=read_head(hp);assert head['source']==source
            for s,target in SCENARIOS:
                if s!=source:continue
                meta,X,ih=load_units(args.features,model,target)
                params=head['state_dict']
                logits=X@params['weight'].T+params['bias']
                score=(logits[:,1]-logits[:,0]).astype(float)
                old=archived[(archived.model==model)&(archived.source==source)&
                             (archived.seed==42)&(archived.target==target)].set_index('unit_id')
                assert set(meta.unit_id)==set(old.index)
                old=old.loc[meta.unit_id]
                assert np.array_equal(meta.y.to_numpy(),old.y.to_numpy())
                err=float(np.abs(score-old.score.to_numpy()).max())
                auc=float(roc_auc_score(meta.y,score));apv=float(average_precision_score(meta.y,score))
                auco=float(roc_auc_score(old.y,old.score));apo=float(average_precision_score(old.y,old.score))
                assert err<2e-5,(model,source,target,err)
                assert abs(auc-auco)<1e-12 and abs(apv-apo)<1e-12
                rows.append({'model':model,'source':source,'target':target,'seed':42,
                  'units':len(meta),'max_absolute_score_error':err,'auc':auc,'ap':apv,
                  'auc_difference':auc-auco,'ap_difference':apv-apo,
                  'head_sha256':hashlib.sha256(hp.read_bytes()).hexdigest(),
                  'feature_sha256':ih['feature'],'manifest_sha256':ih['manifest']})
    d=pd.DataFrame(rows);assert len(d)==12
    d.to_csv(args.output/'feature_cache_head_correspondence.csv',index=False)
    (args.output/'feature_cache_head_correspondence.json').write_text(json.dumps({
       'scope':'Six original seed-42 heads; all nine model/dataset feature arrays used',
       'external_evaluations':12,'score_tolerance':2e-5,
       'max_absolute_score_error':float(d.max_absolute_score_error.max()),
       'metrics_identical':True,'no_arbitrary_pickle_globals':True,
       'note':'Small float32 differences reflect equivalent image-score and patient-feature averaging or BLAS order.'
    },indent=2)+'\n')
    print(d[['model','source','target','max_absolute_score_error','auc_difference','ap_difference']].to_string(index=False))

if __name__=='__main__':main()
