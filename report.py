"""Ratios of pooled counts. Resample whole layouts and paired training seeds."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)
def contrast(a,b):
    """Each array is seed by layout by [denominator, numerator]."""
    assert a.shape==b.shape
    rng=np.random.default_rng(98321)
    sims=[]
    for _ in range(25):
        j=rng.integers(a.shape[0],size=(200,a.shape[0],1))
        i=rng.integers(a.shape[1],size=(200,1,a.shape[1]))
        av,bv=a[j,i].sum((1,2)),b[j,i].sum((1,2))
        sims.extend(av[:,1]/av[:,0]-bv[:,1]/bv[:,0])
    x,y=a.sum((0,1)),b.sum((0,1))
    return [x[1]/x[0]-y[1]/y[0], *np.quantile(sims,[.025,.975])]

def counts(source,keys,cols=(0,1)):
    return np.stack([source[k].sum(1)[:,list(cols)] for k in keys])

summary={}
for panel in ("original","shift","search"):
    with np.load(R/f"{panel}.npz") as src:
        scores, effects = {}, {}
        for k in src.files:
            if k=="layout": continue
            v=src[k].sum((0,1))
            if panel=="search":
                measures={}
                for i,name in enumerate(("all","flat","textured")):
                    block=v[4*i:4*i+4]
                    measures[name]=dict(sites=int(block[0]),hits=int(block[1]),exact=block[1]/block[0],
                                        uniform_tie=block[2]/block[0],inadmissible=block[3]/block[0])
                scores[k]=measures
            else:
                scores[k]={name:dict(sites=int(v[i]),hits=int(v[i+1]),exact=v[i+1]/v[i])
                           for i,name in zip(range(0,10,2),("all","visible","occluded","flat","textured"))}
                scores[k]["novel_view_mae"]=v[11]/v[10]
        if panel=="search":
            for label in ("four","dense"):
                a=counts(src,[f"{label}_{s}" for s in SEEDS])
                b=counts(src,[f"{label}_stereo"]*3)
                effects[label+"_trained_minus_stereo"]=contrast(a,b)
            effects["dense_minus_four"]=contrast(counts(src,[f"dense_{s}" for s in SEEDS]),counts(src,[f"four_{s}" for s in SEEDS]))
        else:
            for baseline in ("initial","single","stereo"):
                arms=([f"initial_{s}" for s in SEEDS] if baseline=="initial" else [baseline]*3)
                effects["trained_minus_"+baseline]=contrast(counts(src,[f"trained_{s}" for s in SEEDS]),counts(src,arms))
        summary[panel]=dict(scores=scores,effects=effects)
(R/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
for k,d in summary.items():print(k,d["effects"])

