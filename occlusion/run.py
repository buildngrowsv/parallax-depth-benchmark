import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scenes import W,features,quartet_panel,render

OUT=Path(os.environ["STUDIO_JOB_DIR"])
def cost(left,view,b,y,x):
    f=features(left,view,b,y,x)
    return np.abs(f[:,:,:9]-f[:,:,9:]).mean(-1)
def main():
    validation=sys.argv[1]=="pilot"
    seed=492731 if validation else 743915
    names=("single","duplicate","matched","mismatched")
    counts=[]
    for left,owner,kinds,b0,worlds in quartet_panel(True,count=512,seed=seed):
        b=b0 if validation else int(np.sign(b0))
        y,x=np.nonzero(left);ids=y*W+x;world_counts=[]
        for j,(_,_,_,d) in enumerate(worlds):
            va,ia=render(left,d,b,True);vb,ib=render(left,d,-b,True)
            wrong=render(left,worlds[(j+1)%4][3],-b)
            z=d[y,x];vis_a=ia[y,x-b*z]==ids;vis_b=ib[y,x+b*z]==ids
            a=cost(left,va,b,y,x)
            scores=[a,np.minimum(a,a),np.minimum(a,cost(left,vb,-b,y,x)),np.minimum(a,cost(left,wrong,-b,y,x))]
            pred=np.stack([s.argmin(1)+1 for s in scores])
            assert np.array_equal(pred[0],pred[1])
            h=pred==z
            row=[]
            for mask in (np.ones(len(z),bool),~vis_a&vis_b,~vis_a&~vis_b):
                row.append([int(mask.sum()),*h[:,mask].sum(1).tolist(),int((~h[0]&h[2]&mask).sum()),int((h[0]&~h[2]&mask).sum())])
            world_counts.append(row)
        counts.append(world_counts)
    a=np.array(counts);np.save(OUT/"per-layout.npy",a)
    total=a.sum((0,1));draw=np.random.default_rng(3712).integers(len(a),size=(4000,len(a)))
    d=a.sum(1)[:,1];delta=(d[:,3]-d[:,1])
    limits=np.quantile(delta[draw].sum(1)/np.maximum(1,d[:,0][draw].sum(1)),[.025,.975]).tolist()
    metrics={}
    for label,row in zip(("all","newly_visible","hidden_both"),total):
        n=int(row[0]);metrics[label]=dict(n=n,correct=dict(zip(names,row[1:5].tolist())),gained=int(row[5]),lost=int(row[6]))
    overall,visible=total[:2]
    go=bool(visible[0]>0 and (visible[3]-visible[1])/max(1,visible[0])>=.10 and limits[0]>0 and visible[3]>visible[4] and (overall[3]-overall[1])/overall[0]>-.02)
    p=OUT/"result.tmp";p.write_text(json.dumps(dict(decision="confirm" if validation and go else "complete",phase=sys.argv[1],
        metrics=metrics,matched_gain_interval=limits,layouts=len(a)),indent=2)+"\n");os.replace(p,OUT/"result.json")
if __name__=="__main__":main()

