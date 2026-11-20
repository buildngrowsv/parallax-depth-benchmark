import json
import os
from pathlib import Path
import numpy as np
from run import OUT,W,cost,quartet_panel,render

def backward(ref,context,b,y,x,depth):
    # The second search can extend outside the original image margins. Use constant black padding.
    margin=20
    a=np.pad(context,margin);c=np.pad(ref,margin)
    right_x=x-b*depth
    back=cost(a,c,-b,y+margin,right_x+margin).argmin(1)+1
    return right_x+b*back!=x

def main():
    panels={}
    for displacement,seed in ((2,781341),(1,1043921)):
        rows=[];rng=np.random.default_rng(seed+97)
        for left,owner,kinds,b0,worlds in quartet_panel(True,count=512,seed=seed):
            b=int(np.sign(b0))*displacement;y,x=np.nonzero(left);counts=[]
            for _,_,_,d in worlds:
                a,ia=render(left,d,b,True);c,ic=render(left,d,-b,True)
                z=d[y,x];c1=cost(left,a,b,y,x);c2=cost(left,c,-b,y,x)
                p1=c1.argmin(1)+1;p2=c2.argmin(1)+1
                choose=backward(left,a,b,y,x,p1)
                rand=np.zeros(len(x),bool);rand[rng.permutation(len(x))[:choose.sum()]]=True
                methods=[p1,np.minimum(c1,c2).argmin(1)+1,np.where(choose,p2,p1),np.where(rand,p2,p1)]
                hit=np.stack(methods)==z
                vis=(ia[y,x-b*z]!=y*W+x)&(ic[y,x+b*z]==y*W+x)
                row=[]
                for mask in (np.ones(len(x),bool),vis):
                    row.append([int(mask.sum()),*hit[:,mask].sum(1).tolist(),int((choose&mask).sum()),
                                int((~hit[0]&hit[2]&mask).sum()),int((hit[0]&~hit[2]&mask).sum())])
                counts.append(row)
            rows.append(counts)
        v=np.array(rows);np.save(OUT/f"baseline-{displacement}.npy",v);q=v.sum(1)[:,0]
        ix=np.random.default_rng(9165).integers(len(q),size=(4000,len(q)))
        intervals={}
        for j,k in ((1,"single"),(4,"random")):
            diff=q[:,3]-q[:,j]
            intervals[k]=np.quantile(diff[ix].sum(1)/q[:,0][ix].sum(1),[.025,.975]).tolist()
        panels[str(displacement)]=dict(columns=["n","single","minimum","routed","random","opposite_view_calls","gained","lost"],
                                      all=v.sum((0,1))[0].tolist(),newly_visible=v.sum((0,1))[1].tolist(),route_advantage_ci=intervals)
    passed=all(ci[0]>0 for p in panels.values() for ci in p["route_advantage_ci"].values())
    tmp=OUT/"result.tmp";tmp.write_text(json.dumps(dict(decision="qualified" if passed else "complete",panels=panels),indent=2)+"\n")
    os.replace(tmp,OUT/"result.json")
if __name__=="__main__":main()

