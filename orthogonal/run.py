"""Compare collinear and perpendicular cameras on periodic depth aliases."""

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scenes import H, W, LEVELS, PATCH, layout

ARMS=("first_horizontal","opposite_horizontal","perpendicular","wrong_perpendicular")


def textured_layout(rng):
    _,owner,_=layout(rng,novel=True)
    image=np.zeros(owner.shape,np.float32)
    horizontal=rng.permutation([True,True,False,False])
    for surface,repeat_x in enumerate(horizontal):
        y,x=np.nonzero(owner==surface)
        if repeat_x:
            tex=rng.uniform(.18,.88,size=(H,3)).astype(np.float32)
            image[y,x]=tex[y,x%3]
        else:
            tex=rng.uniform(.18,.88,size=(W,3)).astype(np.float32)
            image[y,x]=tex[x,y%3]
    return image,owner,horizontal


def project(image,depth,move):
    frame=np.zeros_like(image)
    ids=np.full(image.shape,-1,np.int32)
    for d in LEVELS:
        y,x=np.nonzero((image>0)&(depth==d))
        a,b=y-move[0]*d,x-move[1]*d
        keep=(0<=a)&(a<H)&(0<=b)&(b<W)
        frame[a[keep],b[keep]]=image[y[keep],x[keep]]
        ids[a[keep],b[keep]]=y[keep]*W+x[keep]
    return frame,ids


def patch_cost(reference, frame, move, y, x):
    pad=5
    dy,dx=PATCH.T
    yy=y[:,None]+dy+pad
    xx=x[:,None]+dx+pad
    a=np.pad(reference,pad)[yy,xx]
    b=np.pad(frame,pad)[yy[:,None,:]-move[0]*LEVELS[None,:,None],
                        xx[:,None,:]-move[1]*LEVELS[None,:,None]]
    return np.abs(a[:,None,:]-b).mean(2)


def visible_patch(ids,move,y,x,depth):
    dy,dx=PATCH.T
    yy=y[:,None]+dy
    xx=x[:,None]+dx
    a,b=yy-depth[:,None]*move[0],xx-depth[:,None]*move[1]
    inside=((yy>=0)&(yy<H)&(xx>=0)&(xx<W)&(a>=0)&(a<H)&(b>=0)&(b<W)).all(1)
    picked=ids[np.clip(a,0,H-1),np.clip(b,0,W-1)]
    return inside&(picked==yy*W+xx).all(1)


def sample(rng):
    image,owner,horizontal=textured_layout(rng)
    y,x=np.nonzero(image)
    kind=horizontal[owner[y,x]]
    start=rng.permutation(4)
    maps=[(start[owner]+phase)%4+1 for phase in range(4)]
    b=int(rng.choice((-1,1)))
    first=(0,b)
    opposite=(0,-b)
    vertical=(b,0)
    frames_y=[project(image,d,vertical)[0] for d in maps]
    all_predictions, all_depth, aliases=[],[],[]
    for i,d in enumerate(maps):
        a,ai=project(image,d,first)
        c,ci=project(image,d,opposite)
        base=patch_cost(image,a,first,y,x)
        back=patch_cost(image,c,opposite,y,x)
        normal=patch_cost(image,frames_y[i],vertical,y,x)
        wrong=patch_cost(image,frames_y[(i+1)%4],vertical,y,x)
        predictions=np.stack([z.argmin(1)+1 for z in (base,base+back,base+normal,base+wrong)],axis=1)
        true=d[y,x]
        a_visible=visible_patch(ai,first,y,x,true)
        c_visible=visible_patch(ci,opposite,y,x,true)
        aliases.append((base[:,0]==0)&(base[:,3]==0)&(back[:,0]==0)&(back[:,3]==0)&a_visible&c_visible)
        all_depth.append(true)
        all_predictions.append(predictions)
    depth=np.array(all_depth)
    pred=np.array(all_predictions)
    alias=np.array(aliases)
    j1=(depth==1).argmax(0)
    j4=(depth==4).argmax(0)
    p=np.arange(len(y))
    core=kind&alias[j1,p]&alias[j4,p]
    success=(pred[j1,p]==1)&(pred[j4,p]==4)
    hits=pred==depth[...,None]
    pooled=[]
    for mask in (np.ones(len(y),bool),kind,~kind):
        pooled.append([4*int(mask.sum()),*hits[:,mask,:].sum(axis=(0,1)).tolist()])
    return np.array([*pooled,[int(core.sum()),*success[core,:].sum(0).tolist()]],dtype=np.int64)


def summarize(counts):
    total=counts.sum(0)
    labels=("all_foreground","horizontal_repeat","vertical_repeat","exact_alias_pairs")
    record={k:dict(n=int(row[0]),correct=dict(zip(ARMS,row[1:].tolist(),strict=True)))
            for k,row in zip(labels,total,strict=True)}
    target=counts[:,-1]
    n=int(target[:,0].sum())
    intervals={}
    if n:
        draws=np.random.default_rng(603118).integers(len(target),size=(4000,len(target)))
        denominators=target[:,0][draws].sum(1).clip(min=1)
        for index,name in ((2,"same_axis"),(4,"wrong_image")):
            d=target[:,3]-target[:,index]
            intervals[name]=dict(mean=float(d.sum()/n),
                                 interval=np.percentile(d[draws].sum(1)/denominators,[2.5,97.5]).tolist())
    passed=n>=500 and all(v["mean"]>=.2 and v["interval"][0]>0 for v in intervals.values())
    return dict(decision="orthogonal_resolves_alias" if passed else "unresolved",layouts=len(counts),
                subsets=record,perpendicular_advantages=intervals)


def main():
    out=Path(os.environ["STUDIO_JOB_DIR"])
    rng=np.random.default_rng(1605831)
    counts=np.array([sample(rng) for _ in range(1024)])
    result=summarize(counts)
    np.save(out/"per-layout.npy",counts)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
