"""Orthographic image coordinates with calibrated perspective disparity b*f/Z on visible ray samples."""
import numpy as np

H,W=30,56
LEVELS=np.arange(1,5,dtype=np.int32)
PATCH=np.array([(y,x) for y in (-1,0,1) for x in (-1,0,1)],dtype=np.int32)


def layout(rng,novel=False):
    yy,xx=np.mgrid[:H,:W]
    image=np.zeros((H,W),np.float32)
    owner=np.full((H,W),-1,np.int16)
    kinds=rng.permutation(4)
    for i in range(4):
        cx=(18,31)[i%2]+rng.integers(-1,2)
        cy=(8,21)[i//2]+rng.integers(-1,2)
        rx,ry=int(rng.integers(5,8)),int(rng.integers(4,6))
        x=(xx-cx)/rx
        y=(yy-cy)/ry
        kind=rng.integers(2)
        if novel:
            mask=(np.abs(x)+np.abs(y)<=1) if kind else ((np.abs(y)<=1)&(np.abs(x)<=(1-y)/2))
        else:
            mask=(x*x+y*y<=1) if kind else ((np.abs(x)<=1)&(np.abs(y)<=1))
        mask&=owner<0
        base=rng.uniform(.29,.71)
        if kinds[i]==0: tex=base
        elif kinds[i]==1: tex=base+.18*((xx+int(rng.integers(3)))%3-1)
        elif kinds[i]==2: tex=base+rng.uniform(-.25,.25,(H,W))
        else: tex=base+.10*np.sin(.8*xx+.7*yy+rng.uniform(0,6))+.09*np.cos(.35*xx-.5*yy)
        image[mask]=np.broadcast_to(tex,(H,W))[mask]
        owner[mask]=i
    return image,owner,kinds


def render(image,disparity,baseline,indices=False):
    # Ray samples at (u*Z/f, v*Z/f, Z) project to u-baseline*f/Z. Larger levels are closer.
    frame=np.zeros_like(image)
    pick=np.full(image.shape,-1,np.int32)
    for d in LEVELS:
        y,x=np.nonzero((image>0)&(disparity==d))
        at=x-baseline*d
        inside=(at>=0)&(at<W)
        y,x,at=y[inside],x[inside],at[inside]
        frame[y,at]=image[y,x]
        if indices: pick[y,at]=y*W+x
    return (frame,pick) if indices else frame


def offset_features(left,right,shifts,y,x):
    dy,dx=PATCH.T
    yp=y[:,None]+dy;xp=x[:,None]+dx
    # The renderer's margins keep all sampled reference patches and candidates on the canvas.
    lp=left[yp,xp]
    xx=xp[:,None,:]-np.asarray(shifts)[None,:,None]
    rp=right[yp[:,None,:],xx]
    return np.concatenate((np.broadcast_to(lp[:,None,:],rp.shape),rp),axis=-1).astype(np.float32)


def features(left,right,baseline,y,x):
    return offset_features(left,right,baseline*LEVELS,y,x)


def training_batch(rng):
    features_all=[];residuals=[]
    for _ in range(8):
        left,owner,_=layout(rng)
        depths=rng.permutation(LEVELS)
        baseline=int(rng.choice((-1,1)))
        right=render(left,depths[owner],baseline)
        y,x=np.nonzero(left>0)
        ids=rng.choice(len(y),32,replace=False)
        f=features(left,right,baseline,y[ids],x[ids])
        features_all.append(f)
        residuals.append(np.abs(f[:,:,4]-f[:,:,13]))
    return np.concatenate(features_all),np.concatenate(residuals)


def quartet_panel(novel,count=96,seed=None):
    rng=np.random.default_rng((182763 if novel else 118298) if seed is None else seed)
    for _ in range(count):
        left,owner,kinds=layout(rng,novel)
        perm=rng.permutation(4)
        b=int(rng.choice((-1,1)))*(2 if novel else 1)
        worlds=[]
        for phase in range(4):
            d=(perm[owner]+phase)%4+1
            r,pick=render(left,d,b,True)
            target=render(left,d,-3*int(np.sign(b)))
            worlds.append((r,pick,target,d))
        yield left,owner,kinds,b,worlds
