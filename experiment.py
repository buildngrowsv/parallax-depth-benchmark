"""Train on image intensity agreement and measure depth on held-out ambiguity quartets."""
import json
from pathlib import Path
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from scenes import LEVELS,W,features,quartet_panel,render,training_batch

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)


class Matcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(18,32)
        self.output=nn.Linear(32,1)
    def __call__(self,f):
        return self.output(nn.relu(self.hidden(f)))[...,0]


def fit(seed):
    mx.random.seed(seed)
    net=Matcher()
    opt=optim.Adam(.008)
    rng=np.random.default_rng(33891+seed)
    def objective(m,f,error):
        return (mx.softmax(m(f),axis=1)*error).sum(1).mean()
    vg=nn.value_and_grad(net,objective)
    state=[net.state,opt.state]
    from functools import partial
    @partial(mx.compile,inputs=state,outputs=state)
    def step(f,e):
        loss,g=vg(net,f,e)
        opt.update(net,g)
        return loss
    history=[]
    start=time.perf_counter()
    for j in range(1,401):
        f,err=training_batch(rng)
        loss=step(mx.array(f),mx.array(err))
        mx.eval(loss,state)
        if j in (1,25,100,250,400): history.append([j,loss.item()])
    net.save_weights(str(R/f"matcher-{seed}.safetensors"))
    return net,dict(seed=seed,seconds=time.perf_counter()-start,history=history,device=mx.device_info()["device_name"])


def evaluate(models,novel):
    metrics={}; primary=[]
    for scene,(left,owner,kinds,b,worlds) in enumerate(quartet_panel(novel)):
        y,x=np.nonzero(left>0)
        per={}; label_stack=[]
        for right,indices,target,truth in worlds:
            z=truth[y,x]
            vis=indices[y,x-b*z]==y*W+x
            isflat=kinds[owner[y,x]]==0
            f=features(left,right,b,y,x)
            guesses={"stereo":np.abs(f[:,:,:9]-f[:,:,9:]).mean(-1).argmin(1)+1,
                     "single":np.full(len(x),2,np.int32),"oracle":z}
            guesses.update({k:np.array(m(mx.array(f)).argmax(-1))+1 for k,m in models.items()})
            label_stack.append(z)
            for k,pred in guesses.items():
                plane=np.zeros_like(owner)
                plane[y,x]=pred
                view=render(left,plane,-3*int(np.sign(b)))
                union=(view>0)|(target>0)
                score=[]
                for mask in (np.ones(len(z),bool),vis,~vis,isflat,~isflat):
                    score.extend([mask.sum(),(mask&(pred==z)).sum()])
                score.extend([union.sum(),np.abs(view-target)[union].sum(dtype=np.float64)])
                per.setdefault(k,[]).append(score)
                if k=="oracle": assert np.array_equal(view,target)
        assert np.all(np.sort(np.stack(label_stack),axis=0)==LEVELS[:,None])
        for k,vals in per.items():
            arr=np.array(vals,dtype=np.float64)
            if k=="single": assert arr[:,1].sum()*4==arr[:,0].sum()
            metrics.setdefault(k,[]).append(arr)
        primary.append([scene,len(x),int(b)])
    name="shift" if novel else "original"
    np.savez_compressed(R/f"{name}.npz",layout=np.array(primary),**{k:np.stack(v) for k,v in metrics.items()})


def main():
    R.mkdir(exist_ok=True)
    models={};metadata=[]
    for seed in SEEDS:
        mx.random.seed(seed)
        models[f"initial_{seed}"]=Matcher()
        net,r=fit(seed);models[f"trained_{seed}"]=net;metadata.append(r)
        print("fit",seed,r["history"][-1],flush=True)
    evaluate(models,False);evaluate(models,True)
    (R/"training.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Scored both held-out panels",flush=True)


if __name__=="__main__": main()
