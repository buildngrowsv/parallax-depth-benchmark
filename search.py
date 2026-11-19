import mlx.core as mx
import numpy as np
from experiment import Matcher,R,SEEDS
from scenes import LEVELS,offset_features,quartet_panel


def main():
    models={}
    for seed in SEEDS:
        m=Matcher();m.load_weights(str(R/f"matcher-{seed}.safetensors"));models[str(seed)]=m
    result={}
    for left,owner,kinds,b,worlds in quartet_panel(True,seed=917341):
        y,x=np.nonzero(left>0);flat=kinds[owner[y,x]]==0
        record={}
        for right,_,_,truth in worlds:
            target=b*truth[y,x]
            for label,shifts in (("four",b*LEVELS),("dense",np.arange(-8,9))):
                f=offset_features(left,right,shifts,y,x)
                scores={"stereo":-np.abs(f[:,:,:9]-f[:,:,9:]).mean(-1)}
                scores.update({name:np.array(net(mx.array(f))) for name,net in models.items()})
                for name,logits in scores.items():
                    pred=shifts[logits.argmax(1)]
                    winners=logits==logits.max(1,keepdims=True)
                    fractional=(winners*(shifts[None,:]==target[:,None])).sum(1)/winners.sum(1)
                    row=[]
                    for mask in (np.ones(len(x),bool),flat,~flat):
                        row+= [mask.sum(),(mask&(pred==target)).sum(),fractional[mask].sum(),
                               (mask&~np.isin(pred,b*LEVELS)).sum()]
                    record.setdefault(label+"_"+name,[]).append(row)
        for k,v in record.items():result.setdefault(k,[]).append(v)
    np.savez_compressed(R/"search.npz",**{k:np.array(v) for k,v in result.items()})
    print("Independent search panel complete",flush=True)


if __name__=="__main__": main()
