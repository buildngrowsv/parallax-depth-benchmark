"""Verify layout-level counts, bootstrap intervals, and seeded projection replay."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent/"results"


def main():
    receipt=json.loads((ROOT/"receipt.json").read_text())
    assert receipt["revision"]=="6e4d97bfbd639239d597fd06b78082068573d0c6" and receipt["status"]=="done"
    report=json.loads((ROOT/"result.json").read_text())
    data=np.load(ROOT/"per-layout.npy")
    assert data.shape==(1024,4,5) and np.issubdtype(data.dtype,np.integer)
    assert np.all(data>=0) and np.all(data[:,:,1:]<=data[:,:,:1])
    np.testing.assert_array_equal(data[:,0],data[:,1]+data[:,2])
    assert np.all(data[:,-1,1:3]==0)
    assert np.all(data[:,-1,0]*2<=data[:,1,0])
    labels=("all_foreground","horizontal_repeat","vertical_repeat","exact_alias_pairs")
    modes=("first_horizontal","opposite_horizontal","perpendicular","wrong_perpendicular")
    summary={}
    for i,name in enumerate(labels):
        n,*values=data[:,i].sum(0).tolist()
        summary[name]=dict(n=n,correct=dict(zip(modes,values,strict=True)))
    assert summary==report["subsets"] and report["layouts"]==1024
    paired=data[:,-1]
    n=int(paired[:,0].sum())
    samples=np.random.default_rng(603118).integers(1024,size=(4000,1024))
    counts=paired[samples,0].sum(1).clip(min=1)
    effects={}
    for k,control in (("same_axis",2),("wrong_image",4)):
        diff=paired[:,3]-paired[:,control]
        effects[k]=dict(mean=float(diff.sum()/n),interval=np.quantile(diff[samples].sum(1)/counts,[.025,.975]).tolist())
    assert effects==report["perpendicular_advantages"]
    passes=n>=500 and all(v["mean"]>=.2 and v["interval"][0]>0 for v in effects.values())
    assert report["decision"]==("orthogonal_resolves_alias" if passes else "unresolved")
    from run import sample
    rng=np.random.default_rng(1605831)
    replay=np.array([sample(rng) for _ in range(1024)])
    np.testing.assert_array_equal(replay,data)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
