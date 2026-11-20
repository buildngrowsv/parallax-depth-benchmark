"""Rebuild every cell in the aggregate tables from the four-world layout records."""
import json
from pathlib import Path
import sys
import numpy as np

def main():
    base=Path(sys.argv[1]);audited={}
    for phase in ("pilot","confirm"):
        values=np.load(base/phase/"per-layout.npy")
        assert values.shape==(512,4,3,7)
        np.testing.assert_array_equal(values[:,:,:,1],values[:,:,:,2])
        np.testing.assert_array_equal(values[:,:,:,3]-values[:,:,:,1],values[:,:,:,5]-values[:,:,:,6])
        report=json.loads((base/phase/"result.json").read_text())
        for row,record in zip(values.sum((0,1)),report["metrics"].values()):
            assert row.tolist()==[record["n"],*[record["correct"][m] for m in ("single","duplicate","matched","mismatched")],record["gained"],record["lost"]]
        audited[phase]=values.shape[0]
    result=json.loads((base/"routing/result.json").read_text())
    for scale in (1,2):
        v=np.load(base/"routing"/f"baseline-{scale}.npy")
        assert v.shape==(512,4,2,8)
        np.testing.assert_array_equal(v[:,:,:,3]-v[:,:,:,1],v[:,:,:,6]-v[:,:,:,7])
        for row,key in zip(v.sum((0,1)),("all","newly_visible")):
            assert row.tolist()==result["panels"][str(scale)][key]
        audited["routing_"+str(scale)]=len(v)
    print(json.dumps(dict(layouts_checked=audited,worlds_checked=sum(audited.values())*4),indent=2))
if __name__=="__main__":main()

