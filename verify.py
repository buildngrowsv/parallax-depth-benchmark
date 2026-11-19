"""Verify published depth scores and independent projection with held-out scenes."""
import json
import numpy as np
import mlx.core as mx
from experiment import Matcher, R, SEEDS
from scenes import LEVELS, W, quartet_panel


def project_reference(image, depth, baseline):
    """Project point samples in a single sort, resolving overlapping rays by depth."""
    height, width = image.shape
    y, x = np.nonzero(image > 0)
    z = depth[y, x]
    dest_x = x - baseline * z
    valid = (dest_x >= 0) & (dest_x < width)
    y, x, z, dest_x = (a[valid] for a in (y, x, z, dest_x))
    dest = y * width + dest_x
    perm = np.lexsort((-z, dest))
    output = np.zeros_like(image)
    picked = np.full((height, width), -1, dtype=int)
    _, first = np.unique(dest[perm], return_index=True)
    keep = perm[first]
    output[y[keep], dest_x[keep]] = image[y[keep], x[keep]]
    picked[y[keep], dest_x[keep]] = y[keep] * width + x[keep]
    return output, picked


def patch_pairs(left, right, rows, cols, displacements):
    pairs = []
    for shift in displacements:
        patches = []
        for frame, offset in ((left, 0), (right, shift)):
            patches.extend(frame[rows + dy, cols + dx - offset]
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        pairs.append(np.stack(patches, axis=1))
    return np.stack(pairs, axis=1)


def main():
    networks = {}
    for seed in SEEDS:
        net = Matcher()
        net.load_weights(str(R / f"matcher-{seed}.safetensors"))
        networks[seed] = net
    audited = {}
    near_ties = {}
    max_logit_difference = {}
    for name, novel, panel_seed in (("original", False, None), ("shift", True, None),
                                     ("search", True, 917341)):
        rows_checked = 0
        with np.load(R / f"{name}.npz") as stored:
            for index, (left, owners, kinds, b, worlds) in enumerate(quartet_panel(novel, seed=panel_seed)):
                y, x = np.nonzero(left)
                for phase, (right, ids, new_view, truth) in enumerate(worlds):
                    r, picks = project_reference(left, truth, b)
                    np.testing.assert_array_equal(r, right)
                    np.testing.assert_array_equal(picks, ids)
                    np.testing.assert_array_equal(project_reference(left, truth, -3*np.sign(b))[0], new_view)
                    searches = (("four", b*LEVELS), ("dense", np.arange(-8,9))) if name=="search" else (("", b*LEVELS),)
                    z = truth[y, x]
                    for label, shifts in searches:
                        pairs = patch_pairs(left, right, y, x, shifts)
                        scores = {"stereo":-abs(pairs[...,:9]-pairs[...,9:]).mean(-1)}
                        for seed, model in networks.items():
                            scores[str(seed)] = np.array(model(mx.array(pairs)))
                        for method, score in scores.items():
                            pred = shifts[score.argmax(-1)]
                            correct = pred == b*z
                            key = (label+"_"+method if label else ("stereo" if method=="stereo" else "trained_"+method))
                            obs = stored[key][index, phase]
                            assert (len(z),correct.sum()) == tuple(obs[:2])
                            flat = kinds[owners[y,x]] == 0
                            if label:
                                best = score == score.max(1,keepdims=True)
                                tie = ((shifts[None,:] == (b*z)[:,None]) & best).sum(1)/best.sum(1)
                                for j,mask in enumerate((np.ones_like(flat),flat,~flat)):
                                    np.testing.assert_allclose(obs[4*j:4*j+4], [mask.sum(),correct[mask].sum(),
                                        tie[mask].sum(), (~np.isin(pred[mask], b*LEVELS)).sum()], atol=1e-8,rtol=0)
                            else:
                                visible = ids[y,x-b*z] == y*W+x
                                for j,mask in enumerate((np.ones_like(flat),visible,~visible,flat,~flat)):
                                    np.testing.assert_array_equal(obs[2*j:2*j+2], [mask.sum(),correct[mask].sum()])
                                depth = np.zeros_like(owners)
                                depth[y,x] = pred//b
                                view = project_reference(left,depth,-3*np.sign(b))[0]
                                union = (view>0)|(new_view>0)
                                np.testing.assert_allclose(obs[10:],[union.sum(),abs(view-new_view)[union].sum(dtype=np.float64)],atol=1e-8,rtol=0)
                            if method!="stereo" and index<8:
                                w=networks[int(method)]
                                hidden=np.maximum(pairs @ np.array(w.hidden.weight).T+np.array(w.hidden.bias),0)
                                cpu=(hidden @ np.array(w.output.weight).T+np.array(w.output.bias))[...,0]
                                k=name+"_"+key
                                assert np.isfinite(cpu).all() and np.isfinite(score).all()
                                max_logit_difference[k]=max(max_logit_difference.get(k,0),float(abs(cpu-score).max()))
                                d=near_ties.setdefault(k,[0,0])
                                d[0]+=int((score.argmax(1)!=cpu.argmax(1)).sum())
                                d[1]+=len(z)
                    if name=="shift" and index==0:
                        if phase==0: example={"baseline":int(b),"image":np.rint(255*left).astype(int).tolist(),"worlds":[]}
                        patches=patch_pairs(left,right,y,x,b*LEVELS)
                        predicted=np.zeros_like(owners); predicted[y,x]=np.array(networks[101](mx.array(patches))).argmax(1)+1
                        example["worlds"].append({"truth":np.where(left>0,truth,0).tolist(),"trained":predicted.tolist()})
                        if phase==3: (R/"example.json").write_text(json.dumps(example,separators=(",",":"))+"\n")
                    rows_checked+=1
        audited[name]=rows_checked
    report=dict(worlds_checked=audited,all_saved_trained_and_stereo_counts_reproduced=True,
                projection_identical=True,numpy_vs_metal_max_logit_difference=max_logit_difference,numpy_vs_metal_top1_disagreements_and_total=near_ties)
    (R/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__": main()

