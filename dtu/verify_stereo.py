"""Recompute the real-photo comparisons from published DTU pilot outcomes."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "pilot-results"
METHODS = ("horizontal", "vertical", "random", "geometry", "structure", "repetition", "hindsight")


def main():
    revision = (ARCHIVE / "source-revision.txt").read_text().strip()
    assert revision.startswith("3e71163") and len(revision) == 40
    result = json.loads((ARCHIVE / "result.json").read_text())
    for key, file in (("manifest_sha256", "staged-manifest.json"), ("visibility_sha256", "visibility.json")):
        assert sha256((ROOT / "results" / file).read_bytes()).hexdigest() == result[key]
    source = json.loads((ROOT / "results/staged-manifest.json").read_text())
    qualifiers = json.loads((ROOT / "results/visibility.json").read_text())
    all_proofs = {}
    for scene in ("1", "6"):
        report = result["scenes"][scene]
        ids = [r["reference"] for r in source["triplets"]]
        assert [r["reference"] for r in report["references"]] == ids
        assert [r["reference"] for r in qualifiers["scenes"][scene]["triplets"]] == ids
        with np.load(ARCHIVE / f"scan-{scene}.npz") as data:
            px, truth, depth, path, cues = (data[k] for k in ("pixels", "true", "predicted", "routes", "cues"))
        assert px.shape == (12, 512, 2) and truth.shape == (12, 512) and depth.shape == (12, 2, 512)
        assert path.shape == (12, 7) and cues.shape == (12, 2, 2)
        assert np.isfinite(truth).all() and (truth > 0).all()
        assert np.isin(path, (0, 1)).all()
        expected_random = np.random.default_rng(718403 + int(scene)).permutation(np.arange(len(ids)) % 2)
        np.testing.assert_array_equal(path[:, 2], expected_random)
        geometry = [int(np.argmax(row["baseline_mm"])) for row in source["triplets"]]
        np.testing.assert_array_equal(path[:, 3], geometry)
        for row in px:
            assert len(np.unique(row, axis=0)) == 512
        conditional = (np.abs(depth - truth[:, None]) <= 10).sum(axis=-1)
        for i, row in enumerate(report["references"]):
            assert row["correct"] == conditional[i].tolist()
            assert row["choices"] == dict(zip(METHODS, path[i].tolist(), strict=True))
            np.testing.assert_array_equal(cues[i], [row["cue"]["repetition"], row["cue"]["structure"]])
            assert path[i, 4] == cues[i, 1].argmax() and path[i, 5] == cues[i, 0].argmax()
            assert path[i, 6] == conditional[i].argmax()
        proof = {}
        for j, key in enumerate(METHODS):
            error = np.abs(depth[np.arange(12), path[:, j]] - truth)
            finite = np.isfinite(error)
            hit = int((error <= 10).sum())
            expected = dict(correct=hit, total=truth.size, estimated=int(finite.sum()),
                            median_abs_mm=float(np.median(error[finite])) if finite.any() else None)
            assert expected == report["scores"][key]
            proof[key] = expected
        strongest = max(proof[key]["correct"] for key in ("horizontal", "vertical", "structure"))
        delta = (proof["repetition"]["correct"] - strongest) / truth.size
        assert report["improvement_over_strongest_control"] == delta
        all_proofs[scene] = dict(scores=proof, gain=delta,
                                 truth_mm=np.quantile(truth, [0,.1,.5,.9,1]).tolist(),
                                 choices_vertical={key: int(path[:, j].sum()) for j, key in enumerate(METHODS)})
    passes = all(v["gain"] >= .05 for v in all_proofs.values())
    assert result["decision"] == ("photo_selector_qualifies" if passes else "below_pilot_gate")
    print(json.dumps(dict(decision=result["decision"], scenes=all_proofs, runtime_s=result["runtime_s"]), indent=2))


if __name__ == "__main__":
    main()
