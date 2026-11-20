"""Combine calibrated photo costs before selecting depth on the archived DTU pixels."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from stereo import depth_costs, pick_depth, read_image, score_depth

HERE = Path(__file__).resolve().parent
WIDE_PATCH = np.array([(x, y) for y in range(-3, 4) for x in range(-3, 4)])
ARMS = ("horizontal", "vertical", "mean_estimates", "joint_cost")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def cameras(root, manifest, scene, reference_rows):
    poses = {}
    ids = sorted({row[k] for row in reference_rows for k in ("reference", "horizontal", "vertical")})
    for key in ids:
        image = f"Rectified/scan{scene}/rect_{key:03d}_3_r5000.png"
        matrix = f"calibration/pos_{key:03d}.txt"
        for name in (image, matrix):
            assert digest(root / name) == manifest["files"][name]["sha256"]
        poses[key] = (read_image(root / image), np.loadtxt(root / matrix))
    return poses


def predict(views, row, pixels, earlier):
    ref, mat = views[row["reference"]]
    costs = []
    for key, archived in zip(("horizontal", "vertical"), earlier, strict=True):
        image, target = views[row[key]]
        replay = score_depth(ref, image, pixels, mat, target)
        np.testing.assert_array_equal(replay, archived)
        costs.append(depth_costs(ref, image, pixels, mat, target, patch=WIDE_PATCH))
    a, b = costs
    single_a, single_b = pick_depth(a), pick_depth(b)
    joint = pick_depth((a + b) / 2)
    return np.stack([single_a, single_b, (single_a + single_b) / 2, joint])


def summarize(predicted, true, scene):
    errors = np.abs(predicted - true[:, None])
    hits = errors <= 10.
    by_camera = hits.mean(axis=-1)
    pooled = by_camera.mean(axis=0)
    lift = pooled[3] - pooled[:3].max()
    indices = np.random.default_rng(591106 + scene).integers(len(hits), size=(5000, len(hits)))
    draws = by_camera[indices].mean(axis=1)
    bounds = np.percentile(draws[:, 3] - draws[:, :3].max(axis=1), [2.5, 97.5])
    false_singles = ~(hits[:, 0] | hits[:, 1])
    return dict(total=int(true.size), correct=dict(zip(ARMS, hits.sum(axis=(0, 2)).tolist(), strict=True)),
                median_abs_mm={k: float(np.nanmedian(errors[:, i])) for i, k in enumerate(ARMS)},
                joint_gain_over_strongest=float(lift), interval=bounds.tolist(),
                both_singles_wrong_joint_right=int((hits[:, 3] & false_singles).sum()),
                any_single_right_joint_wrong=int((~hits[:, 3] & ~false_singles).sum()),
                both_wrong_all_depth_controls_joint_right=int((hits[:, 3] & ~hits[:, :3].any(1)).sum()),
                gate=bool(lift >= .04 and bounds[0] > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    flags = parser.parse_args()
    started = time.perf_counter()
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "datasets/dtu/sample-pilot"
    published = HERE / "results/staged-manifest.json"
    assert digest(root / "pilot-manifest.json") == digest(published)
    manifest = json.loads(published.read_text())
    previous = HERE / "pilot-results"
    report = json.loads((previous / "result.json").read_text())
    assert report["decision"] == "below_pilot_gate" and report["manifest_sha256"] == digest(published)
    output = Path(os.environ["STUDIO_JOB_DIR"]) if not flags.smoke else None
    outcomes = {}
    for scene in manifest["scenes"]:
        panel = report["scenes"][str(scene)]["references"]
        if flags.smoke:
            panel = panel[:1]
        views = cameras(root, manifest, scene, panel)
        source = previous / f"scan-{scene}.npz"
        with np.load(source) as archive:
            n = len(panel)
            pixels, truth, older = (archive[k][:n] for k in ("pixels", "true", "predicted"))
        predictions = []
        keys = ("reference", "horizontal", "vertical")
        for i, row in enumerate(panel):
            predictions.append(predict(views, row, pixels[i], older[i]))
            print(f"Scan {scene}, reference {row['reference']}, verified {i + 1}/{len(panel)}", flush=True)
        estimates = np.stack(predictions)
        outcomes[str(scene)] = dict(**summarize(estimates, truth, scene), pilot_sha256=digest(source))
        if output is not None:
            np.savez_compressed(output / f"scan-{scene}.npz", predicted=estimates, true=truth, pixels=pixels,
                                cameras=np.array([[r[k] for k in keys] for r in panel]))
    result = dict(decision="joint_depth_costs_improve" if all(x["gate"] for x in outcomes.values()) else "unresolved",
                  scenes=outcomes, photo_manifest_sha256=digest(published), runtime_s=time.perf_counter()-started)
    if output is None:
        print(json.dumps(result))
    else:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
