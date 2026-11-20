"""Route a one-camera budget before any translated image is available."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orthogonal.run import H, W, PATCH, layout, patch_cost, project

ARMS = ("horizontal", "vertical", "random", "planned", "hindsight")
PANELS = {"clean": (830158, 0.), "noisy": (511804, .025)}
LAYOUTS = 8192


def draw(rng):
    _, owner, _ = layout(rng, novel=True)
    horizontal = np.full(4, rng.integers(2) == 1)
    horizontal[rng.integers(4)] ^= True
    image = np.zeros(owner.shape, np.float32)
    for surface, repeat_x in enumerate(horizontal):
        y, x = np.nonzero(owner == surface)
        if repeat_x:
            tex = rng.uniform(.18, .88, size=(H, 3)).astype(np.float32)
            image[y, x] = tex[y, x % 3]
        else:
            tex = rng.uniform(.18, .88, size=(W, 3)).astype(np.float32)
            image[y, x] = tex[x, y % 3]
    start = rng.permutation(4)
    depths = [(start[owner] + phase) % 4 + 1 for phase in range(4)]
    return image, depths


def observe(image, rng, noise):
    return image if noise == 0 else np.clip(image + rng.normal(0, noise, image.shape), 0, 1).astype(np.float32)


def predict_direction(reference):
    patch_centers = reference.copy() > .1
    patch_centers[:4] = patch_centers[-4:] = False
    patch_centers[:, :4] = patch_centers[:, -4:] = False
    y, x = np.nonzero(patch_centers)
    if len(y) == 0:
        return 0, [0., 0.]
    yy = y[:, None] + PATCH[:, 0]
    xx = x[:, None] + PATCH[:, 1]
    core = reference[yy, xx]
    deltas = np.array([-3, -2, -1, 1, 2, 3])[None, :, None]
    row = yy[:, None, :]
    col = xx[:, None, :]
    scores = []
    for a, b in ((0, 1), (1, 0)):
        guesses = reference[row + a * deltas, col + b * deltas]
        nearest = np.abs(core[:, None] - guesses).mean(axis=2).min(axis=1)
        scores.append(float(np.percentile(nearest, 20)))
    return int(np.argmax(scores)), scores


def sample(rng, noise):
    clean, depths = draw(rng)
    reference = observe(clean, rng, noise)
    plan, cue = predict_direction(reference)
    random_choice = rng.integers(2)
    sign = int(rng.choice((-1, 1)))
    y, x = np.nonzero(clean)
    hits = []
    for move in ((0, sign), (sign, 0)):
        correct = 0
        for depth in depths:
            recorded = observe(project(clean, depth, move)[0], rng, noise)
            guessed = patch_cost(reference, recorded, move, y, x).argmin(1) + 1
            correct += int((depth[y, x] == guessed).sum())
        hits.append(correct)
    return [4 * len(y), *hits, hits[random_choice], hits[plan], max(hits), plan], cue


def summarize(counts):
    total = counts.sum(axis=0)
    n = total[0]
    summary = dict(pixels=int(n), layouts=len(counts), chosen_vertical=int(total[-1]),
                   correct=dict(zip(ARMS, total[1:-1].tolist(), strict=True)))
    rng = np.random.default_rng(513053)
    paired = counts[:, 4:5] - counts[:, 1:4]
    differences = []
    for _ in range(40):
        indices = rng.integers(len(counts), size=(100, len(counts)))
        differences.extend(paired[indices].sum(axis=1) / counts[indices, 0].sum(axis=1, keepdims=True))
    bounds = np.percentile(differences, [2.5, 97.5], axis=0).T
    summary["planned_minus"] = {
        name: dict(mean=float(effect), interval=ci.tolist())
        for name, effect, ci in zip(ARMS[:3], paired.sum(axis=0) / n, bounds, strict=True)
    }
    return summary


def main():
    start = time.perf_counter()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    reports = {}
    for panel, (seed, noise) in PANELS.items():
        rng = np.random.default_rng(seed)
        outcomes = [sample(rng, noise) for _ in range(LAYOUTS)]
        raw, scores = map(np.array, zip(*outcomes, strict=True))
        np.savez_compressed(out / (panel + ".npz"), counts=raw, planning_scores=scores)
        reports[panel] = summarize(raw)
        print(panel, reports[panel], flush=True)
    passed = all(v["mean"] >= .005 and v["interval"][0] > 0
                 for panel in reports.values() for v in panel["planned_minus"].values())
    report = dict(decision="image_selects_view" if passed else "unresolved",
                  runtime_s=time.perf_counter() - start, panels=reports)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
