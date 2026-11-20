"""Compare single-camera selection rules on real photographs and measured depth."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from geometry import camera_summary
from visibility import DOWNSAMPLE, load_points, project

PATCH = np.array([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)])
DEPTHS = 1 / np.linspace(1 / 200, 1 / 900, 96)
OPTIONS = ("horizontal", "vertical", "random", "geometry", "structure", "repetition", "hindsight")


def read_image(path):
    with Image.open(path) as image:
        size = tuple(d // DOWNSAMPLE for d in image.size)
        reduced = image.convert("L").resize(size, Image.Resampling.BOX)
        return np.asarray(reduced, dtype=np.float32) / 255.


def bilinear(image, points):
    pixels = points - .5
    width, height = image.shape[::-1]
    good = ((pixels[..., 0] >= 0) & (pixels[..., 0] < width - 1)
            & (pixels[..., 1] >= 0) & (pixels[..., 1] < height - 1))
    pixels = np.where(good[..., None], pixels, 0.)
    grid = np.floor(pixels).astype(np.int32)
    dx, dy = np.moveaxis(pixels - grid, -1, 0)
    x, y = np.moveaxis(grid, -1, 0)
    top = (1 - dx) * image[y, x] + dx * image[y, x + 1]
    bottom = (1 - dx) * image[y + 1, x] + dx * image[y + 1, x + 1]
    return (1 - dy) * top + dy * bottom, good


def reference_directions(reference, targets):
    center, rotation = camera_summary(reference)
    facing = reference[:, :3] @ rotation[2]
    pupil = facing[:2] / facing[2]
    moves = []
    for target in targets:
        target_center, _ = camera_summary(target)
        ray = reference[:, :3] @ (target_center - center)
        steering = ray[:2] - pupil * ray[2]
        moves.append(steering / np.linalg.norm(steering))
    return moves


def photo_cues(frame, directions):
    height, width = frame.shape
    crop_x, crop_y = int(.1 * width), int(.1 * height)
    y, x = np.mgrid[crop_y:height-crop_y:4, crop_x:width-crop_x:4]
    positions = np.column_stack((x.ravel(), y.ravel())) + .5
    points = positions[:, None, :] + PATCH[None, :, :]
    reference, ok = bilinear(frame, points)
    assert ok.all()
    dy, dx = np.gradient(frame)
    gradients = np.column_stack((dx[crop_y:height-crop_y, crop_x:width-crop_x].ravel(),
                                 dy[crop_y:height-crop_y, crop_x:width-crop_x].ravel()))
    deltas = np.array([-3, -2, -1, 1, 2, 3])
    repetition, structure = [], []
    for direction in directions:
        samples, valid = bilinear(frame, points[:, None] + (deltas[:, None]*direction)[None, :, None])
        assert valid.all()
        nearest = np.abs(reference[:, None] - samples).mean(axis=-1).min(axis=1)
        repetition.append(float(np.percentile(nearest, 20)))
        structure.append(float(np.mean((gradients @ direction)**2)))
    return np.array(repetition), np.array(structure)


def score_depth(reference, secondary, pixels, reference_matrix, target_matrix, depths=DEPTHS):
    change = reference_matrix.copy()
    change[:2] /= DOWNSAMPLE
    target = target_matrix.copy()
    target[:2] /= DOWNSAMPLE
    inverse = np.linalg.inv(change[:, :3])
    center = -inverse @ change[:, 3]
    offset = target[:, :3] @ center + target[:, 3]
    z_scale = np.linalg.norm(reference_matrix[2, :3])
    chunks = []
    for row in np.array_split(pixels, max(1, (len(pixels) + 63) // 64)):
        points = row[:, None, :] + .5 + PATCH
        ref, good_ref = bilinear(reference, points)
        ref = ref - ref.mean(axis=-1, keepdims=True)
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
        slopes = (homogeneous @ inverse.T) @ target[:, :3].T * z_scale
        cameras = slopes[:, None] * depths[None, :, None, None] + offset
        positive = cameras[..., 2] > 0
        img = cameras[..., :2] / np.where(positive[..., None], cameras[..., 2:3], 1.)
        photometric, good_secondary = bilinear(secondary, img)
        photometric -= photometric.mean(axis=-1, keepdims=True)
        dot = (ref[:, None] * photometric).sum(axis=-1)
        norm = np.linalg.norm(ref, axis=-1)[:, None] * np.linalg.norm(photometric, axis=-1)
        valid = good_ref.all(axis=-1)[:, None] & (positive & good_secondary).all(axis=-1) & (norm > 1e-8)
        correlation = np.where(valid, dot / np.maximum(norm, 1e-8), -np.inf)
        estimate = depths[correlation.argmax(axis=1)]
        chunks.append(np.where(valid.any(axis=1), estimate, np.nan))
    return np.concatenate(chunks)


def room(root, manifest, screen, scene):
    points = load_points(root / f"Points/stl/stl{scene:03d}_total.ply")
    panels = {}
    for camera in manifest["camera_ids"]:
        path = root / f"Rectified/scan{scene}/rect_{camera:03d}_3_r5000.png"
        frame = read_image(path)
        matrix = np.loadtxt(root / f"calibration/pos_{camera:03d}.txt")
        front, seen = project(points, matrix, frame.shape[::-1])
        panels[camera] = (matrix, frame, front, seen)

    eligible = {r["reference"]: r["shared_pixels"] for r in screen["triplets"] if r["shared_pixels"] >= 500}
    rows = [r for r in manifest["triplets"] if r["reference"] in eligible]
    draws = np.random.default_rng(718403 + scene)
    random = draws.permutation(np.arange(len(rows)) % 2)
    report, references, truths, estimates, routes, cue_values = [], [], [], [], [], []

    for i, row in enumerate(rows):
        key, hx, vy = (row[k] for k in ("reference", "horizontal", "vertical"))
        p, reference, front, _ = panels[key]
        a, image_a, _, seen_a = panels[hx]
        b, image_b, _, seen_b = panels[vy]
        overlapping = front[seen_a[front] & seen_b[front]]
        assert len(overlapping) == eligible[key]
        selected = draws.choice(overlapping, size=512, replace=False)
        raw = points[selected] @ p[:, :3].T + p[:, 3]
        true = raw[:, 2] / np.linalg.norm(p[2, :3])
        px = np.floor(raw[:, :2] / (DOWNSAMPLE * raw[:, 2:3])).astype(int)
        estimate = np.stack([score_depth(reference, img, px, p, target)
                             for img, target in ((image_a, a), (image_b, b))])
        repetition, texture = photo_cues(reference, reference_directions(p, (a, b)))
        error = np.abs(estimate - true)
        hits = (error <= 10.).sum(axis=1)
        choices = [0, 1, int(random[i]), int(np.argmax(row["baseline_mm"])),
                   int(texture.argmax()), int(repetition.argmax()), int(hits.argmax())]
        report.append(dict(reference=key, horizontal=hx, vertical=vy, samples=len(px),
                           correct=hits.tolist(), cue=dict(repetition=repetition.tolist(),
                                                          structure=texture.tolist()),
                           choices=dict(zip(OPTIONS, choices, strict=True))))
        references.append(px)
        truths.append(true)
        estimates.append(estimate)
        routes.append(choices)
        cue_values.append(np.stack((repetition, texture)))
        print(f"Scan {scene}, reference {key}: {hits.tolist()} / {len(px)}", flush=True)

    evidence = dict(pixels=np.stack(references), true=np.stack(truths), predicted=np.stack(estimates),
                    routes=np.array(routes), cues=np.stack(cue_values))
    return report, evidence


def outcome(evidence):
    found = evidence["predicted"][np.arange(len(evidence["routes"]))[:, None], evidence["routes"]]
    abs_error = np.abs(found - evidence["true"][:, None])
    counts = (abs_error <= 10.).sum(axis=(0, 2))
    out = {}
    for i, key in enumerate(OPTIONS):
        err = abs_error[:, i]
        finite = np.isfinite(err)
        out[key] = dict(correct=int(counts[i]), total=len(evidence["true"])*evidence["true"].shape[1],
                        estimated=int(finite.sum()), median_abs_mm=float(np.median(err[finite])) if finite.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("visibility", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    begin = time.perf_counter()
    manifest_bytes = (args.dataset / "pilot-manifest.json").read_bytes()
    visibility_bytes = args.visibility.read_bytes()
    manifest = json.loads(manifest_bytes)
    screen = json.loads(visibility_bytes)
    assert screen["decision"] == "visible_triplets_qualify"
    assert sha256(manifest_bytes).hexdigest() == screen["manifest_digest"]
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = {}
    for scene in manifest["scenes"]:
        panels, evidence = room(args.dataset, manifest, screen["scenes"][str(scene)], scene)
        np.savez_compressed(args.output / f"scan-{scene}.npz", **evidence)
        scores = outcome(evidence)
        total = scores["repetition"]["total"]
        better = max(scores["horizontal"]["correct"], scores["vertical"]["correct"], scores["structure"]["correct"])
        scenes[str(scene)] = dict(scores=scores, references=panels,
                                  improvement_over_strongest_control=(scores["repetition"]["correct"] - better) / total)
    qualifies = all(s["improvement_over_strongest_control"] >= .05 for s in scenes.values())
    result = dict(decision="photo_selector_qualifies" if qualifies else "below_pilot_gate", scenes=scenes,
                  runtime_s=time.perf_counter()-begin, manifest_sha256=sha256(manifest_bytes).hexdigest(),
                  visibility_sha256=sha256(visibility_bytes).hexdigest())
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
