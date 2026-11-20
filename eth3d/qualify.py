"""Qualify matched, differently directed stereo pairs from camera metadata."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

SCENES = ("courtyard", "delivery_area", "facade", "pipes")
MIN_AXIS = float(np.cos(np.pi / 6))


@dataclass
class View:
    key: int
    name: str
    rotation: np.ndarray
    center: np.ndarray
    tracks: set[int]


def rotation_matrix(values):
    w, x, y, z = np.asarray(values, dtype=np.float64) / np.linalg.norm(values)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def read_views(path):
    with path.open() as stream:
        records = [line.rstrip("\n") for line in stream if not line.startswith("#")]
    assert len(records) % 2 == 0
    views = []
    for header, values in zip(records[::2], records[1::2], strict=True):
        raw = header.split(maxsplit=9)
        assert len(raw) == 10
        rotate = rotation_matrix([float(x) for x in raw[1:5]])
        translate = np.array([float(x) for x in raw[5:8]])
        tracks = {int(s) for s in values.split()[2::3] if int(s) >= 0}
        views.append(View(int(raw[0]), raw[9], rotate, -rotate.T @ translate, tracks))
    assert len({v.key for v in views}) == len(views)
    return views


def candidate(ref, other):
    offset = ref.rotation @ (other.center - ref.center)
    lateral = np.linalg.norm(offset[:2])
    if lateral < .04 or abs(offset[2]) > .45 * np.linalg.norm(offset):
        return None
    fraction = np.abs(offset[:2] / lateral)
    axis = int(fraction.argmax())
    if fraction[axis] < MIN_AXIS:
        return None
    facing = float(np.degrees(np.arccos(np.clip(ref.rotation[2] @ other.rotation[2], -1, 1))))
    matches = len(ref.tracks & other.tracks)
    overlap = matches / max(len(ref.tracks), 1)
    if facing > 35 or matches < 100 or overlap < .15:
        return None
    return dict(id=other.key, image=other.name, axis=axis, offset=offset.tolist(),
                lateral=float(lateral), rotation_degrees=facing, shared_tracks=matches, overlap=overlap)


def choose(ref, views):
    axes = [[], []]
    for other in views:
        match = candidate(ref, other)
        if match is not None:
            axes[match["axis"]].append(match)

    candidates = []
    for horizontal in axes[0]:
        for vertical in axes[1]:
            lengths = (horizontal["lateral"], vertical["lateral"])
            ratio = max(lengths) / min(lengths)
            rotation = abs(horizontal["rotation_degrees"] - vertical["rotation_degrees"])
            overlaps = abs(horizontal["overlap"] - vertical["overlap"])
            if ratio <= 1.25 and rotation <= 10 and overlaps <= .15:
                cost = (ratio - 1) / .25 + rotation / 10 + overlaps / .15
                candidates.append((cost, horizontal["id"], vertical["id"], horizontal, vertical))
    if not candidates:
        return None
    _, _, _, x, y = min(candidates, key=lambda item: item[:3])
    return dict(reference_id=ref.key, reference=ref.name, horizontal=x, vertical=y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    panels = {}
    for scene in SCENES:
        folder = args.calibration_root / scene / "dslr_calibration_undistorted"
        images = folder / "images.txt"
        views = read_views(images)
        pairs = [item for view in views if (item := choose(view, views)) is not None]
        digests = {p.name: sha256(p.read_bytes()).hexdigest() for p in (images, folder / "cameras.txt")}
        panels[scene] = dict(cameras=len(views), triplets=len(pairs), digests=digests, pairs=pairs)
        print(f"{scene}: {len(pairs)} qualifying triplets from {len(views)} references", flush=True)
    eligible = [x["triplets"] for x in panels.values() if x["triplets"] >= 4]
    report = dict(decision="geometry_qualifies" if len(eligible) >= 3 and sum(eligible) >= 24 else "geometry_below_floor",
                  eligible_scenes=len(eligible), eligible_triplets=sum(eligible), scenes=panels)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
