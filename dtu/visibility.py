"""Check shared visibility from the official structured-light point clouds."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np

DOWNSAMPLE = 4
DEPTH_TOLERANCE_MM = 5.
PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
             "uint": "u4", "float": "f4", "double": "f8"}


def load_points(path):
    with path.open("rb") as reader:
        assert reader.readline().strip() == b"ply"
        kind = reader.readline().split()
        assert kind in ([b"format", b"binary_little_endian", b"1.0"],
                        [b"format", b"binary_big_endian", b"1.0"])
        byteorder = "<" if kind[1] == b"binary_little_endian" else ">"
        fields, number, vertex = [], None, False
        for line in reader:
            tokens = line.decode("ascii").split()
            if tokens[0] == "end_header":
                break
            if tokens[0] == "element":
                vertex = tokens[1] == "vertex"
                if vertex:
                    number = int(tokens[2])
            if tokens[0] == "property" and vertex:
                assert len(tokens) == 3
                fields.append((tokens[2], byteorder + PLY_TYPES[tokens[1]]))
        assert number is not None and {"x", "y", "z"} <= {x[0] for x in fields}
        data = np.fromfile(reader, dtype=fields, count=number)
        assert len(data) == number
    points = np.column_stack([data[k] for k in ("x", "y", "z")])
    return points[np.isfinite(points).all(1)]


def png_shape(path):
    with path.open("rb") as source:
        header = source.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
    return tuple(d // DOWNSAMPLE for d in struct.unpack(">II", header[16:24]))


def project(points, projection, shape):
    rays = points @ projection[:, :3].T + projection[:, 3]
    raw = rays[:, 2]
    z = raw / np.linalg.norm(projection[2, :3])
    plane = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(raw) & (raw > 0)
    plane[valid] = rays[valid, :2] / (raw[valid, None] * DOWNSAMPLE)
    w, h = shape
    inside = valid & (plane[:, 0] >= 2) & (plane[:, 0] < w - 2) & (plane[:, 1] >= 2) & (plane[:, 1] < h - 2)
    idx = np.nonzero(inside)[0]
    xy = np.floor(plane[idx]).astype(np.int32)
    flat = xy[:, 0] + xy[:, 1] * w
    order = np.lexsort((z[idx], flat))
    first = np.r_[True, flat[order][1:] != flat[order][:-1]] if len(order) else np.empty(0, bool)
    front = idx[order[first]]
    depth_map = np.full(w*h, np.inf)
    depth_map[flat[order[first]]] = z[front]
    seen = np.zeros(len(points), bool)
    seen[idx] = z[idx] <= depth_map[flat] + DEPTH_TOLERANCE_MM
    return front, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.dataset
    manifest_file = root / "pilot-manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for key, value in manifest["files"].items():
        path = root / key
        assert path.stat().st_size == value["bytes"]
        with path.open("rb") as stream:
            assert sha256(stream.read()).hexdigest() == value["sha256"]
    checked = {}
    for scene in manifest["scenes"]:
        points = load_points(root / f"Points/stl/stl{scene:03d}_total.ply")
        panels = {}
        for cam in manifest["camera_ids"]:
            img = root / f"Rectified/scan{scene}/rect_{cam:03d}_3_r5000.png"
            p = np.loadtxt(root / f"calibration/pos_{cam:03d}.txt")
            panels[cam] = project(points, p, png_shape(img))
        counts = []
        for row in manifest["triplets"]:
            front = panels[row["reference"]][0]
            seen_x = panels[row["horizontal"]][1][front]
            seen_y = panels[row["vertical"]][1][front]
            counts.append(dict(reference=row["reference"], horizontal=row["horizontal"], vertical=row["vertical"],
                               reference_pixels=len(front), horizontal_pixels=int(seen_x.sum()),
                               vertical_pixels=int(seen_y.sum()), shared_pixels=int((seen_x & seen_y).sum())))
        checked[str(scene)] = dict(scan_points=len(points),
                                   qualifying=sum(item["shared_pixels"] >= 500 for item in counts), triplets=counts)
        print(f"Scan {scene}: {checked[str(scene)]['qualifying']}/{len(counts)} qualify", flush=True)
    report = dict(decision="visible_triplets_qualify" if all(s["qualifying"] >= 8 for s in checked.values())
                  else "visibility_below_floor", manifest_digest=sha256(manifest_file.read_bytes()).hexdigest(),
                  downsample=DOWNSAMPLE, tolerance_mm=DEPTH_TOLERANCE_MM, scenes=checked)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
