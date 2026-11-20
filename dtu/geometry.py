"""Read only the calibration entries of the official DTU robot capture archive."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from functools import lru_cache
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np

URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
BLOCK = 1024 * 1024


class RangedFile:
    def __init__(self, url):
        self.url = url
        with urlopen(Request(url, method="HEAD"), timeout=15) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        base = (0, self.pos, self.size)[whence]
        self.pos = base + offset
        return self.pos

    @lru_cache(maxsize=20)
    def block(self, index):
        start = index * BLOCK
        end = min(start + BLOCK, self.size) - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError(f"server did not accept a bounded range: {response.status}")
            content = response.read()
        if len(content) != end - start + 1:
            raise RuntimeError("incomplete range")
        return content

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        data = []
        while self.pos < end:
            index, offset = divmod(self.pos, BLOCK)
            piece = self.block(index)[offset:offset + end - self.pos]
            data.append(piece)
            self.pos += len(piece)
        return b"".join(data)


PREFIX = "SampleSet/MVS Data/Calibration/cal18"


def camera_summary(matrix):
    block = matrix[:, :3]
    if np.linalg.det(block) < 0:
        block = -block
    row_z = block[2] / np.linalg.norm(block[2])
    row_y = block[1] - np.dot(block[1], row_z) * row_z
    row_y /= np.linalg.norm(row_y)
    return -np.linalg.solve(matrix[:, :3], matrix[:, 3]), np.stack((np.cross(row_y, row_z), row_y, row_z))


def paired_options(positions, rotations):
    panels = []
    for index, (origin, rotation) in enumerate(zip(positions, rotations, strict=True)):
        h, v = [], []
        for other, center in enumerate(positions):
            offset = rotation @ (center - origin)
            extent = np.linalg.norm(offset[:2])
            if extent < 40. or abs(offset[2]) > .45 * np.linalg.norm(offset):
                continue
            scaled = np.abs(offset[:2]) / extent
            axis = int(scaled.argmax())
            relative_angle = np.degrees(np.arccos(np.clip(rotation[2] @ rotations[other, 2], -1, 1)))
            if scaled[axis] >= np.cos(np.pi / 6) and relative_angle <= 35:
                (h, v)[axis].append((other + 1, float(extent), float(relative_angle)))
        pairs = []
        for x in h:
            for y in v:
                ratio = max(x[1], y[1]) / min(x[1], y[1])
                contrast = abs(x[2] - y[2])
                if ratio <= 1.25 and contrast <= 10:
                    score = (ratio - 1) / .25 + contrast / 10
                    pairs.append((score, x, y, ratio, contrast))
        if pairs:
            _, x, y, ratio, contrast = min(pairs)
            panels.append(dict(reference=index + 1, horizontal=x[0], vertical=y[0], baseline_mm=[x[1], y[1]],
                               view_rotation_degrees=[x[2], y[2]], baseline_ratio=ratio, rotation_difference=contrast))
    return panels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    matrices = []
    digests = {}
    with ZipFile(RangedFile(URL)) as archive:
        for index in range(1, 50):
            name = f"{PREFIX}/pos_{index:03d}.txt"
            raw = archive.read(name)
            matrices.append(np.loadtxt(raw.splitlines()))
            digests[name] = sha256(raw).hexdigest()
    assert all(matrix.shape == (3, 4) for matrix in matrices)
    positions, rotations = map(np.stack, zip(*(camera_summary(p) for p in matrices), strict=True))
    np.testing.assert_allclose(np.linalg.det(rotations), np.ones(len(rotations)), atol=1e-9)
    pairs = paired_options(positions, rotations)
    report = dict(source=URL, cameras=len(matrices), matched_triplets=len(pairs),
                  decision="dtu_pose_geometry_qualifies" if len(pairs) >= 12 else "dtu_pose_geometry_below_floor",
                  calibration_sha256=digests, pairs=pairs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(pairs)} matched-pose combinations from {len(matrices)} camera positions")


if __name__ == "__main__":
    main()
