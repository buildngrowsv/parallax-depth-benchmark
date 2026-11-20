"""Tabulate which reference cameras keep a horizontal and vertical candidate."""
import argparse
import json
from pathlib import Path

import numpy as np

from qualify import MIN_AXIS, SCENES, candidate, read_views


def count_axes(views):
    counts = np.zeros((3, 2), dtype=int)
    for reference in views:
        raw, lateral, eligible = set(), set(), set()
        for other in views:
            offset = reference.rotation @ (other.center - reference.center)
            transverse = np.linalg.norm(offset[:2])
            if transverse < .04:
                continue
            unit = np.abs(offset[:2]) / transverse
            axis = int(unit.argmax())
            if unit[axis] < MIN_AXIS:
                continue
            raw.add(axis)
            if abs(offset[2]) > .45 * np.linalg.norm(offset):
                continue
            lateral.add(axis)
            if candidate(reference, other) is not None:
                eligible.add(axis)
        for row, remaining in enumerate((raw, lateral, eligible)):
            counts[row, :] += (0 in remaining, 1 in remaining)
    return dict(cameras=len(views), by_axis=dict(zip(("direction_only", "lateral", "all_individual_gates"),
                                                   counts.tolist(), strict=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_root", type=Path)
    args = parser.parse_args()
    result = {scene: count_axes(read_views(args.calibration_root / scene / "dslr_calibration_undistorted/images.txt"))
              for scene in SCENES}
    print(json.dumps(result, indent=2))
