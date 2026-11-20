"""Stage two official DTU scenes and only the camera views needed for a pilot."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from geometry import RangedFile, URL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = json.loads(args.geometry.read_text())
    assert source["decision"] == "dtu_pose_geometry_qualifies"
    sampled = np.linspace(0, len(source["pairs"]) - 1, 12, dtype=int)
    triplets = [source["pairs"][i] for i in sampled]
    camera_ids = sorted({row[key] for row in triplets for key in ("reference", "horizontal", "vertical")})
    assignments = {}
    for scene in (1, 6):
        for key in camera_ids:
            rel = f"Rectified/scan{scene}/rect_{key:03d}_3_r5000.png"
            assignments[f"SampleSet/MVS Data/{rel}"] = rel
        for rel in (f"Points/stl/stl{scene:03d}_total.ply",):
            assignments[f"SampleSet/MVS Data/{rel}"] = rel
    for entry in source["calibration_sha256"]:
        assignments[entry] = f"calibration/{Path(entry).name}"
    args.destination.mkdir(parents=True, exist_ok=True)
    files = {}
    with ZipFile(RangedFile(URL)) as archive:
        for entry in archive.infolist():
            if entry.filename not in assignments:
                continue
            local_name = assignments[entry.filename]
            path = args.destination / local_name
            path.parent.mkdir(parents=True, exist_ok=True)
            values = archive.read(entry)
            digest = sha256(values).hexdigest()
            if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(values)
                temporary.replace(path)
            files[local_name] = {"bytes": len(values), "sha256": digest}
            if len(files) % 16 == 0:
                print(f"Verified {len(files)}/{len(assignments)} source files", flush=True)
    assert len(files) == len(assignments)
    manifest = dict(source=URL, scenes=[1, 6], triplets=triplets, camera_ids=camera_ids, files=files)
    (args.destination / "pilot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged two scans, {len(camera_ids)} positions and {len(triplets)} matched triplets", flush=True)


if __name__ == "__main__":
    main()
