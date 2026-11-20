"""Recompute real-photo errors, matched view contrasts and new correct pixels."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
root = here / "results"
receipt = json.loads((root / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "dtu-joint-photo-depth-costs", "a0bffd5f97b2aff5b4f1c495058b3fd8e44287f7", "done", 0)
result = json.loads((root / "result.json").read_text())
manifest = here.parent / "results/staged-manifest.json"
assert sha256(manifest.read_bytes()).hexdigest() == result["photo_manifest_sha256"]
pilot = here.parent / "pilot-results"
prior = json.loads((pilot / "result.json").read_text())["scenes"]
reviewed = {}
arms = ("horizontal", "vertical", "mean_estimates", "joint_cost")
for scene in (1, 6):
    stated = result["scenes"][str(scene)]
    with np.load(root / f"scan-{scene}.npz") as new, np.load(pilot / f"scan-{scene}.npz") as old:
        assert sha256((pilot / f"scan-{scene}.npz").read_bytes()).hexdigest() == stated["pilot_sha256"]
        for k in ("pixels", "true"):
            np.testing.assert_array_equal(old[k], new[k])
        references = prior[str(scene)]["references"]
        np.testing.assert_array_equal(new["cameras"], [[r[k] for k in ("reference", "horizontal", "vertical")] for r in references])
        values, truth = new["predicted"], new["true"]
        assert values.shape == (12, 4, 512) and truth.shape == (12, 512)
        np.testing.assert_array_equal(values[:, 2], (values[:, 0] + values[:, 1]) / 2)
        errors = abs(values - truth[:, None])
    hit = errors <= 10
    counts = hit.sum(axis=(0, 2))
    assert counts.tolist() == [stated["correct"][k] for k in arms]
    np.testing.assert_allclose(np.nanmedian(errors.transpose(1, 0, 2).reshape(4, -1), axis=1),
                               [stated["median_abs_mm"][k] for k in arms])
    new_singles = hit[:, 3] & ~hit[:, :2].any(axis=1)
    lots = ~hit[:, 3] & hit[:, :2].any(axis=1)
    delta = (counts[3] - counts[:3].max()) / truth.size
    samps = np.random.default_rng(591106 + scene).integers(12, size=(5000, 12))
    boots = hit.sum(axis=2)[samps].sum(axis=1) / truth.size
    interval = np.percentile(boots[:, 3] - boots[:, :3].max(axis=1), [2.5, 97.5])
    np.testing.assert_allclose(delta, stated["joint_gain_over_strongest"])
    np.testing.assert_allclose(interval, stated["interval"])
    assert int(new_singles.sum()) == stated["both_singles_wrong_joint_right"]
    assert int(lots.sum()) == stated["any_single_right_joint_wrong"]
    assert (hit[:, 3] & ~hit[:, :3].any(axis=1)).sum() == stated["both_wrong_all_depth_controls_joint_right"]
    assert delta >= .04 and interval[0] > 0 and stated["gate"]
    reviewed[str(scene)] = stated
assert result["decision"] == "joint_depth_costs_improve"
print(json.dumps(reviewed, indent=2))
