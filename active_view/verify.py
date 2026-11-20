"""Check the matched camera budget and paired statistics from score archives."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
receipt = json.loads((ROOT / "receipt.json").read_text())
assert receipt["revision"] == "f095af9d7158dd4fa07cc7a03b951b816267aa90"
assert receipt["status"] == "done" and receipt["exit"] == 0
archived = json.loads((ROOT / "result.json").read_text())
outcomes = {}
for name in ("clean", "noisy"):
    with np.load(ROOT / f"{name}.npz") as z:
        rows, cue = z["counts"], z["planning_scores"]
    assert rows.shape == (8192, 7) and cue.shape == (8192, 2)
    index = np.arange(len(rows))
    panels = archived["panels"][name]
    assert (rows[:, 1:6] <= rows[:, :1]).all() and (rows >= 0).all()
    np.testing.assert_array_equal(rows[:, -1], cue.argmax(axis=1))
    np.testing.assert_array_equal(rows[:, 4], rows[index, 1 + cue.argmax(axis=1)])
    np.testing.assert_array_equal(rows[:, 5], np.maximum(rows[:, 1], rows[:, 2]))
    assert np.all((rows[:, 3] == rows[:, 1]) | (rows[:, 3] == rows[:, 2]))
    n, *correct, vertical = rows.sum(axis=0)
    assert panels["pixels"] == n and panels["chosen_vertical"] == vertical
    np.testing.assert_array_equal(correct, list(panels["correct"].values()))
    rng = np.random.default_rng(513053)
    differences = rows[:, 4, None] - rows[:, 1:4]
    draws = []
    for _ in range(4000):
        boot = rng.integers(len(rows), size=len(rows))
        draws.append(differences[boot].sum(axis=0) / rows[boot, 0].sum())
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    actual = differences.sum(axis=0) / n
    for j, label in enumerate(("horizontal", "vertical", "random")):
        estimate = panels["planned_minus"][label]
        np.testing.assert_allclose([estimate["mean"], *estimate["interval"]], [actual[j], low[j], high[j]])
    outcomes[name] = dict(accuracy={m: value / n for m, value in panels["correct"].items()},
                          planned_minus=panels["planned_minus"], layouts=len(rows))
assert archived["decision"] == "image_selects_view"
print(json.dumps(dict(receipt=receipt, panels=outcomes), indent=2))
