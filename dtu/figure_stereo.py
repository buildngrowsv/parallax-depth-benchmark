"""Plot matched-budget accuracy and retrospective headroom in dark mode."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "pilot-results/audit.json").read_text())["scenes"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
parts = ("horizontal", "vertical", "random", "geometry", "structure", "repetition", "hindsight")
labels = ("Fixed horizontal", "Fixed vertical", "Random", "Larger baseline", "Structure tensor", "Patch repetition", "Hindsight")
for scene, offset, color, marker in (("1", -.13, "#75d7f3", "o"), ("6", .13, "#ffc184", "D")):
    values = [100 * data[scene]["scores"][arm]["correct"] / data[scene]["scores"][arm]["total"] for arm in parts]
    ax.scatter(values, np.arange(len(parts)) + offset, color=color, marker=marker, label=f"Scan {scene}", s=38)
ax.set_yticks(range(len(parts)), labels)
ax.invert_yaxis()
ax.set_xlim(12.5, 19.4)
ax.set_xticks([13, 14, 15, 16, 17, 18, 19], [f"{x}%" for x in range(13, 20)])
ax.set_xlabel("Estimated depth within 10 mm of the scan")
ax.grid(axis="both", alpha=.4, color="#44566b")
ax.spines[["right", "top"]].set_visible(False)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False)
fig.savefig(root / "dtu-pilot.png", dpi=160)
