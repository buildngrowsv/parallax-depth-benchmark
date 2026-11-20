"""Plot same-budget fixed, randomized and selected camera outcomes."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
raw = json.loads((root / "audit.json").read_text())["panels"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.35, 3.7), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
arms = ("horizontal", "vertical", "random", "planned", "hindsight")
names = ("Fixed horizontal", "Fixed vertical", "Random", "Selected", "Hindsight")
for i, (panel, offset, color) in enumerate((("clean", -.12, "#73d8ec"), ("noisy", .12, "#fac884"))):
    accuracy = raw[panel]["accuracy"]
    ax.scatter([100 * accuracy[k] for k in arms], np.arange(len(arms)) + offset,
               s=42, color=color, label=panel.title(), marker=("o", "D")[i])
ax.set_yticks(range(len(arms)), names)
ax.set_xticks([97, 97.5, 98, 98.5, 99], ["97%", "97.5%", "98%", "98.5%", "99%"])
ax.set_xlim(96.95, 99.05)
ax.set_ylim(-.5, len(arms)-.5)
ax.invert_yaxis()
ax.set_xlabel("Correct depth across all foreground pixels")
ax.grid(alpha=.45, color="#3e4c60")
ax.set_axisbelow(True)
ax.spines[["right", "top"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig(root / "active.png", dpi=160)
