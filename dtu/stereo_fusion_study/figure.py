"""Plot matched real-photo depth accuracy under the four registered policies."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
data = json.loads((here / "audit.json").read_text())
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.1, 4.5), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
arms = ("horizontal", "vertical", "mean_estimates", "joint_cost")
names = ("Horizontal", "Vertical", "Mean of estimates", "Joint photometric cost")
for i, (arm, label, color) in enumerate(zip(arms, names, ("#b6acf0", "#a7b4ae", "#e8bc79", "#84dfc2"), strict=True)):
    values = [100 * data[s]["correct"][arm] / data[s]["total"] for s in ("1", "6")]
    bars = ax.bar(np.arange(2)+(i-1.5)*.18, values, width=.17, color=color, label=label)
    ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
ax.set(title="Combining photo evidence before choosing depth", ylabel="Pixels within 10 mm (%)", ylim=(0, 57))
ax.set_xticks([0, 1], ["DTU scan 1", "DTU scan 6"])
ax.grid(axis="y", alpha=.18)
ax.legend(ncol=2, frameon=False, loc="upper center")
fig.savefig(here / "fusion.png", dpi=170)
