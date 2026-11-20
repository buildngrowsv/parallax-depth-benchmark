"""Show the camera geometry for one simulated scene and count paired recoveries."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run import project,textured_layout

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())["subsets"]["exact_alias_pairs"]
rng=np.random.default_rng(1605831)
reference,owner,_=textured_layout(rng)
depth=rng.permutation(4)[owner]+1
b=int(rng.choice((-1,1)))
views=[reference,*[project(reference,depth,delta)[0] for delta in ((0,b),(0,-b),(b,0))]]
plt.style.use("dark_background")
fig=plt.figure(figsize=(10.6,5),layout="constrained",facecolor="#101916")
grid=fig.add_gridspec(2,4,height_ratios=(1,1.65))
for j,(title,frame) in enumerate(zip(("Reference","Horizontal","Opposite","Perpendicular"),views,strict=True)):
    ax=fig.add_subplot(grid[0,j])
    ax.imshow(frame,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
    ax.set_title(title,fontsize=11)
    ax.axis("off")
ax=fig.add_subplot(grid[1,:])
ax.set_facecolor("#101916")
keys=("first_horizontal","opposite_horizontal","perpendicular","wrong_perpendicular")
bars=ax.barh(range(4),[d["correct"][k] for k in keys],color=["#a7b8d7","#a7b8d7","#85dabc","#a7b8d7"])
ax.bar_label(bars,padding=4)
ax.set(xlim=(0,4500),yticks=range(4),yticklabels=("First horizontal","Add opposite","Add perpendicular","Mismatched perpendicular"),
       xlabel="Both depth variants correct, out of 4,057 exact alias pairs")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(root/"orthogonal-depth.png",dpi=165)
