"""Create the result figure from the first shifted layout and the reported scores."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scenes import render

R=Path(__file__).resolve().parent/"results"
example=json.loads((R/"example.json").read_text())
scores=json.loads((R/"summary.json").read_text())
image=np.array(example["image"])/255
truth=np.array(example["worlds"][0]["truth"])
pred=np.array(example["worlds"][0]["trained"])
colors=["#131d28","#56677f","#5179a0","#53aa9b","#ffd385"]
fig=plt.figure(figsize=(11.8,6.5),layout="constrained")
grid=GridSpec(2,4,figure=fig,height_ratios=[.77,1.3])
fig.suptitle("What depth can a second view reveal?",fontsize=20,weight="bold")
for i,(title,im) in enumerate((("Reference image",image),("Second camera",render(image,truth,example["baseline"])),
                              ("True depth",truth),("Learned depth",pred))):
    ax=fig.add_subplot(grid[0,i])
    ax.imshow(im,cmap="gray" if i<2 else ListedColormap(colors),vmin=0,vmax=1 if i<2 else 4,interpolation="nearest")
    ax.set_title(title,fontsize=11)
    ax.axis("off")
for panel, g, title in (("shift", grid[1,:2],"Depth accuracy on unseen shapes"),("search",grid[1,2:],"Removing the calibrated shortlist")):
    ax=fig.add_subplot(g)
    values=scores[panel]["scores"]
    if panel=="shift":
        labels=["One view","Untrained","Image-trained","Patch difference"]
        nums=[values["single"]["all"]["exact"],np.mean([values[f"initial_{s}"]["all"]["exact"] for s in (101,211,307)]),
              np.mean([values[f"trained_{s}"]["all"]["exact"] for s in (101,211,307)]),values["stereo"]["all"]["exact"]]
        col=["#acb4bc","#acb4bc","#297a70","#254e71"]
    else:
        labels=["Learned · 4 shifts","Learned · 17","Fixed · 4","Fixed · 17"]
        nums=[np.mean([values[f"{k}_{s}"]["all"]["exact"] for s in (101,211,307)]) for k in ("four","dense")]
        nums+= [values[f"{k}_stereo"]["all"]["exact"] for k in ("four","dense")]
        col=["#297a70","#5d9d91","#254e71","#5986a8"]
    bars=ax.barh(labels,100*np.array(nums),color=col,height=.58)
    ax.bar_label(bars,fmt="%.1f%%",padding=3,fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel("Correct foreground pixels (%)")
    ax.set_title(title,fontsize=12,loc="left",weight="bold")
    ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
    ax.set_axisbelow(True);ax.grid(axis="x",color="#dde2e5")
fig.savefig(R/"parallax.png",dpi=170,bbox_inches="tight")

