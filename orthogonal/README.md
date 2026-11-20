# A perpendicular camera resolved 3,939 depth aliases left tied by collinear cameras

With texture repeating every three horizontal pixels, depths 1 and 4 produced identical local matches in both horizontal camera directions. The perpendicular view recovered both correct depths on **3,939 of 4,057** aligned pixel pairs. An opposite horizontal view and a perpendicular image from the wrong depth assignment each recovered zero of those pairs. The comparison uses calibrated synthetic stereo and fixed image matching. The additional evidence comes from camera direction.

![Example reference and camera frames above the paired depth-recovery counts](orthogonal-depth.png)

## Design

Each of 1,024 independently generated layouts has four surfaces. Two have intensity windows that repeat every three columns and vary across rows. The other two repeat every three rows and vary across columns. Four cyclic assignments of discrete depth preserve the original reference image. At depth level d an x-directed camera shifts a visible sample by d columns. A y-directed camera shifts it by d rows. View spacing has the same magnitude in both directions. Nearer occupied samples replace farther samples on collision. The matcher sees the reference frame, translated images and camera offsets. For each level it computes the mean absolute difference of two 3-by-3 patches. When two images are used, their per-level costs are added before selecting the lowest-cost depth. [Registered plan](PLAN.md) · [Renderer and matcher](run.py) · [Multiple-baseline stereo](https://publications.ri.cmu.edu/a-multiple-baseline-stereo-2)

The paired endpoint uses horizontally repeating reference pixels present at both depth levels 1 and 4 in the corresponding worlds. In both worlds, each horizontal view must give zero cost to both depths, and the true patch must be fully visible. The renderer's source IDs are used only for that scoring subset. A pair succeeds when both worlds get their correct depth. The horizontal comparator deliberately receives a second view with the same orientation of displacement. It leaves this exact degeneracy intact. The mismatched control renders the perpendicular frame from the next depth world of the same layout. [Per-layout counts](results/per-layout.npy)

| Camera condition | Both alias depths correct, out of 4,057 pairs | All foreground, out of 996,980 instances |
|---|---:|---:|
| First horizontal | 0 | 967,775 |
| Add opposite horizontal | 0 | 988,133 |
| Add corresponding perpendicular | 3,939 | 993,895 |
| Add mismatched perpendicular | 0 | 427,111 |

The perpendicular view gained 97.09 percentage points in the exact alias subset versus either additional-view control. The layout-paired 95% interval was +96.30 to +97.84. Across all foreground, it added 5,762 correct instances over the opposite-view arm. The gain concentrated on horizontally repeating surfaces. On vertically repeating surfaces the opposite view gave 498,556 correct and the perpendicular view gave 497,904. These outcomes describe the generated texture and calibrated camera choices. Transferring the result to natural images or to learned 3D representations needs separate evidence. [Audited scores for both texture orientations](audit.json) · [Verifier](verify.py)

Run `uv run --with numpy python orthogonal/verify.py` to reconstruct every layout and recompute the paired counts and intervals. The NumPy run took three seconds on the owner's Apple M5 Ultra. The owner proposed distinguishing geometric information available in images from directly supplied 3D data. OpenAI Codex designed and implemented the comparison, ran it, verified it and assisted with writing. [Version-pinned Studio receipt](results/receipt.json)

[Published Pubroot article](https://pubroot.com/ai/computer-vision/a-perpendicular-camera-resolved-3-939-depth-aliases-left-tied-by-collinear-cameras/) · [Pubroot submission and review](https://github.com/buildngrowsv/pubroot-website/issues/453)
