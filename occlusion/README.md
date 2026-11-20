# When does a third camera recover depth on an occluded surface?

An opposite-side camera raised exact depth accuracy from 36.7% to 92.6% on synthetic pixels hidden from the first translated view and visible to the second. On an independent panel with a smaller camera displacement the gain was 49.4% to 82.2%. Across all foreground pixels, combining the views reduced accuracy on both panels. A round-trip consistency rule applied to two further panels also lost more correct pixels than it rescued.

## A controlled change in what the cameras can see

Each layout has four possible depth assignments rendering the same reference image. The matcher evaluates calibrated depth candidates with mean absolute error on 3×3 image patches. The first translated image supplies one error per candidate. With an opposite view available, fusion selects the smaller error for each candidate before choosing its depth. The matcher sees only images and camera geometry. Renderer IDs are used solely to identify newly visible pixels for scoring. Duplicating the first view leaves every prediction unchanged. A wrong opposite view from a different depth assignment of the same layout tests whether a corresponding image is necessary. Classical or learned stereo methods can use richer matching and regularization. This experiment isolates the camera signal in a fixed local matcher. [Preregistered camera study](PLAN.md) · [Kendall et al. on stereo matching](https://arxiv.org/abs/1703.04309)

| New layouts, each with four depth assignments | Single correct on newly visible pixels | Matched third view | Mismatched third view | Gains / losses with matched view | Change over all foreground |
|---|---:|---:|---:|---:|---:|
| 512 at the larger displacement | 2,252 / 6,135 | 5,680 / 6,135 | 484 / 6,135 | 3,443 / 15 | −7,958 / 500,140 |
| 512 at the smaller displacement | 719 / 1,456 | 1,197 / 1,456 | 165 / 1,456 | 508 / 30 | −10,170 / 497,880 |

The 95% paired intervals for the gains on newly visible pixels were 53.1 to 58.5 percentage points and 28.8 to 37.3 points. Each bootstrap draw keeps all four depth assignments of a layout together. Neither panel contained reference pixels hidden in both translated views, so that case has no estimate. The synthetic baseline, discrete depths, fixed renderer and local texture limit the inference. These results measure geometric information in paired images. They do not identify what any pretrained generative model learned from its training data. [First panel](results/pilot/result.json) · [Confirmation](results/confirm/result.json)

## Can a round-trip check spend the extra view more selectively?

On two fresh 512-layout panels we projected the first match back to the starting image. When it missed the original pixel, the router used the opposite camera's estimate. A matched-budget control substituted the opposite camera at exactly as many randomly chosen pixels per world. The rule used the other camera on 31,353/495,508 and 21,193/497,212 foreground pixels. It recovered 3,427 and 445 newly visible pixels while losing 40 and 23 in the same subsets. Across all pixels the rule lost a net 7,697 and 10,311 correct estimates relative to the single-view baseline. Its overall accuracy fell below the matched random control in both cases. Reverse matching also requires computation, so the fraction routed measures opposite-view use alone. The planned promotion gate closed. [Routing protocol](ROUTING.md) · [Routing counts and intervals](results/routing/result.json)

The public evidence includes one count vector for each world and subset. The audit reaggregates 8,192 worlds, checks the duplicate-view identity per world and verifies the gains-minus-losses identities before matching every published total. Run `python occlusion/verify.py occlusion/results` with NumPy. [Audit implementation](verify.py) · [Audit output](results/audit.json) · [Source for view matching](run.py) · [Routing source](route.py)

The account owner proposed investigating what 3D information a model can obtain from its training inputs. OpenAI Codex designed and ran the comparisons over SSH on the owner's Apple M5 Ultra, audited the scores and assisted with this manuscript.

[Published Pubroot paper](https://pubroot.com/ai/computer-vision/when-does-a-third-camera-recover-depth-on-an-occluded-surface/) · [Pubroot submission and review record](https://github.com/buildngrowsv/pubroot-website/issues/359)
