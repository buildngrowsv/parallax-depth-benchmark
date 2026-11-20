# Four ETH3D scenes lacked matched horizontal and vertical camera options

The geometry-only screen found **zero qualifying triplets across 172 reference cameras** in four ETH3D scenes. The main limit was the vertical candidate. After the direction and lateral-movement requirements, three scenes had none. Delivery area had four reference cameras with a vertical option, and none paired it with a horizontal option that met the matched-baseline criteria. This result concerns acquisition geometry. The screen used no pixel values or depth errors. [Registered criteria](PLAN.md) · [Camera counts and hashes](results/qualification.json) · [Attrition audit](results/attrition.json)

| Scene | Reference cameras | References with an eligible horizontal candidate | References with an eligible vertical candidate | Matched triplets |
|---|---:|---:|---:|---:|
| Courtyard | 38 | 36 | 0 | 0 |
| Delivery area | 44 | 39 | 4 | 0 |
| Facade | 76 | 63 | 0 | 0 |
| Pipes | 14 | 13 | 0 | 0 |

The camera centers, orientations and shared feature tracks come from the official calibrated, undistorted ETH3D high-resolution training images. The underlying benchmark provides independent laser-based reference geometry. A fair horizontal-versus-vertical acquisition comparison also needs comparable translated views. This sample lacked that matched geometry at the registered limits. We retained the data and the zero-triplet result. A different capture trajectory can reopen the depth test. [ETH3D dataset](https://www.eth3d.net/datasets) · [Calibration format](https://www.eth3d.net/documentation) · [Schöps et al., CVPR 2017](https://www.eth3d.net/data/schoeps2017cvpr.pdf)

To reproduce, extract `<scene>/dslr_calibration_undistorted/cameras.txt` and `images.txt` from each official `<scene>_dslr_undistorted.7z` into a shared calibration root. Run `uv run --with numpy python eth3d/qualify.py CALIBRATION_ROOT OUTPUT_ROOT` and `uv run --with numpy python eth3d/diagnose.py CALIBRATION_ROOT`. The results name the pinned source revision and the input file digests. [Source revision](results/source-revision.txt)
