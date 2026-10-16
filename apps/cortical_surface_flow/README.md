# Cortical Surface Flow

Deforms one of a subject's FreeSurfer surfaces onto another, conditioned on the T1:

| task | `--source_surface` | `--target_surface` | notes |
|---|---|---|---|
| white -> pial (validated) | `white` | `pial.T1` | default |
| coarse white -> white | `orig` | `white` | `orig` = topology-corrected tessellation |
| white -> inflated | `white` | `inflated` | `--select_on corr` |
| inflated -> sphere | `inflated` | `sphere` | `--select_on corr --no_t1` (not anatomical, leaves the T1 grid) |

Either hemisphere: `--hemi lh | rh`; one experiment = one hemisphere + one surface pair.

All FreeSurfer surfaces of a hemisphere (orig, smoothwm, white, pial, inflated, sphere)
share the vertex set that `sphere.reg` registers to fsaverage.  The cache resamples
every one of them onto the fsaverage 164k vertices with the same barycentric weights,
so vertex *i* is the same point of cortex in every cached surface and in the
prediction.  `orig.nofix` has its own vertex set and cannot be used.

## Model

Closed-loop estimator + controller (`--architecture closed_loop`, default):

```
estimator   (x, t, T1) -> mu, sigma_n^2, sigma_t^2, n_hat      posterior over the vertex end point
controller  dx/dt = (m - x) / (1 - t)                         m = mu, Kalman-fused along the path
```

- T1 features are sampled where the vertex currently is (closed loop); a learnable
  positional encoding is concatenated to the T1 at the encoder input.
- Trained on states off the ideal path: Gaussian perturbation (sigma = `perturb_mm` x t)
  and the model's own rollouts (`onpolicy_p`).  The perturbation must match the size of
  the deformation; `perturb_mm: null` sets it to `perturb_rel` x the training set's mean
  |X1 - X0| (0.18 x 2.77 mm = 0.5 mm for white -> pial) and writes the value back.
- beta-NLL with separate normal / tangential variance: how much to trust the
  along-surface vertex position is learned per vertex.
- `--architecture flow`: open-loop flow-matching baseline.

### T1 encoder (`model.json: __encoder__.type`)

| type | what it is | params | GPU peak / time per subject (fwd+bwd) |
|---|---|---|---|
| `residual_unet` (default) | residual encoder + decoder with skip connections; decoder levels are sampled | 9.3 M | 7.1 GiB / 0.35 s |
| `residual` | residual encoder only | 6.3 M | 4.0 GiB / 0.20 s |
| `plain` | stride-2 CNN pyramid (the prototype, `own_s2_t1`) | 4.0 M | 3.6 GiB / 0.18 s |

ResBlock: conv3-GN-SiLU-conv3-GN + identity (1x1 conv if the width changes), SiLU; the
second GN starts at zero gain, so each block starts as the identity.  The decoder
upsamples with transposed stride-2 convs (k3, p1), which place decoder voxel j exactly
on encoder voxel 2j -- the alignment sample_pyramid relies on (checked for odd and even
grid sizes).  `full_res_blocks: false` keeps the full-resolution level light (stem only;
decoder output = SiLU(GN(up(y1))) + stem features): with residual blocks at full
resolution too the encoder needs 12.2 GiB and 0.72 s per subject.

### Converged comparison (all 20 test subjects, Kalman q = 1 mm)

| model | surface | T→P p95 | corr | fold r4 / r16 | flipped |
|---|---|---|---|---|---|
| plain CNN, 30k (`lh_white_pial_own_s2_t1`) | **0.160** | **0.493** | 0.496 | 0.825 / 0.900 | 1.90 % |
| residual U-Net, 10k | 0.175 | 0.495 | 0.496 | 0.824 / 0.901 | **1.80 %** |
| residual U-Net, 30k | 0.165 | 0.528 | 0.496 | 0.825 / 0.900 | 1.97 % |

At an equal 10k-step budget the decoder is significantly better on correspondence and fold
detail (ablation table), but after 30k steps the plain CNN matches it and has a slightly
better tail (p95 better on 18/20 subjects).  The decoder speeds up convergence; it does not
raise the converged accuracy -- at twice the cost per step.  The 10k ablation therefore
measures effects at a fixed budget.

## Ablation and figure

```bat
:: one variant after another (never two GPU jobs), resumable; tables after every variant
python -m apps.cortical_surface_flow.ablation --spec apps/cortical_surface_flow/ablation_default.json
::   -> output/abl_lh_white_pial_ablation/ablation.{md,json,tex}

:: method figure (all panels are network outputs; runs on the CPU, ~5 min)
python -m apps.cortical_surface_flow.figures.flow_figure_data ^
    --config_dir output/abl_lh_white_pial_full_30k ^
    --checkpoint output/abl_lh_white_pial_full_30k/checkpoints/best.pth
python -m apps.cortical_surface_flow.figures.make_figure_v2
::   -> output/figures/v2/method.{pdf,svg,png} (+ method_caption.tex)
::   panels: 3D input, encoder-decoder with real feature maps, per-vertex trajectories,
::   outputs, velocity field at t = .1/.5/.9, disturbance rejection (open vs closed
::   loop), Bayesian setpoint fusion
```

## Usage

```bat
:: 1. cache (incremental: add hemispheres / surfaces later without re-registering)
python -m apps.cortical_surface_flow.preprocess --raw_root D:\NeuroPreprocess\output ^
    --cache_dir input/cortical_surface_flow/cache --hemis lh rh ^
    --surfaces orig white pial.T1 inflated sphere --workers 6

:: 2. train (+ test evaluation of best.pth at the end)
python -m apps.cortical_surface_flow.train --cache_dir input/cortical_surface_flow/cache ^
    --hemi lh --source_surface white --target_surface pial.T1 --experiment_name lh_white_pial_001

:: reproduce / resume from the saved configs
python -m apps.cortical_surface_flow.train --reproduce output/lh_white_pial_001/
python -m apps.cortical_surface_flow.train --reproduce output/lh_white_pial_001/ ^
    --resume output/lh_white_pial_001/checkpoints/latest.pth

:: 3. evaluate / predict
python -m apps.cortical_surface_flow.test --config_dir output/lh_white_pial_001 --split test
python -m apps.cortical_surface_flow.test --config_dir output/lh_white_pial_001 --subjects IXI002-Guys-0828

:: import a research-prototype run (cortical_flow/) as an experiment
python -m apps.cortical_surface_flow.import_prototype ^
    --checkpoint cortical_flow/runs/own_s2_t1/ckpt_best.pt --experiment_name lh_white_pial_own_s2_t1
```

## Persisted configuration

`output/<experiment_name>/`:

| file | contents |
|---|---|
| `data.json` | cache, hemisphere, source / target surface, X0 mode, the resolved split |
| `augment.json` | rotation / noise / gamma (no deformable transform) |
| `model.json` | architecture, use_t1, encoder (`__encoder__`: features, depth, positional encoding), head (`__head__`: MLP, Fourier) |
| `loss.json` | beta-NLL, normal weight, (flow) tangential weight |
| `control.json` | perturbation (resolved), on-policy rollouts, Euler steps, Kalman process noise |
| `optimizer.json`, `scheduler.json`, `training.json`, `infra.json` | optimisation, steps, selection metric, logging, resume |
| `checkpoints/best.pth`, `latest.pth` | weights + optimiser state + a snapshot of every config |
| `tensorboard/`, `training.log`, `test_eval/` | logs; test metrics, predictions, renders |

`<cache_dir>/preprocess.json` records how the cache was built (paths, N4, elastix,
hemispheres, surfaces).

## Dataset layouts (`preprocess.py --layout`)

| layout | raw data | images | registration |
|---|---|---|---|
| `neuropreprocess` (default) | `<root>/<subj>/{T1/*.nii.gz, freesurfer_brainmask/, surf/}` | N4, then resampled onto the MNI152 grid | elastix **affine** native T1 -> MNI152 brain; surfaces use the same matrix |
| `rigid_dataset` | `<root>/<subj>/{T1_Rigid/NIFTI/T1_Rigid.nii.gz, FreeSurfer_Brainmask_Rigid/NIFTI/..., T1/<subj>-T1.nii.gz, FreeSurfer_Surf/surf.tar.gz}` (patterns are config fields) | `T1_Rigid` and its mask **as they are** (already on the MNI152 grid; no N4, no resampling) | surfaces are native: a **rigid** registration native T1 -> `T1_Rigid` gives the matrix (it is the same image under a rigid transform), then the same surface chain as above |

```bat
python -m apps.cortical_surface_flow.preprocess --layout rigid_dataset --raw_root D:\DATASET-IXI ^
    --cache_dir D:\NeuroTrainer\input\cortical_surface_flow\cache_ixi_rigid ^
    --hemis lh rh --surfaces orig white pial.T1 --workers 6 --threads 2
python -m apps.cortical_surface_flow.cache_report --cache_dir <cache>     :: per-subject QC, 0 = nothing to check
```

**The rigid registration needs a fallback.**  Starting from the centre of mass converges for 186 of
203 subjects (correlation of the registered native T1 with `T1_Rigid` = 1.0000, in ~5 s).  For the
other 17 (all Guys-site) it ended at correlation ~ -0.1, with or without 27 rotated starts; a head
mirror was ruled out (no reflection of the image matches).  Those subjects start from the principal
axes of the native cortical surfaces against those of the rigid brain mask (4 sign choices, 4-level
pyramid): all 17 then reach 0.99998 - 1.00000.  The principal-axis start rotation can be ~90 deg
(axis permutation); the resulting transforms are small (2 - 29 deg, 7 - 61 mm).  `qc.json` records
how each subject was started (`rigid_start`).

QC that does not depend on the registration's own metric: the white surface sits on the strongest
T1 gradient, so the mean gradient magnitude at its vertices must peak at zero shift (`white_boundary`;
203 / 203 peak at zero, minimum centre / best = 0.9969); surfaces lie inside the mask (>= 0.99); pial
vertices lie outside white along the white normal for 92 - 94 % of vertices in both hemispheres
(catches flipped normals).

## Formal runs

`queue_formal_rigid.json` + `run_queue.py`: four experiments on the rigid cache, one after another
(`lh white -> pial`, `lh orig -> white`, `rh white -> pial`, `rh orig -> white`), the ablation's best
configuration without the positional encoding (closed loop, Bayesian posterior, Kalman setpoint,
perturbation + on-policy rollouts, residual encoder-decoder), 30k steps, the 163 / 20 / 20 split of all
earlier experiments (`--split_file`).

```bat
python -m apps.cortical_surface_flow.run_queue --spec apps/cortical_surface_flow/queue_formal_rigid.json
```

Resumes interrupted experiments, runs the test split after each, and keeps a TensorBoard on the
experiment that is currently training (first free port from 6006; a TensorBoard someone else runs is
never touched).  The queue's own folder: `output/formal_rigid_queue/`.

## Cache

```
<cache>/canonical_<hemi>.npz                 fsaverage faces + the MNI152 grid
<cache>/<subject>/T1_N4_mni.nii.gz           N4, affine (elastix) to MNI152, template header
<cache>/<subject>/Brainmask_mni.nii.gz
<cache>/<subject>/T1_input_mni.nii.gz        z-score inside the mask, -10 outside
<cache>/<subject>/<hemi>.<surface>.surf.gii  fsaverage 164k vertices, MNI mm
<cache>/<subject>/qc.json                    registration matrix + per-surface QC
```

A surface already in the cache is re-derived on every preprocessing run and must match
(< 1e-3 mm).  Preprocessing two subjects from scratch reproduced the prototype's
`cortical_flow/cache_ixi` exactly (registration matrix, T1 and surfaces: max |diff| 0).

## Test outputs

`test_eval/predictions/`: `<subject>_<hemi>.<target>.pred.surf.gii` (MNI),
`.pred_scanner.surf.gii` (the subject's scanner RAS, overlays the raw T1) and
`.sigma.func.gii` (predicted posterior sigma along the normal / along the surface, mm).
`metrics.json` / `summary.md`: exact point-to-triangle surface distance,
correspondence error, correspondence-free fold detail, mesh quality, for the model and
four predictors that ignore the T1 (source surface; source + mean normal offset, i.e.
population thickness for white -> pial; source + mean displacement; mean target).

## ONNX export

```bat
python -m apps.cortical_surface_flow.export_onnx --config_dir output/ixi_rigid_lh_white_pial ^
    --checkpoint output/ixi_rigid_lh_white_pial/checkpoints/lh-white2pial-t1-1.0.0.0-20261016.pth
::   -> output/<experiment>/onnx/<experiment>.{encoder,estimator}.onnx + .json
::      (--checkpoint defaults to checkpoints/best.pth; several --config_dir are exported one after another)
```

Two graphs (`--layout split`, default); the closed-loop integration stays with the caller:

| graph | in | out |
|---|---|---|
| `encoder` | `t1` (1, 1, 182, 218, 182): T1 z-scored inside the Brainmask, `z_outside` elsewhere | `f0` .. `f4`: the feature pyramid, run once per subject |
| `estimator` | `x_mm` (N, 3) vertices in MNI RAS mm, `t` (1,) path time, `f0` .. `f4` | `mu_mm` (N, 3) posterior mean, `var_normal_mm2`, `var_tangential_mm2` (N,), `n_hat` (N, 3) |

The caller runs `integrate_split` (export_onnx.py; C++: `CorticalSurfaceFlow.cpp`): at every one of `steps`
Euler steps estimate at (x, t), fuse with the Kalman setpoint (q in mm) and move x by dt (m - x) / (1 - t).
`euler_steps` and `kalman_q_mm` (from `control.json`) are only defaults, stored in the estimator's
`metadata_props`.  The features are shared, so vertices can go through the estimator in chunks: the C++
implementation needs 5.2 GB and 88 s per surface (10 physical cores), one estimator call per chunk of 32768
vertices and step.  The graphs are small (encoder 682 nodes / 35 MB, estimator 185 nodes / 1.4 MB).

`--layout unrolled` writes the earlier single graph `<experiment>.onnx` (t1, x0_mm -> pred_mm, sigmas) with
all steps unrolled (~22k nodes): the same result, but steps and q are fixed at export and onnxruntime needs ~13 GB.

mm <-> normalised coordinates are inside the graphs; N is dynamic, the grid is static.  The export
finishes by running the graphs in onnxruntime on the first test subject and comparing them with the PyTorch
model (`--no_verify` skips it): max |difference| about 1e-3 mm.  Opset >= 20 (GridSample on 5-D volumes),
onnxruntime >= 1.17.

`nn.GroupNorm` is replaced by an axis-by-axis mean / variance (`StableGroupNorm`) for the export only.
The exporter's own decomposition (InstanceNormalization) is exact in PyTorch but onnxruntime's float32
sum over the ~14 M voxels of a full-resolution group is off by ~1 % of the activation scale, which showed
up as ~3e-3 mm mean / 3e-2 mm max error in the predicted surface even after three steps.

## Validated result (lh white -> pial.T1, 163 / 20 / 20 IXI, 30 000 steps)

Test, 12 subjects, Kalman setpoint (q = 1 mm):

| | surface | corr | fold r4 / r16 | flipped | self-contact (target 0.252) |
|---|---|---|---|---|---|
| model | 0.15 mm | 0.47 mm | 0.835 / 0.905 | 0.019 | 0.253 |
| own white + mean thickness (no T1) | 0.45 mm | 1.24 mm | 0.658 / 0.741 | 0.135 | 0.300 |
| own white | 2.41 mm | 2.81 mm | 0.671 / 0.783 | 0.015 | 0.004 |

Inference needs the subject's FreeSurfer source surface and `sphere.reg`.
