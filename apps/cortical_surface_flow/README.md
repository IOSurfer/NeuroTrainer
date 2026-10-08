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

## Validated result (lh white -> pial.T1, 163 / 20 / 20 IXI, 30 000 steps)

Test, 12 subjects, Kalman setpoint (q = 1 mm):

| | surface | corr | fold r4 / r16 | flipped | self-contact (target 0.252) |
|---|---|---|---|---|---|
| model | 0.15 mm | 0.47 mm | 0.835 / 0.905 | 0.019 | 0.253 |
| own white + mean thickness (no T1) | 0.45 mm | 1.24 mm | 0.658 / 0.741 | 0.135 | 0.300 |
| own white | 2.41 mm | 2.81 mm | 0.671 / 0.783 | 0.015 | 0.004 |

Inference needs the subject's FreeSurfer source surface and `sphere.reg`.
