"""
All configuration classes for the Cortical Surface Flow application.

Each class covers exactly one concern.  Instances are registered with
ConfigManager and persist to ``<exp_dir>/<type_name>.json`` at the start of every
run; ``preprocess.py`` writes its own ``preprocess.json`` into the cache directory.

Task
----
Deform one of a subject's FreeSurfer surfaces (``DataConfig.source_surface``) onto
another (``DataConfig.target_surface``) of the same hemisphere, conditioned on the T1:

    orig -> white,  white -> pial.T1,  white -> inflated,  inflated -> sphere, ...

All FreeSurfer surfaces of a hemisphere (orig, smoothwm, white, pial, inflated,
sphere) share one vertex set, the one ``sphere.reg`` registers to fsaverage, so the
cache resamples every one of them onto the fsaverage 164k vertices with the same
barycentric weights: vertex i is the same point of cortex in every cached surface
and in the prediction.  One experiment = one hemisphere + one surface pair.

Defaults reproduce the validated experiment (lh white -> pial.T1, 163 / 20 / 20 IXI
subjects, test surface distance 0.15 mm with the Kalman setpoint).
"""

from typing import Dict

from configuration.base import AbstractConfig, ConfigField
from configuration.model.base import EncoderConfig, ModelConfig


class PreprocessConfig(AbstractConfig):
    """Raw data -> cache.  Saved as ``<cache_dir>/preprocess.json``."""

    config_type = "Preprocess"

    raw_root = ConfigField(
        r"D:\NeuroPreprocess\output",
        doc="NeuroPreprocess output: <subject>/{T1/*.nii.gz, freesurfer_brainmask/, surf/}. "
        "Subjects without FreeSurfer surfaces are skipped.",
    )
    cache_dir = ConfigField("./input/cortical_surface_flow/cache", doc="Where the cache is written")
    mni_template = ConfigField(
        r"D:\MNI152_T1_1mm_Brain.nii",
        doc="Registration target; its grid and header are the cache's image geometry",
    )
    fsaverage_dir = ConfigField(
        r"D:\CorticalFlowMatching",
        doc="Holds tpl-fsaverage_den-164k_hemi-{L,R}_{sphere,white,pial,inflated}.surf.gii",
    )
    elastix = ConfigField(r"C:\Users\OS\Downloads\elastix\elastix.exe", doc="elastix executable")
    elastix_params = ConfigField(
        "", doc="Affine parameter file; empty = the app's elastix_affine.txt"
    )
    hemis = ConfigField(("lh", "rh"), doc="Hemispheres to cache")
    surfaces = ConfigField(
        ("orig", "white", "pial.T1", "inflated", "sphere"),
        doc="FreeSurfer surfaces to cache per hemisphere; each must share sphere.reg's "
        "vertex set (orig.nofix does not and is rejected)",
    )
    n4_shrink = ConfigField(4, doc="N4: bias field estimated on a grid shrunk by this factor")
    n4_iterations = ConfigField((50, 50, 50, 50), doc="N4 iterations per level")
    z_outside = ConfigField(-10.0, doc="Network input value outside the Brainmask")
    workers = ConfigField(1, doc="Subjects processed in parallel (CPU / elastix only)")
    limit = ConfigField(None, doc="Only the first N subjects (testing)")


class DataConfig(AbstractConfig):
    config_type = "Data"

    cache_dir = ConfigField(
        "./input/cortical_surface_flow/cache", doc="Cache written by preprocess.py"
    )
    hemi = ConfigField("lh", doc="lh | rh -- one experiment per hemisphere")
    source_surface = ConfigField("white", doc="X0: surface the deformation starts from")
    target_surface = ConfigField("pial.T1", doc="X1: surface to predict")
    x0_mode = ConfigField(
        "subject",
        doc="subject: every subject starts from its own cached source_surface | "
        "template: everybody starts from the fsaverage template of that name",
    )
    train_subjects = ConfigField(
        None, doc="Explicit split; None = deterministic split (resolved and saved on first run)"
    )
    val_subjects = ConfigField(None)
    test_subjects = ConfigField(None)
    n_val = ConfigField(20, doc="Deterministic split: validation subjects")
    n_test = ConfigField(20, doc="Deterministic split: test subjects")
    split_seed = ConfigField(0, doc="Deterministic split seed")
    cache_device = ConfigField(
        "cpu",
        doc="Where subject volumes wait between steps: cpu (float16, pinned; scales to "
        "hundreds of subjects) | cuda",
    )


class AugmentConfig(AbstractConfig):
    """
    Training-only augmentation, applied on the GPU.  No deformable transform: the
    surfaces must stay exactly on the anatomy.

    Rotation is about the grid centre and moves the T1, the Brainmask, the target
    surface, its normals and (x0_mode subject) the source surface together.
    Order: rotate -> z-score inside the rotated Brainmask -> noise -> gamma.
    """

    config_type = "Augment"

    enabled = ConfigField(True, doc="Global switch; False disables all augmentation")

    rotation = ConfigField(True, doc="Enable random rotation")
    rotation_p = ConfigField(0.3, doc="Apply probability")
    rotation_degrees = ConfigField(15.0, doc="Max rotation per axis in degrees")

    noise = ConfigField(True, doc="Enable additive Gaussian noise (z-score units)")
    noise_p = ConfigField(0.3, doc="Apply probability")
    noise_std = ConfigField((0.0, 0.1), doc="Noise std range (min, max)")

    gamma = ConfigField(True, doc="Enable sign-preserving gamma |x|^g on z-scores")
    gamma_p = ConfigField(0.3, doc="Apply probability")
    gamma_log_gamma = ConfigField((-0.3, 0.3), doc="Log-gamma range (min, max)")


class SurfaceFlowEncoderConfig(EncoderConfig):
    """
    3D CNN feature pyramid over the full MNI grid.

    Inherits from :class:`~configuration.model.base.EncoderConfig`:
        in_channels   -- T1 channels (1); the positional encoding is added on top
        base_features -- channels at level 0, doubled per level
        depth         -- pyramid levels
    """

    config_type = "SurfaceFlowEncoder"

    type = ConfigField(
        "residual_unet",
        doc="plain: stride-2 CNN pyramid | residual: residual encoder | residual_unet: "
        "residual encoder-decoder with skip connections (decoder levels are sampled)",
    )
    blocks_per_level = ConfigField(1, doc="Residual blocks per encoder / decoder level")
    full_res_blocks = ConfigField(
        False,
        doc="Residual blocks and a concat decoder block also at full resolution "
        "(~4x the cost of the plain encoder; False: light level 0, decoder output added "
        "to the stem features)",
    )
    base_features = ConfigField(16, doc="Feature channels at the first encoding level")
    depth = ConfigField(5, doc="Pyramid levels (stride-2 convs between them)")
    pe_channels = ConfigField(
        8, doc="Learnable positional-encoding channels concatenated to the T1 (0 = off)"
    )
    pe_stride = ConfigField(8, doc="Positional-encoding grid = T1 grid / stride")
    checkpoint = ConfigField(
        False, doc="Recompute encoder activations in backward (less memory, ~30 percent slower)"
    )


class SurfaceFlowHeadConfig(AbstractConfig):
    """Per-vertex MLP on [Fourier(x), Fourier(t), sampled T1 features]."""

    config_type = "SurfaceFlowHead"

    hidden = ConfigField(256, doc="Hidden width")
    layers = ConfigField(4, doc="Hidden layers")
    n_freq_x = ConfigField(6, doc="Fourier frequencies of the position")
    n_freq_t = ConfigField(4, doc="Fourier frequencies of the time")


class SurfaceFlowModelConfig(ModelConfig):
    """
    architecture
        closed_loop  estimator (posterior over the vertex end point: mean, normal /
                     tangential variance, end-point normal) + controller
                     dx/dt = (m - x)/(1 - t); m optionally Kalman-filtered
        flow         open-loop velocity field v(x, t, T1) (flow-matching baseline)
    use_t1
        False for tasks the image cannot inform (e.g. inflated -> sphere, which also
        leave the T1 grid): no encoder, the MLP sees position and time only.
    """

    config_type = "Model"

    architecture = ConfigField("closed_loop", doc="closed_loop | flow")
    use_t1 = ConfigField(True, doc="Condition on the T1 (False: geometry only)")

    def __init__(self, file_path: str = "") -> None:
        super().__init__(file_path=file_path)
        self._encoder = SurfaceFlowEncoderConfig()
        self._head = SurfaceFlowHeadConfig()

    @property
    def encoder(self) -> SurfaceFlowEncoderConfig:
        return self._encoder

    @property
    def head(self) -> SurfaceFlowHeadConfig:
        return self._head

    def sub_configs(self) -> Dict[str, AbstractConfig]:
        return {"encoder": self._encoder, "head": self._head}


class LossConfig(AbstractConfig):
    config_type = "Loss"

    beta_nll = ConfigField(
        0.5,
        doc="closed_loop: beta-NLL exponent of the end-point Gaussian "
        "(0 = plain NLL, 1 = MSE-like mean gradients)",
    )
    normal_weight = ConfigField(
        0.1, doc="closed_loop: weight of (1 - n_hat . n) for the predicted end-point normal"
    )
    tangential_weight = ConfigField(
        0.1,
        doc="flow, and closed_loop with uncertainty=False: weight of the along-surface "
        "part of the error",
    )
    uncertainty = ConfigField(
        True,
        doc="closed_loop: learned Gaussian posterior (beta-NLL, enables the Kalman "
        "setpoint).  False = fixed-variance MSE on the end point (ablation)",
    )


class ControlConfig(AbstractConfig):
    """Closed-loop training states and inference integration."""

    config_type = "Control"

    perturb_mm = ConfigField(
        None,
        doc="Gaussian perturbation of path states, sigma = perturb_mm * t.  None = "
        "perturb_rel x the training set's mean |X1 - X0| (it must match the size of the "
        "deformation: 0.5 mm for white -> pial, 2 mm was right for template -> pial)",
    )
    perturb_rel = ConfigField(
        0.18, doc="perturb_mm = None: perturbation as a fraction of the mean displacement"
    )
    onpolicy_p = ConfigField(0.5, doc="Fraction of training points taken from the model's own rollout")
    onpolicy_warmup = ConfigField(1000, doc="Steps before on-policy states are used")
    rollout_steps = ConfigField(6, doc="Euler steps of the on-policy rollout")
    euler_steps = ConfigField(100, doc="Integration steps at test / inference")
    val_euler_steps = ConfigField(50, doc="Integration steps during validation")
    kalman_q_mm = ConfigField(
        1.0,
        doc="closed_loop: Kalman-filter the setpoint with this process noise "
        "(mm per unit time); 0 = plain closed loop",
    )


class OptimizerConfig(AbstractConfig):
    config_type = "Optimizer"

    type = ConfigField("adamw", doc="adam | adamw | sgd")
    lr = ConfigField(1e-3, doc="Peak learning rate")
    weight_decay = ConfigField(0.0)
    grad_clip = ConfigField(1.0, doc="Max gradient L2-norm (0 = disabled)")


class SchedulerConfig(AbstractConfig):
    config_type = "Scheduler"

    type = ConfigField("cosine", doc="cosine | none")
    warmup_steps = ConfigField(0, doc="Linear LR warm-up steps")
    min_lr_ratio = ConfigField(0.01, doc="Cosine floor as a fraction of the peak LR")


class TrainingConfig(AbstractConfig):
    config_type = "Training"

    steps = ConfigField(30000, doc="Optimiser steps")
    gradient_accumulation = ConfigField(
        2, doc="Subjects per optimiser step (one subject in memory at a time)"
    )
    points = ConfigField(16384, doc="Vertices sampled per subject per micro-batch")
    amp = ConfigField(True, doc="bf16 autocast for the encoder (CUDA only)")
    select_on = ConfigField(
        "fold",
        doc="Validation metric selecting best.pth: fold (mean correspondence-free fold "
        "r4 / r16, higher is better; anatomical targets) | surface | corr (mm, lower is "
        "better; use corr for inflated / sphere targets)",
    )
    eval_train_subjects = ConfigField(8, doc="Training subjects also scored at each validation")


class InfraConfig(AbstractConfig):
    config_type = "Infra"

    output_dir = ConfigField("./output")
    experiment_name = ConfigField("cortical_surface_flow")
    device = ConfigField("auto", doc="auto | cpu | cuda")
    seed = ConfigField(0)
    resume = ConfigField(None, doc="Checkpoint file path to resume from")
    resume_state = ConfigField(
        True,
        doc="When resuming, also restore optimizer / scheduler / step / RNG state.  "
        "False = load model weights only (fine-tuning)",
    )
    val_interval = ConfigField(2000, doc="Validate every N steps")
    save_interval = ConfigField(1000, doc="Write checkpoints/latest.pth every N steps")
    log_interval = ConfigField(100, doc="TensorBoard scalars every N steps")
    log_images = ConfigField(True)
    log_images_interval = ConfigField(2500)
    run_test = ConfigField(True, doc="Evaluate best.pth on the test split after training")
