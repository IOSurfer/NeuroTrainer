"""
Template-free surface flow: one FreeSurfer surface -> another, conditioned on the T1.

    T1 (+ learnable positional encoding)  ->  residual 3D U-Net  ->  feature pyramid
    (encoder type: plain CNN | residual encoder | residual encoder-decoder)
    vertex x, time t  ->  trilinear sampling of the pyramid at x  ->  f(x)
    [Fourier(x), Fourier(t), f(x)]  ->  MLP

closed_loop (ClosedLoopFlow)
    estimator   (x, t, T1) -> mu, var_n, var_t, n_hat: Gaussian posterior over the
                vertex end point, separate variance along the predicted end-point
                normal and per tangent direction
    controller  dx/dt = (m - x) / (1 - t), reaching the setpoint m exactly at t = 1;
                m = mu, or (kalman_q_mm) a Kalman fusion of the estimates made along
                the path, weighted by their predicted variances

flow (OpenLoopFlow)
    velocity    v(x, t, T1), trained on straight paths by flow matching

Coordinates are normalised, u = (mm - centre) / half_fov (GridFrame.mm2norm);
``norm2vox`` maps u to voxel indices.  Parameter names match the research prototype
(cortical_flow/), so its checkpoints load unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from apps.cortical_surface_flow.geometry import apply_affine


def _gn(c):
    return nn.GroupNorm(max(1, c // 8), c)


class T1Encoder(nn.Module):
    """Small 3D CNN feature pyramid.  Level l voxel j is centred on level-0 voxel
    2^l * j (3x3x3 convs, stride 2, padding 1), which sample_pyramid relies on."""

    def __init__(self, base=16, levels=5, checkpoint=False, in_ch=1):
        super().__init__()
        self.checkpoint = checkpoint
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base, 3, padding=1), _gn(base), nn.SiLU())
        blocks, c = [], base
        for _ in range(levels - 1):
            blocks.append(nn.Sequential(
                nn.Conv3d(c, 2 * c, 3, stride=2, padding=1), _gn(2 * c), nn.SiLU(),
                nn.Conv3d(2 * c, 2 * c, 3, padding=1), _gn(2 * c), nn.SiLU()))
            c *= 2
        self.blocks = nn.ModuleList(blocks)
        self.dims = [base * 2 ** i for i in range(levels)]
        self.feat_dim = sum(self.dims)

    def _run(self, block, x):
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, vol):                       # vol: (1, C, D, H, W)
        feats = [self._run(self.stem, vol)]
        for b in self.blocks:
            feats.append(self._run(b, feats[-1]))
        return feats


class ResBlock(nn.Module):
    """3x3x3 conv - GN - SiLU - 3x3x3 conv - GN, + skip (1x1 conv when the width
    changes), SiLU.  The second GN starts at zero gain, so every block starts as the
    identity (or the projection) and the network as its shallow counterpart."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1, self.n1 = nn.Conv3d(cin, cout, 3, padding=1), _gn(cout)
        self.conv2, self.n2 = nn.Conv3d(cout, cout, 3, padding=1), _gn(cout)
        nn.init.zeros_(self.n2.weight)
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.n1(self.conv1(x)))
        return self.act(self.n2(self.conv2(h)) + self.skip(x))


class ResidualEncoderDecoder(nn.Module):
    """Residual encoder (+ decoder) producing the same aligned feature pyramid as
    T1Encoder: level l has base * 2^l channels and voxel j of level l sits on level-0
    voxel 2^l j, so sample_pyramid is unchanged.

    encoder  stem -> [ResBlock x blocks] -> (stride-2 conv -> [ResBlock x blocks]) x (levels-1)
    decoder  (decoder=True) from the bottleneck up: transposed stride-2 conv (k3, p1,
             which centres input j on output 2j, i.e. exactly inverts the encoder's
             sampling) -> concat the encoder skip -> [ResBlock x blocks]; every
             decoder level is part of the returned pyramid (deepest = bottleneck).
    full_res_blocks=False keeps the full-resolution level light (most of the cost: 8x the
             voxels of level 1): no residual blocks there, and the decoder's level-0 output
             is SiLU(GN(up(y1))) + stem features instead of a concat + ResBlock."""

    def __init__(self, base=16, levels=5, checkpoint=False, in_ch=1, blocks=1, decoder=True,
                 full_res_blocks=False):
        super().__init__()
        self.checkpoint, self.decoder, self.full_res = checkpoint, decoder, full_res_blocks
        ch = [base * 2 ** i for i in range(levels)]
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base, 3, padding=1), _gn(base), nn.SiLU())
        self.down = nn.ModuleList([nn.Sequential(nn.Conv3d(ch[l - 1], ch[l], 3, stride=2, padding=1),
                                                 _gn(ch[l]), nn.SiLU()) for l in range(1, levels)])
        self.enc = nn.ModuleList([nn.Sequential(*[ResBlock(c, c) for _ in range(blocks)])
                                  if (l > 0 or full_res_blocks) else nn.Identity()
                                  for l, c in enumerate(ch)])
        if decoder:
            self.up = nn.ModuleList([nn.ConvTranspose3d(ch[l + 1], ch[l], 3, stride=2, padding=1)
                                     for l in range(levels - 1)])
            self.fuse = nn.ModuleList([nn.Sequential(ResBlock(2 * ch[l], ch[l]),
                                                     *[ResBlock(ch[l], ch[l]) for _ in range(blocks - 1)])
                                       if (l > 0 or full_res_blocks) else nn.Sequential(_gn(ch[l]), nn.SiLU())
                                       for l in range(levels - 1)])
        self.dims = ch
        self.feat_dim = sum(ch)

    def _run(self, block, x):
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, vol):                       # vol: (1, C, D, H, W)
        x = self._run(self.enc[0], self._run(self.stem, vol))
        skips = [x]
        for l, down in enumerate(self.down, start=1):
            x = self._run(self.enc[l], self._run(down, x))
            skips.append(x)
        if not self.decoder:
            return skips
        outs, y = [None] * len(skips), skips[-1]
        outs[-1] = y
        for l in range(len(skips) - 2, -1, -1):
            y = self.up[l](y, output_size=list(skips[l].shape[-3:]))
            if l == 0 and not self.full_res:
                y = self.fuse[0](y) + skips[0]
            else:
                y = self._run(self.fuse[l], torch.cat([y, skips[l]], 1))
            outs[l] = y
        return outs


ENCODERS = ("plain", "residual", "residual_unet")


def make_encoder(kind, base, levels, checkpoint, in_ch, blocks, full_res_blocks=False):
    if kind == "plain":
        return T1Encoder(base, levels, checkpoint, in_ch)
    if kind in ("residual", "residual_unet"):
        return ResidualEncoderDecoder(base, levels, checkpoint, in_ch, blocks, decoder=kind == "residual_unet",
                                      full_res_blocks=full_res_blocks)
    raise ValueError("encoder type must be one of %s, got %r" % (ENCODERS, kind))


def sample_pyramid(feats, x, norm2vox):
    """Trilinear sample of every level at normalised points x (P, 3).  Level l voxel j
    sits on level-0 index 2^l j: sampled with align_corners=True at j = idx / 2^l, exact
    for any grid size.  Points outside the grid take the border value."""
    idx = x @ norm2vox[:3, :3].T + norm2vox[:3, 3]
    out = []
    for level, f in enumerate(feats):
        n = torch.tensor(f.shape[2:], dtype=x.dtype, device=x.device)
        g = 2.0 * (idx / 2 ** level) / (n - 1).clamp(min=1) - 1.0
        s = F.grid_sample(f, g[..., [2, 1, 0]].view(1, 1, 1, -1, 3), mode="bilinear",
                          padding_mode="border", align_corners=True)
        out.append(s.view(f.shape[1], -1).t())
    return torch.cat(out, -1)


def check_grid_sample(frame, device="cpu", levels=5):
    """End-to-end check of the coordinate chain: a 3-channel volume holding its own
    (i, j, k) index, pyramid levels made by a centre-tap stride-2 conv (the encoder's
    voxel centres), sampled through the normalised frame, must return the index the
    mm -> vox affine predicts.  -> (max error in voxels, mm->norm->mm round trip mm)"""
    shape = np.array(frame.shape)
    s = 2 ** (levels - 1)
    hi = np.minimum(shape - 2, s * (np.ceil(shape / s) - 1))
    idx = np.random.default_rng(0).uniform(1.0, hi, size=(1024, 3))
    mm = apply_affine(frame.vox2mm, idx)
    xn = apply_affine(frame.mm2norm, mm)
    x = torch.tensor(xn, dtype=torch.float32, device=device)
    ijk = torch.stack(torch.meshgrid(*[torch.arange(int(k), dtype=torch.float32, device=device)
                                       for k in shape], indexing="ij"))[None]
    with torch.no_grad():
        conv = nn.Conv3d(3, 3, 3, stride=2, padding=1, groups=3, bias=False).to(device)
        conv.weight.zero_()
        conv.weight[:, :, 1, 1, 1] = 1.0
        feats = [ijk]
        for _ in range(levels - 1):
            feats.append(conv(feats[-1]))
        got = sample_pyramid(feats, x, torch.tensor(frame.norm2vox, dtype=torch.float32, device=device))
    want = torch.tensor(idx, dtype=torch.float32, device=device).repeat(1, levels)
    err = (got - want).abs().max().item()
    rt = float(np.abs(apply_affine(frame.norm2mm, xn) - mm).max())
    return err, rt


def fourier(x, n):
    """[x, sin(2^k pi x), cos(2^k pi x) ...]"""
    if n == 0:
        return x
    k = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * k
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def _mlp(d_in, hidden, layers):
    net = [nn.Linear(d_in, hidden), nn.SiLU()]
    for _ in range(layers - 1):
        net += [nn.Linear(hidden, hidden), nn.SiLU()]
    return nn.Sequential(*net)


class _Conditioned(nn.Module):
    """Shared T1 encoder / positional encoding / feature sampling."""

    def _init_encoder(self, shape, use_t1, base, levels, checkpoint, norm2vox, pe_ch, pe_stride,
                      encoder_type="plain", blocks=1, full_res_blocks=False):
        self.use_t1 = use_t1
        self.shape = tuple(int(s) for s in shape)
        self.pe_ch = pe_ch if use_t1 else 0
        if use_t1 and pe_ch > 0:
            pe_shape = [max(2, -(-s // pe_stride)) for s in self.shape]
            self.pos = nn.Parameter(torch.zeros(1, pe_ch, *pe_shape))
        else:
            self.pos = None
        self.encoder = (make_encoder(encoder_type, base, levels, checkpoint, 1 + self.pe_ch, blocks,
                                     full_res_blocks) if use_t1 else None)
        self.register_buffer("norm2vox", torch.as_tensor(norm2vox, dtype=torch.float32), persistent=False)
        return self.encoder.feat_dim if use_t1 else 0

    def encode(self, vol):
        """vol: (D, H, W) network input -> feature pyramid (None without T1)."""
        if not self.use_t1:
            return None
        v = vol.view(1, 1, *vol.shape[-3:])
        if self.pos is not None:
            pe = F.interpolate(self.pos, size=v.shape[-3:], mode="trilinear", align_corners=True)
            v = torch.cat([v, pe.to(v.dtype)], 1)
        return self.encoder(v)

    def _features(self, x, t, feats):
        h = [fourier(x, self.n_freq_x), fourier(t, self.n_freq_t)]
        if feats is not None:
            h.append(sample_pyramid(feats, x, self.norm2vox))
        return torch.cat(h, -1)


class ClosedLoopFlow(_Conditioned):
    """Estimator + controller.  See module docstring."""

    LV_MIN, LV_MAX = -4.6, 6.0          # sigma in [0.1, 20] mm

    def __init__(self, shape, half_fov, norm2vox, use_t1=True, base=16, levels=5, hidden=256,
                 layers=4, n_freq_x=6, n_freq_t=4, checkpoint=False, pe_ch=8, pe_stride=8,
                 encoder_type="plain", blocks=1, full_res_blocks=False):
        super().__init__()
        self.half_fov = float(half_fov)
        self.n_freq_x, self.n_freq_t = n_freq_x, n_freq_t
        fd = self._init_encoder(shape, use_t1, base, levels, checkpoint, norm2vox, pe_ch, pe_stride,
                                encoder_type, blocks, full_res_blocks)
        d_in = 3 * (1 + 2 * n_freq_x) + (1 + 2 * n_freq_t) + fd
        self.net = _mlp(d_in, hidden, layers)
        self.head = nn.Linear(hidden, 8)          # delta(3), lv_n, lv_t, n_hat(3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():                     # n_hat must not start at exactly zero
            self.head.weight[5:].normal_(0.0, 1e-3)
        self.kalman_q_mm = None
        self.last_sigma = None

    def estimate(self, x, t, feats):
        """x (P, 3) normalised, t (P, 1) -> mu (P, 3), lv_n (P,), lv_t (P,) [log mm^2], n_hat (P, 3)"""
        o = self.head(self.net(self._features(x, t, feats)))
        lv = o[:, 3:5].clamp(self.LV_MIN, self.LV_MAX)
        return x + o[:, :3], lv[:, 0], lv[:, 1], F.normalize(o[:, 5:], dim=-1, eps=1e-6)

    @torch.no_grad()
    def rollout(self, x0, t_end, feats, n_steps):
        """Closed loop from x0 to a per-point time t_end (P, 1): the states the
        controller actually visits, drift included (on-policy training states)."""
        x = x0.clone()
        dt = t_end / n_steps
        for k in range(n_steps):
            s = k * dt
            mu = self.estimate(x, s, feats)[0]
            x = x + dt * (mu - x) / (1.0 - s)
        return x

    @torch.no_grad()
    def integrate(self, x0, feats, n_steps=100, return_at=()):
        """Closed-loop Euler integration t = 0 -> 1; the last step lands on the
        setpoint.  Leaves the final per-vertex posterior sigma (mm) in last_sigma."""
        x = x0.clone()
        dt = 1.0 / n_steps
        hf2 = self.half_fov ** 2
        want = sorted(set(list(return_at) + [1.0]))
        snaps, j = {}, 0
        if want and want[0] <= 0.0:
            snaps[want[0]] = x.clone()
            j = 1
        m = pn = pt = None
        for i in range(n_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
            mu, lvn, lvt, nh = self.estimate(x, t, feats)
            rn, rt = lvn.exp() / hf2, lvt.exp() / hf2
            if self.kalman_q_mm is None or m is None:
                m, pn, pt = mu, rn, rt
            else:
                q = dt * (self.kalman_q_mm ** 2) / hf2
                pn, pt = pn + q, pt + q
                kn, kt = pn / (pn + rn), pt / (pt + rt)
                d = mu - m
                dn = (d * nh).sum(-1, keepdim=True) * nh
                m = m + kn[:, None] * dn + kt[:, None] * (d - dn)
                pn, pt = (1 - kn) * pn, (1 - kt) * pt
            x = x + dt * (m - x) / (1.0 - i * dt)
            while j < len(want) and want[j] <= (i + 1) * dt + 1e-9:
                snaps[want[j]] = x.clone()
                j += 1
        self.last_sigma = ((pn * hf2).sqrt(), (pt * hf2).sqrt())
        return x, snaps


class VelocityNet(nn.Module):
    def __init__(self, d_in, hidden=256, layers=4):
        super().__init__()
        self.net = _mlp(d_in, hidden, layers)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, h):
        return self.head(self.net(h))


class OpenLoopFlow(_Conditioned):
    """Flow-matching baseline: dX/dt = v(X, t, T1), Euler-integrated."""

    def __init__(self, shape, half_fov, norm2vox, use_t1=True, base=16, levels=5, hidden=256,
                 layers=4, n_freq_x=6, n_freq_t=4, checkpoint=False, pe_ch=8, pe_stride=8,
                 encoder_type="plain", blocks=1, full_res_blocks=False):
        super().__init__()
        self.half_fov = float(half_fov)
        self.n_freq_x, self.n_freq_t = n_freq_x, n_freq_t
        fd = self._init_encoder(shape, use_t1, base, levels, checkpoint, norm2vox, pe_ch, pe_stride,
                                encoder_type, blocks, full_res_blocks)
        self.velocity_net = VelocityNet(3 * (1 + 2 * n_freq_x) + (1 + 2 * n_freq_t) + fd, hidden, layers)
        self.last_sigma = None

    def velocity(self, x, t, feats):
        return self.velocity_net(self._features(x, t, feats))

    @torch.no_grad()
    def integrate(self, x0, feats, n_steps=100, return_at=()):
        x = x0.clone()
        dt = 1.0 / n_steps
        want = sorted(set(list(return_at) + [1.0]))
        snaps, j = {}, 0
        if want and want[0] <= 0.0:
            snaps[want[0]] = x.clone()
            j = 1
        for i in range(n_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(x, t, feats)
            while j < len(want) and want[j] <= (i + 1) * dt + 1e-9:
                snaps[want[j]] = x.clone()
                j += 1
        return x, snaps


def build_model(model_cfg, frame) -> nn.Module:
    """SurfaceFlowModelConfig + GridFrame -> model."""
    e, h = model_cfg.encoder, model_cfg.head
    kw = dict(use_t1=model_cfg.use_t1, base=e.base_features, levels=e.depth, hidden=h.hidden,
              layers=h.layers, n_freq_x=h.n_freq_x, n_freq_t=h.n_freq_t, checkpoint=e.checkpoint,
              pe_ch=e.pe_channels, pe_stride=e.pe_stride, encoder_type=e.type, blocks=e.blocks_per_level,
              full_res_blocks=e.full_res_blocks)
    cls = {"closed_loop": ClosedLoopFlow, "flow": OpenLoopFlow}.get(model_cfg.architecture)
    if cls is None:
        raise ValueError("unknown architecture %r (closed_loop | flow)" % model_cfg.architecture)
    return cls(frame.shape, frame.half_fov, frame.norm2vox, **kw)


def set_kalman(model, q_mm) -> None:
    """Kalman setpoint on (q_mm > 0) or off; no-op for the open-loop model."""
    if isinstance(model, ClosedLoopFlow):
        model.kalman_q_mm = float(q_mm) if q_mm and q_mm > 0 else None
