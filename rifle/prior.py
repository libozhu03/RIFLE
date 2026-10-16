"""One-channel banding prior and checkpoint-compatible FiLM conditioning."""

import cv2
import numpy as np
import torch
from torch import nn


class BandingPriorFiLM(nn.Module):
    """Direction-aware FiLM conditioning for stripe/banding patterns.

    Adds horizontal and vertical average pooling to capture directional
    structure that global average pooling misses.
    """

    def __init__(self, in_ch, feat_ch_list, hidden=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.dir_reduce = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Linear(in_ch * 3, hidden),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, int(f) * 2) for f in feat_ch_list]
        )

    def _pool_features(self, prior):
        g = self.gap(prior).flatten(1)
        h = self.h_pool(prior).mean(dim=-1).flatten(1)
        v = self.v_pool(prior).mean(dim=-2).flatten(1)
        return torch.cat([g, h, v], dim=1)

    def forward(self, prior, feats):
        prior = self.dir_reduce(prior)
        g = self._pool_features(prior)
        g = self.shared(g)
        out = []
        for feat, head in zip(feats, self.heads):
            gb = head(g)
            gamma, beta = gb.chunk(2, dim=1)
            gamma = gamma.view(-1, feat.shape[1], 1, 1)
            beta = beta.view(-1, feat.shape[1], 1, 1)
            out.append(feat * (1 + gamma) + beta)
        return out

    def apply(self, prior, feat, idx: int):
        prior = self.dir_reduce(prior)
        g = self._pool_features(prior)
        g = self.shared(g)
        gb = self.heads[idx](g)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(-1, feat.shape[1], 1, 1)
        beta = beta.view(-1, feat.shape[1], 1, 1)
        return feat * (1 + gamma) + beta


def _robust_normalize(arr, q_low=1.0, q_high=99.0):
    lo = float(np.percentile(arr, q_low))
    hi = float(np.percentile(arr, q_high))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_gray01(rgb_np):
    rgb = rgb_np.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
        np.float32
    )


def _local_repeat_map(rmid: np.ndarray) -> np.ndarray:
    x = _robust_normalize(np.abs(rmid), 1.0, 99.0)
    angles = list(range(0, 180, 15))
    lags = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    shifts = []
    seen = set()
    for lag in lags:
        for ang in angles:
            theta = np.deg2rad(float(ang))
            dx = int(round(np.cos(theta) * lag))
            dy = int(round(np.sin(theta) * lag))
            if dx == 0 and dy == 0:
                continue
            key = (dy, dx)
            if key not in seen:
                seen.add(key)
                shifts.append(key)
    maps = []
    for dy, dx in shifts:
        shifted = np.roll(x, shift=(dy, dx), axis=(0, 1))
        corr = x * shifted
        if dy > 0:
            corr[:dy, :] = 0
        elif dy < 0:
            corr[dy:, :] = 0
        if dx > 0:
            corr[:, :dx] = 0
        elif dx < 0:
            corr[:, dx:] = 0
        maps.append(cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 2.0))
    return _robust_normalize(np.max(np.stack(maps, axis=0), axis=0), 1.0, 99.0)


def extract_banding_prior(rgb_np):
    """
    GP banding prior used by RIFLE FiLM.
    Single-channel prior:
    ch0: fused GP banding cue G^gamma_g * P^gamma_p
    """
    gray = _to_gray01(rgb_np)
    yb = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = np.abs(cv2.Scharr(yb, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(yb, cv2.CV_32F, 0, 1))
    g = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    g01 = _robust_normalize(g, 1.0, 99.0)

    low1 = cv2.GaussianBlur(gray, (0, 0), 3.5)
    low2 = cv2.GaussianBlur(gray, (0, 0), 8.0)
    rmid = (0.65 * (gray - low1) + 0.35 * (gray - low2)).astype(np.float32)
    rep = _local_repeat_map(rmid)

    band = _robust_normalize((g01**0.95) * (rep**1.10), 1.0, 99.0)
    return band[None, ...].astype(np.float32)
