#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Banding Simulator (scale-first) — with sensor-style noise,
perpendicular edge jitter, and GLOBAL CURVATURE (arc-like stripes)

新增功能：条纹整体弯曲（弧线）
- 让所有条纹彼此依然大致平行，但共享同一条弯曲形状 C(u)。
- 通过在条纹局部坐标 (u, v) 中，对 v 做 v_eff = v - C(u) 的一致形变实现。
- 支持两种弯曲模式：
    1) 'quad'  : 二次曲率，C(u) = sign * amp_px * (u_norm^2)   （u_norm ∈ [-1,1]）
    2) 'sin'   : 正弦曲率，C(u) = amp_px * sin( 2π * u / wavelength_px + phase )
- 可选 'curve_centered': True 时对 C(u) 做去均值，使整体上条带不发生全局平移，只体现弧度。

Flow:
  choose REF -> scale by --scale -> simulate banding on scaled REF
Outputs:
  out/reference     : scaled REF (GT)
  out/banding       : simulated LQ
  out/banding_mask  : grayscale mask (same size as LQ)
  out/index.json    : metadata
"""

import os, json, argparse, random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import cv2, yaml

# -------------------------- Robust IO --------------------------
_VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _pick_ext(ext: str) -> str:
    ext = (ext or "").lower()
    return ext if ext in _VALID_EXTS else ".png"


def imread_safe(path: str, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return None
    buf = np.fromfile(p, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_safe(path: str, img: np.ndarray) -> str:
    if img is None or img.size == 0:
        raise ValueError("imwrite_safe: empty image")
    if img.dtype != np.uint8:
        raise ValueError(f"imwrite_safe: need uint8, got {img.dtype}")
    if img.ndim not in (2, 3):
        raise ValueError(f"imwrite_safe: ndim must be 2/3, got {img.ndim}")
    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    ext = _pick_ext(Path(p).suffix)
    ok, enc = cv2.imencode(ext, np.ascontiguousarray(img))
    if not ok:
        raise RuntimeError(f"imwrite_safe: imencode failed ({ext})")
    enc.tofile(p)
    chk = imread_safe(p, cv2.IMREAD_UNCHANGED)
    if chk is None or chk.size == 0:
        raise RuntimeError(f"imwrite_safe: wrote but cannot read back: {p}")
    return p


# -------------------------- Utils --------------------------
def list_images_recursively(roots: List[Path]) -> List[Path]:
    files = []
    for root in roots:
        for p in root.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in _VALID_EXTS
                and not any(part.startswith(".") for part in p.relative_to(root).parts)
            ):
                files.append(p)
    files.sort()
    return files


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    H, W = img.shape[:2]
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (newW, newH), interpolation=interp)


def resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


# -------------------------- Random specs --------------------------
def rfloat(spec: Any, default: float = 0.0) -> float:
    if spec is None:
        return float(default)
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = float(spec[0]), float(spec[1])
        return float(np.random.uniform(lo, hi))
    raise ValueError(f"Unsupported float spec: {spec}")


def rint(spec: Any, default: int = 0) -> int:
    if spec is None:
        return int(default)
    if isinstance(spec, int):
        return spec
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = int(spec[0]), int(spec[1])
        return int(np.random.randint(lo, hi + 1))
    raise ValueError(f"Unsupported int spec: {spec}")


def rchoice(spec: Any, default: Any = None):
    if spec is None:
        return default
    if isinstance(spec, dict):
        choices = spec.get("choices", [])
        probs = spec.get("p", None)
        if not choices:
            return default
        return np.random.choice(choices, p=probs)
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return default
        return random.choice(list(spec))
    return spec


def get_range_tuple(spec: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if spec is None:
        return (float(default[0]), float(default[1]))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (float(spec[0]), float(spec[1]))
    if isinstance(spec, (int, float)):
        v = float(spec)
        return (v, v)
    raise ValueError(f"Unsupported range spec: {spec}")


# -------------------------- Coarse noise fields --------------------------
def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m, s = np.mean(x), (np.std(x) + 1e-8)
    return (x - m) / s


def coarse_field_lowres(h, w, cell=16, sparkle=0.0, seed=None):
    rng = np.random.default_rng(seed)
    hh = max(1, int(round(h / float(cell))))
    ww = max(1, int(round(w / float(cell))))
    low = rng.standard_normal((hh, ww)).astype(np.float32)
    coarse = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    if sparkle > 0:
        hi = rng.standard_normal((h, w)).astype(np.float32) * float(sparkle)
        coarse = coarse + hi
    return _normalize01(coarse)


def coarse_field_fft(h, w, beta=1.4, seed=None):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((h, w)).astype(np.float32)
    F = np.fft.rfft2(white)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    rad = np.sqrt(fx**2 + fy**2).astype(np.float32)
    rad[0, 0] = 1.0
    Hf = (rad ** (-float(beta))).astype(np.float32)
    F_colored = F * Hf
    field = np.fft.irfft2(F_colored, s=(h, w)).astype(np.float32)
    return _normalize01(field)


def coarse_field_blur(h, w, sigma=6.0, sparkle=0.0, seed=None):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((h, w)).astype(np.float32)
    k = int(max(3, 2 * int(round(3 * sigma)) + 1))
    low = cv2.GaussianBlur(base, (k, k), sigma)
    if sparkle > 0:
        hi = rng.standard_normal((h, w)).astype(np.float32) * float(sparkle)
        low = low + hi
    return _normalize01(low)


def _noise_to_uniform01(z: np.ndarray) -> np.ndarray:
    # approx normal->[0,1]
    return 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * (z**3))))


# -------------------------- Stripe occupancy & curvature --------------------------
def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_cutline_shift(
    mask: np.ndarray,
    num_cutlines: int = 0,
    direction: str = "horizontal",
    shift: int = 0,
) -> np.ndarray:
    if num_cutlines <= 0 or shift == 0:
        return mask
    H, W = mask.shape
    axis = H if direction == "horizontal" else W
    if axis <= 2:
        return mask
    cuts = np.linspace(1, axis - 1, num_cutlines, dtype=int).tolist()
    out = np.zeros_like(mask)
    prev = 0
    for i, pos in enumerate(cuts + [axis]):
        if direction == "horizontal":
            reg = mask[prev:pos, :]
            out[prev:pos, :] = np.roll(reg, shift * i, axis=1)
        else:
            reg = mask[:, prev:pos]
            out[:, prev:pos] = np.roll(reg, shift * i, axis=0)
        prev = pos
    return out


def _smooth_noise_1d(L, sigma, seed):
    r = np.random.default_rng(seed).standard_normal(L).astype(np.float32)
    if sigma > 0:
        k = int(max(3, 2 * int(round(3 * sigma)) + 1))
        r = cv2.GaussianBlur(r[None, :], (k, 1), sigma)[0]
    s = np.std(r) + 1e-8
    return (r / s).astype(np.float32)


def _curvature_offset(u: np.ndarray, p: Dict, u_min: float, u_max: float) -> np.ndarray:
    """
    统一弯曲偏移 C(u)。返回与 u 同形状的 2D 数组（单位：像素，沿 v 方向的偏移量）。
    - quad:  C(u) = sign * amp_px * (u_norm ** 2) ，u_norm∈[-1,1]
    - sin :  C(u) = amp_px * sin(2π * u / wavelength_px + phase)
    """
    if not bool(p.get("curve_enable", False)):
        return np.zeros_like(u, dtype=np.float32)

    mode = str(p.get("curve_mode", "quad")).lower()
    amp = float(p.get("curve_amp_px", 0.0))
    if abs(amp) < 1e-6:
        return np.zeros_like(u, dtype=np.float32)

    if mode == "sin":
        wl = float(p.get("curve_wavelength_px", 0.0))
        if wl <= 1e-6:
            wl = max(1.0, (u_max - u_min) * 2.0)  # 默认很缓的周期
        phase = np.deg2rad(float(p.get("curve_phase_deg", 0.0)))
        C = amp * np.sin(2.0 * np.pi * (u / wl) + phase).astype(np.float32)
    else:
        # quad（默认）：抛物线弯曲；支持符号
        sign = float(p.get("curve_sign", 1.0))
        u_mid = 0.5 * (u_min + u_max)
        u_span = max(1e-6, 0.5 * (u_max - u_min))
        u_norm = ((u - u_mid) / u_span).astype(np.float32)  # [-1,1]
        C = sign * amp * (u_norm**2)

    if bool(p.get("curve_centered", True)):
        C = C - float(np.mean(C))  # 去均值，使整体不发生平移

    return C.astype(np.float32)


def generate_stripe_occupancy(
    shape_hw: Tuple[int, int], p: Dict, return_debug: bool = False
):
    """
    返回 occ∈[0,1]：1=条内（含羽化 + “法线方向”边缘抖动 + 全局弯曲）
    - 全局弯曲：对 v 做一致的 v_eff = v - C(u)，所有条纹共享同一 C(u)。
    - 法线抖动：仅随 u 变化，使上下两条边在垂直方向上下浮动（像素）。
    """
    H, W = shape_hw
    theta = np.deg2rad(float(p.get("angle_deg", 0.0)))
    width = max(1.0, float(p.get("stripe_width", 10.0)))
    gap = max(1.0, float(p.get("stripe_gap", 20.0)))
    period = width + gap
    phase = float(p.get("phase_px", 0.0))

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xc = xs - (W - 1) / 2.0
    yc = ys - (H - 1) / 2.0
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    # 局部坐标：u 沿条纹方向；v 垂直于条纹方向
    u = xc * cos_t + yc * sin_t
    v = -xc * sin_t + yc * cos_t
    u_min, u_max = float(u.min()), float(u.max())

    # ---------- 全局弯曲偏移 C(u) ----------
    C = _curvature_offset(u, p, u_min, u_max).astype(np.float32)
    v_eff = v - C  # 在“弯曲坐标”中计算条带

    # 条序号（在弯曲坐标中取整）
    k_map = np.round((v_eff - phase) / period).astype(np.int32)
    k_min, k_max = int(k_map.min()), int(k_map.max())
    K = max(1, k_max - k_min + 1)
    idx = (k_map - k_min).astype(np.int32)

    rng = np.random.default_rng(p.get("seed", None))
    offset_per_k = rng.uniform(
        -float(p.get("spacing_jitter_px", 0.0)),
        float(p.get("spacing_jitter_px", 0.0)),
        size=K,
    ).astype(np.float32)
    width_jitter = rng.uniform(
        -float(p.get("width_base_jitter_px", 0.0)),
        float(p.get("width_base_jitter_px", 0.0)),
        size=K,
    ).astype(np.float32)

    # 沿 u 的慢变“宽度波动”
    wiggle_len = int(p.get("wiggle_table_len", 1024))
    wiggle_sigma = float(p.get("wiggle_smooth_sigma", 20.0))
    wiggle_amp = float(p.get("width_wiggle_px", 1.5))
    wiggle_1d = _smooth_noise_1d(max(8, wiggle_len), wiggle_sigma, p.get("seed", None))

    uu = np.clip(
        (u - u_min) / max(u_max - u_min, 1e-6) * (len(wiggle_1d) - 1),
        0,
        len(wiggle_1d) - 1,
    )
    i0 = np.floor(uu).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, len(wiggle_1d) - 1)
    t = (uu - i0).astype(np.float32)
    wiggle_map = ((1 - t) * wiggle_1d[i0] + t * wiggle_1d[i1]) * wiggle_amp

    center_v = (k_map.astype(np.float32) * period + phase + offset_per_k[idx]).astype(
        np.float32
    )
    w0_per_k = (width + width_jitter[idx]).astype(np.float32)
    width_here = np.maximum(1.0, w0_per_k + wiggle_map)

    # ---------- 法线方向上下浮动 ----------
    perp_max = float(p.get("edge_perp_max_px", 0.0))
    perp_sig = float(p.get("edge_perp_sigma", 12.0))
    indep_edges = bool(p.get("edge_perp_independent_edges", True))
    per_stripe = bool(p.get("edge_perp_per_stripe_jitter", True))

    top_1d = _smooth_noise_1d(
        max(8, wiggle_len),
        perp_sig,
        None if p.get("seed", None) is None else p["seed"] + 101,
    )
    bot_1d = _smooth_noise_1d(
        max(8, wiggle_len),
        perp_sig,
        None if p.get("seed", None) is None else p["seed"] + 202,
    )
    top_1d = top_1d / (np.max(np.abs(top_1d)) + 1e-8)
    bot_1d = bot_1d / (np.max(np.abs(bot_1d)) + 1e-8)

    top_u = ((1 - t) * top_1d[i0] + t * top_1d[i1]).astype(np.float32)
    bot_u = ((1 - t) * bot_1d[i0] + t * bot_1d[i1]).astype(np.float32)
    if not indep_edges:
        bot_u = top_u

    if per_stripe:
        g_top = rng.uniform(0.6, 1.0, size=K).astype(np.float32)
        g_bot = rng.uniform(0.6, 1.0, size=K).astype(np.float32)
        g_top_map = g_top[idx]
        g_bot_map = g_bot[idx]
    else:
        g_top_map = 1.0
        g_bot_map = 1.0

    # 注意：这里的 top/bot 是在“弯曲坐标 v_eff”下的边界
    top_edge_eff = center_v + 0.5 * width_here + perp_max * g_top_map * top_u
    bot_edge_eff = center_v - 0.5 * width_here + perp_max * g_bot_map * bot_u

    # 纠正偶发的“交叉”
    t_eff = np.maximum(top_edge_eff, bot_edge_eff)
    b_eff = np.minimum(top_edge_eff, bot_edge_eff)

    # 在弯曲坐标中计算 signed distance（inside 为负）
    d = -np.minimum(t_eff - v_eff, v_eff - b_eff).astype(np.float32)

    # 可选：额外 2D 边界噪声
    ej_amp = float(p.get("edge_jitter_px", 0.0))
    ej_sigma = float(p.get("edge_jitter_sigma", 0.0))
    if ej_amp > 0:
        mode = str(p.get("edge_jitter_mode", "blur")).lower()
        if mode == "fft":
            j = coarse_field_fft(
                H,
                W,
                beta=float(p.get("edge_jitter_beta", 1.4)),
                seed=p.get("seed", None),
            )
        elif mode == "lowres":
            j = coarse_field_lowres(
                H, W, cell=int(p.get("edge_jitter_cell", 16)), seed=p.get("seed", None)
            )
        else:
            j = coarse_field_blur(
                H, W, sigma=max(0.1, ej_sigma), seed=p.get("seed", None)
            )
        d = d - ej_amp * j

    feather = float(p.get("edge_feather_px", 1.4))
    occ = 1.0 - _smoothstep(-feather, +feather, d)  # 1=inside

    nh = int(p.get("cutlines_horizontal", 0))
    nv = int(p.get("cutlines_vertical", 0))
    shift = int(p.get("cutline_shift", 0))
    if nh > 0:
        occ = apply_cutline_shift(occ, nh, "horizontal", shift)
    if nv > 0:
        occ = apply_cutline_shift(occ, nv, "vertical", shift)

    if return_debug:
        return occ.astype(np.float32), {
            "top_eff": t_eff,
            "bot_eff": b_eff,
            "d": d,
            "C": C,
        }
    return occ.astype(np.float32)


# -------------------------- DARK application --------------------------
def _apply_dark_rgb(base_f: np.ndarray, occ: np.ndarray, p: Dict) -> np.ndarray:
    out = base_f.copy()
    chs = [c.upper() for c in p["dark_channels_rgb"]]
    if p["dark_scope"] == "unified":
        v = float(p["dark_value_unified"])
        m = 1.0 - occ * (1.0 - v)
        if "B" in chs:
            out[..., 0] *= m
        if "G" in chs:
            out[..., 1] *= m
        if "R" in chs:
            out[..., 2] *= m
    else:
        vb = float(p.get("dark_value_b", 0.8))
        vg = float(p.get("dark_value_g", 0.8))
        vr = float(p.get("dark_value_r", 0.8))
        if "B" in chs:
            out[..., 0] *= 1.0 - occ * (1.0 - vb)
        if "G" in chs:
            out[..., 1] *= 1.0 - occ * (1.0 - vg)
        if "R" in chs:
            out[..., 2] *= 1.0 - occ * (1.0 - vr)
    return clamp01(out)


def _apply_dark_ycbcr(base_f: np.ndarray, occ: np.ndarray, p: Dict) -> np.ndarray:
    ycc = cv2.cvtColor(base_f, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    chs = [c.capitalize() for c in p["dark_channels_ycbcr"]]
    if p["dark_scope"] == "unified":
        v = float(p["dark_value_unified"])
        m = 1.0 - occ * (1.0 - v)
        if "Y" in chs:
            Y = clamp01(Y * m)
        if "Cr" in chs:
            Cr = clamp01(Cr * m)
        if "Cb" in chs:
            Cb = clamp01(Cb * m)
    else:
        vy = float(p.get("dark_value_y", 0.8))
        vcb = float(p.get("dark_value_cb", 0.8))
        vcr = float(p.get("dark_value_cr", 0.8))
        if "Y" in chs:
            Y = clamp01(Y * (1.0 - occ * (1.0 - vy)))
        if "Cb" in chs:
            Cb = clamp01(Cb * (1.0 - occ * (1.0 - vcb)))
        if "Cr" in chs:
            Cr = clamp01(Cr * (1.0 - occ * (1.0 - vcr)))
    ycc[..., 0], ycc[..., 1], ycc[..., 2] = Y, Cr, Cb
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# -------------------------- BLACK noise images --------------------------
def _rand_field(h, w, mode="fft", beta=1.4, cell=16, blur_sigma=6.0, seed=None):
    if mode == "fft":
        return coarse_field_fft(h, w, beta=beta, seed=seed)
    if mode == "lowres":
        return coarse_field_lowres(h, w, cell=cell, seed=seed)
    return coarse_field_blur(h, w, sigma=blur_sigma, seed=seed)


def _noise_image_rgb(h, w, p: Dict) -> np.ndarray:
    mode = str(p.get("black_noise_mode", "fft")).lower()
    if mode == "sensor":
        return _noise_image_ycbcr(h, w, p)
    beta = float(p.get("black_noise_beta", 1.4))
    cell = int(p.get("black_noise_cell", 16))
    bsig = float(p.get("black_noise_blur_sigma", 6.0))
    seed = p.get("seed", None)

    def one(lo, hi, sd):
        z = _rand_field(h, w, mode, beta, cell, bsig, sd)
        u = _noise_to_uniform01(z)
        return (float(lo) + u * (float(hi) - float(lo))).astype(np.float32)

    blo, bhi = get_range_tuple(p.get("black_b_range", [0.02, 0.12]), (0.02, 0.12))
    glo, ghi = get_range_tuple(p.get("black_g_range", [0.02, 0.12]), (0.02, 0.12))
    rlo, rhi = get_range_tuple(p.get("black_r_range", [0.02, 0.12]), (0.02, 0.12))
    B = one(blo, bhi, seed)
    G = one(glo, ghi, None if seed is None else seed + 1)
    R = one(rlo, rhi, None if seed is None else seed + 2)
    sigma = float(p.get("black_smooth_sigma", 0.0))
    if sigma > 0:
        k = int(max(3, 2 * int(round(3 * sigma)) + 1))
        B = cv2.GaussianBlur(B, (k, k), sigma)
        G = cv2.GaussianBlur(G, (k, k), sigma)
        R = cv2.GaussianBlur(R, (k, k), sigma)
    return np.stack([B, G, R], axis=-1)


def _noise_image_ycbcr(h, w, p: Dict) -> np.ndarray:
    mode = str(p.get("black_noise_mode", "fft")).lower()
    seed = p.get("seed", None)
    if mode == "sensor":
        # 传感器风格：暗部粗块 + 细颗粒 + 轻微彩斑/色偏
        ylo, yhi = get_range_tuple(p.get("sensor_y_range", [0.03, 0.10]), (0.03, 0.10))
        y_beta = float(p.get("sensor_y_beta", 1.8))
        y_cell = int(p.get("sensor_y_cell", 24))
        y_mode = str(p.get("sensor_y_mode", "fft")).lower()
        y_blur = float(p.get("sensor_y_blur_sigma", 0.0))
        y_grain = float(p.get("sensor_y_grain", 0.010))
        if y_mode == "lowres":
            y_coarse = coarse_field_lowres(h, w, cell=y_cell, seed=seed)
        elif y_mode == "blur":
            y_coarse = coarse_field_blur(
                h,
                w,
                sigma=max(0.1, float(p.get("sensor_y_coarse_sigma", 6.0))),
                seed=seed,
            )
        else:
            y_coarse = coarse_field_fft(h, w, beta=y_beta, seed=seed)
        y_u = _noise_to_uniform01(y_coarse)
        Y = (ylo + y_u * (yhi - ylo)).astype(np.float32)
        if y_grain > 0:
            Y += np.random.standard_normal((h, w)).astype(np.float32) * y_grain
        if y_blur > 0:
            k = int(max(3, 2 * int(round(3 * y_blur)) + 1))
            Y = cv2.GaussianBlur(Y, (k, k), y_blur)

        cb_center = float(p.get("sensor_cb_center", 0.5))
        cr_center = float(p.get("sensor_cr_center", 0.5))
        cb_amp = rfloat(p.get("sensor_cb_amp_range", [0.02, 0.06]), 0.04)
        cr_amp = rfloat(p.get("sensor_cr_amp_range", [0.02, 0.06]), 0.04)
        cb_beta = float(p.get("sensor_cb_beta", 1.2))
        cr_beta = float(p.get("sensor_cr_beta", 1.2))
        cb_grain = float(p.get("sensor_cb_grain", 0.010))
        cr_grain = float(p.get("sensor_cr_grain", 0.010))
        cb_bias = rfloat(p.get("sensor_cb_bias_range", [-0.01, 0.01]), 0.0)
        cr_bias = rfloat(p.get("sensor_cr_bias_range", [-0.01, 0.01]), 0.0)
        chroma_blur = float(p.get("sensor_chroma_blur_sigma", 0.0))
        Cb_coarse = coarse_field_fft(
            h, w, beta=cb_beta, seed=None if seed is None else seed + 1
        )
        Cr_coarse = coarse_field_fft(
            h, w, beta=cr_beta, seed=None if seed is None else seed + 2
        )
        Cb = (
            cb_center
            + cb_amp * Cb_coarse
            + cb_grain * np.random.standard_normal((h, w)).astype(np.float32)
            + cb_bias
        ).astype(np.float32)
        Cr = (
            cr_center
            + cr_amp * Cr_coarse
            + cr_grain * np.random.standard_normal((h, w)).astype(np.float32)
            + cr_bias
        ).astype(np.float32)
        if chroma_blur > 0:
            k = int(max(3, 2 * int(round(3 * chroma_blur)) + 1))
            Cb = cv2.GaussianBlur(Cb, (k, k), chroma_blur)
            Cr = cv2.GaussianBlur(Cr, (k, k), chroma_blur)
        Y = np.clip(Y, 0.0, 1.0)
        Cb = np.clip(Cb, 0.0, 1.0)
        Cr = np.clip(Cr, 0.0, 1.0)
        ycc = np.stack([Y, Cr, Cb], axis=-1)  # OpenCV: Y,Cr,Cb
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    # 旧的 fft/lowres/blur
    beta = float(p.get("black_noise_beta", 1.4))
    cell = int(p.get("black_noise_cell", 16))
    bsig = float(p.get("black_noise_blur_sigma", 6.0))

    def one(lo, hi, sd):
        z = _rand_field(h, w, mode, beta, cell, bsig, sd)
        u = _noise_to_uniform01(z)
        return (float(lo) + u * (float(hi) - float(lo))).astype(np.float32)

    ylo, yhi = get_range_tuple(p.get("black_y_range", [0.02, 0.10]), (0.02, 0.10))
    cblo, cbhi = get_range_tuple(p.get("black_cb_range", [0.49, 0.51]), (0.49, 0.51))
    crlo, crhi = get_range_tuple(p.get("black_cr_range", [0.49, 0.51]), (0.49, 0.51))
    Y = one(ylo, yhi, seed)
    Cb = one(cblo, cbhi, None if seed is None else seed + 1)
    Cr = one(crlo, crhi, None if seed is None else seed + 2)
    sigma = float(p.get("black_smooth_sigma", 0.0))
    if sigma > 0:
        k = int(max(3, 2 * int(round(3 * sigma)) + 1))
        Y = cv2.GaussianBlur(Y, (k, k), sigma)
        Cb = cv2.GaussianBlur(Cb, (k, k), sigma)
        Cr = cv2.GaussianBlur(Cr, (k, k), sigma)
    ycc = np.stack([Y, Cr, Cb], axis=-1)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# -------------------------- Stripe-only noise in Y (optional) --------------------------
def _add_noise_on_luma(
    sim_bgr_f: np.ndarray, stripe_bool: np.ndarray, p: Dict
) -> np.ndarray:
    ycc = cv2.cvtColor(sim_bgr_f, cv2.COLOR_BGR2YCrCb)
    y = ycc[..., 0]
    mask = stripe_bool.astype(bool)
    ntype = str(p.get("noise_type", "gaussian")).lower()
    if ntype == "gaussian":
        sigma = float(p.get("sigma_scalar", 0.0))
        if sigma > 0:
            n = np.random.normal(0.0, sigma, size=y.shape).astype(np.float32)
            y[mask] = clamp01(y[mask] + n[mask])
    elif ntype == "poisson":
        strength = float(p.get("noise_strength_scalar", 0.0))
        if strength > 0:
            lam = np.maximum(y * 255.0 * max(strength, 1e-8), 1e-8)
            pr = np.random.poisson(lam).astype(np.float32) / (
                255.0 * max(strength, 1e-8)
            )
            y[mask] = clamp01(pr[mask])
    elif ntype == "speckle":
        alpha = float(p.get("speckle_alpha", 0.0))
        if alpha > 0:
            n = np.random.normal(0.0, 1.0, size=y.shape).astype(np.float32)
            y[mask] = clamp01(y[mask] * (1.0 + alpha * n[mask]))
    elif ntype in ("saltpepper", "s&p"):
        amount = float(p.get("sp_amount", 0.0))
        if amount > 0:
            sp2 = np.random.rand(*y.shape[:2]) < amount
            coin2 = np.random.rand(*y.shape[:2]) < 0.5
            sp2 = sp2 & mask
            y[sp2 & coin2] = 0.0
            y[sp2 & ~coin2] = 1.0
    ycc[..., 0] = y
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# -------------------------- Simulation core (runs on scaled REF) --------------------------
def simulate_banding_on_ref(
    ref_bgr_scaled: np.ndarray, p: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = ref_bgr_scaled.shape[:2]
    base = clamp01(
        ref_bgr_scaled.astype(np.float32)
        / 255.0
        * float(p.get("global_brightness", 1.0))
    )
    occ = generate_stripe_occupancy((H, W), p)  # 1=inside stripe

    if str(p.get("stripe_mode", "dark")).lower() == "black":
        noise_img = (
            _noise_image_rgb(H, W, p)
            if str(p.get("black_space", "ycbcr")).lower() == "rgb"
            else _noise_image_ycbcr(H, W, p)
        )
        w = occ[..., None]
        sim_f = (1.0 - w) * base + w * noise_img
        export_v = float(p.get("export_black_mask_value", 0.25))
        banding_mask = 1.0 - occ * (1.0 - export_v)
    else:
        if str(p.get("dark_apply_space", "ycbcr")).lower() == "rgb":
            sim_f = _apply_dark_rgb(base, occ, p)
        else:
            sim_f = _apply_dark_ycbcr(base, occ, p)
        export_v = float(
            p.get("export_dark_mask_value", p.get("dark_value_unified", 0.8))
        )
        banding_mask = 1.0 - occ * (1.0 - export_v)

    if p.get("add_noise", False):
        sim_f = _add_noise_on_luma(sim_f, occ > 1e-6, p)

    post_blur_sigma = float(p.get("post_blur_sigma", 0.0))
    if post_blur_sigma > 0:
        k = int(max(3, 2 * int(round(3 * post_blur_sigma)) + 1))
        sim_f = cv2.GaussianBlur(sim_f, (k, k), post_blur_sigma)

    sim_bgr = (clamp01(sim_f) * 255.0 + 0.5).astype(np.uint8)
    return sim_bgr, banding_mask.astype(np.float32)


# -------------------------- Param sampler --------------------------
def sample_runtime_params(cfg: Dict, seed: Optional[int]) -> Dict:
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    p: Dict[str, Any] = {}

    # Geometry + slow irregularities
    p["angle_deg"] = rfloat(cfg.get("angle_range", 0.0), 0.0)
    p["stripe_width"] = rfloat(cfg.get("stripe_width_range", 12.0), 12.0)
    p["stripe_gap"] = rfloat(cfg.get("stripe_gap_range", 28.0), 28.0)
    p["phase_px"] = rfloat(cfg.get("phase_range", 0.0), 0.0)
    p["spacing_jitter_px"] = rfloat(cfg.get("spacing_jitter_range", [0.5, 2.5]), 1.5)
    p["width_base_jitter_px"] = rfloat(
        cfg.get("width_base_jitter_range", [0.5, 2.0]), 1.0
    )
    p["width_wiggle_px"] = rfloat(cfg.get("width_wiggle_px_range", [0.8, 2.5]), 1.8)
    p["wiggle_smooth_sigma"] = rfloat(
        cfg.get("wiggle_smooth_sigma_range", [16.0, 28.0]), 22.0
    )
    p["wiggle_table_len"] = int(cfg.get("wiggle_table_len", 1024))
    p["edge_feather_px"] = rfloat(cfg.get("edge_feather_px_range", [1.0, 2.2]), 1.4)

    # NEW: perpendicular edge jitter (normal to centerline)
    p["edge_perp_max_px"] = rfloat(cfg.get("edge_perp_max_px_range", [0.0, 0.0]), 0.0)
    p["edge_perp_sigma"] = rfloat(cfg.get("edge_perp_sigma_range", [8.0, 16.0]), 12.0)
    p["edge_perp_independent_edges"] = bool(
        cfg.get("edge_perp_independent_edges", True)
    )
    p["edge_perp_per_stripe_jitter"] = bool(
        cfg.get("edge_perp_per_stripe_jitter", True)
    )

    # (optional) legacy 2D jitter
    p["edge_jitter_px"] = rfloat(cfg.get("edge_jitter_px_range", [0.0, 0.0]), 0.0)
    p["edge_jitter_sigma"] = rfloat(cfg.get("edge_jitter_sigma_range", [0.0, 0.0]), 0.0)
    p["edge_jitter_mode"] = str(cfg.get("edge_jitter_mode", "blur")).lower()
    p["edge_jitter_cell"] = int(cfg.get("edge_jitter_cell", 16))
    p["edge_jitter_beta"] = rfloat(cfg.get("edge_jitter_beta", 1.4), 1.4)

    # NEW: Global curvature params
    p["curve_enable"] = bool(cfg.get("curve_enable", False))
    p["curve_mode"] = str(cfg.get("curve_mode", "quad")).lower()  # 'quad' | 'sin'
    p["curve_amp_px"] = rfloat(
        cfg.get("curve_amp_px_range", [0.0, 0.0]), 0.0
    )  # 弯曲幅度（像素）
    p["curve_centered"] = bool(cfg.get("curve_centered", True))  # 是否去均值
    if p["curve_mode"] == "sin":
        p["curve_wavelength_px"] = rfloat(
            cfg.get("curve_wavelength_px_range", [800.0, 1600.0]), 1200.0
        )
        p["curve_phase_deg"] = rfloat(
            cfg.get("curve_phase_deg_range", [0.0, 360.0]), 0.0
        )
        # sign 无意义
        p["curve_sign"] = 1.0
    else:
        p["curve_sign"] = rchoice(cfg.get("curve_sign_choices", [-1, 1]), 1)

    # Cutline shifts
    p["cutlines_horizontal"] = rint(cfg.get("cutlines_horizontal_range", [0, 2]), 0)
    p["cutlines_vertical"] = rint(cfg.get("cutlines_vertical_range", [0, 2]), 0)
    p["cutline_shift"] = rint(cfg.get("cutline_shift_range", [6, 20]), 12)

    # Global brightness
    p["global_brightness"] = rfloat(cfg.get("global_brightness_range", 1.0), 1.0)

    # Mode & dark/black params（保持原有）
    mode = str(cfg.get("stripe_mode", "dark")).lower()
    p["stripe_mode"] = mode
    if mode == "black":
        p["black_space"] = str(cfg.get("black_space", "ycbcr")).lower()
        p["black_noise_mode"] = str(cfg.get("black_noise_mode", "fft")).lower()
        p["black_noise_beta"] = rfloat(cfg.get("black_noise_beta", 1.4), 1.4)
        p["black_noise_cell"] = rint(cfg.get("black_noise_cell", 16), 16)
        p["black_noise_blur_sigma"] = rfloat(
            cfg.get("black_noise_blur_sigma", 6.0), 6.0
        )
        p["black_smooth_sigma"] = rfloat(cfg.get("black_smooth_sigma_range", 0.0), 0.0)
        p["black_b_range"] = cfg.get("black_b_range", [0.02, 0.12])
        p["black_g_range"] = cfg.get("black_g_range", [0.02, 0.12])
        p["black_r_range"] = cfg.get("black_r_range", [0.02, 0.12])
        p["black_y_range"] = cfg.get("black_y_range", [0.02, 0.10])
        p["black_cb_range"] = cfg.get("black_cb_range", [0.49, 0.51])
        p["black_cr_range"] = cfg.get("black_cr_range", [0.49, 0.51])
        # sensor-specific
        p["sensor_y_range"] = cfg.get("sensor_y_range", [0.03, 0.10])
        p["sensor_y_beta"] = rfloat(cfg.get("sensor_y_beta", 1.8), 1.8)
        p["sensor_y_cell"] = rint(cfg.get("sensor_y_cell", 24), 24)
        p["sensor_y_mode"] = str(cfg.get("sensor_y_mode", "fft")).lower()
        p["sensor_y_blur_sigma"] = rfloat(cfg.get("sensor_y_blur_sigma", 0.0), 0.0)
        p["sensor_y_grain"] = rfloat(cfg.get("sensor_y_grain", 0.010), 0.010)
        p["sensor_cb_center"] = rfloat(cfg.get("sensor_cb_center", 0.5), 0.5)
        p["sensor_cr_center"] = rfloat(cfg.get("sensor_cr_center", 0.5), 0.5)
        p["sensor_cb_amp_range"] = cfg.get("sensor_cb_amp_range", [0.02, 0.06])
        p["sensor_cr_amp_range"] = cfg.get("sensor_cr_amp_range", [0.02, 0.06])
        p["sensor_cb_beta"] = rfloat(cfg.get("sensor_cb_beta", 1.2), 1.2)
        p["sensor_cr_beta"] = rfloat(cfg.get("sensor_cr_beta", 1.2), 1.2)
        p["sensor_cb_grain"] = rfloat(cfg.get("sensor_cb_grain", 0.010), 0.010)
        p["sensor_cr_grain"] = rfloat(cfg.get("sensor_cr_grain", 0.010), 0.010)
        p["sensor_cb_bias_range"] = cfg.get("sensor_cb_bias_range", [-0.01, 0.01])
        p["sensor_cr_bias_range"] = cfg.get("sensor_cr_bias_range", [-0.01, 0.01])
        p["sensor_chroma_blur_sigma"] = rfloat(
            cfg.get("sensor_chroma_blur_sigma", 0.0), 0.0
        )
        p["export_black_mask_value"] = rfloat(
            cfg.get("export_black_mask_value", 0.25), 0.25
        )
    else:
        space = str(cfg.get("dark_apply_space", "ycbcr")).lower()
        p["dark_apply_space"] = space
        scope = str(cfg.get("dark_scope", "unified")).lower()
        p["dark_scope"] = scope
        if space == "rgb":
            chs = cfg.get("dark_channels", ["R", "G", "B"])
            p["dark_channels_rgb"] = [c.upper() for c in chs]
            if scope == "unified":
                p["dark_value_unified"] = rfloat(
                    cfg.get("dark_value_range", [0.70, 0.90]), 0.80
                )
            else:
                p["dark_value_b"] = rfloat(
                    cfg.get("dark_value_b_range", [0.70, 0.90]), 0.80
                )
                p["dark_value_g"] = rfloat(
                    cfg.get("dark_value_g_range", [0.70, 0.90]), 0.80
                )
                p["dark_value_r"] = rfloat(
                    cfg.get("dark_value_r_range", [0.70, 0.90]), 0.80
                )
        else:
            chs = cfg.get("dark_channels", ["Y"])
            p["dark_channels_ycbcr"] = [c.capitalize() for c in chs]
            if scope == "unified":
                p["dark_value_unified"] = rfloat(
                    cfg.get("dark_value_range", [0.70, 0.90]), 0.80
                )
            else:
                p["dark_value_y"] = rfloat(
                    cfg.get("dark_value_y_range", [0.70, 0.90]), 0.80
                )
                p["dark_value_cb"] = rfloat(
                    cfg.get("dark_value_cb_range", [0.80, 0.98]), 0.90
                )
                p["dark_value_cr"] = rfloat(
                    cfg.get("dark_value_cr_range", [0.80, 0.98]), 0.90
                )
        p["export_dark_mask_value"] = rfloat(
            cfg.get("export_dark_mask_value", 0.8), 0.8
        )

    # stripe-only extra noise & blur
    p["add_noise"] = bool(cfg.get("add_noise", False))
    p["noise_type"] = rchoice(cfg.get("noise_type", "gaussian"), "gaussian")
    p["sigma_scalar"] = rfloat(cfg.get("sigma_scalar_range", [0.004, 0.010]), 0.006)
    p["noise_strength_scalar"] = rfloat(
        cfg.get("noise_strength_scalar_range", [0.03, 0.08]), 0.05
    )
    p["speckle_alpha"] = rfloat(cfg.get("speckle_alpha_range", [0.04, 0.10]), 0.06)
    p["sp_amount"] = rfloat(cfg.get("sp_amount_range", [0.005, 0.015]), 0.01)
    p["post_blur_sigma"] = rfloat(cfg.get("post_blur_sigma_range", [0.0, 0.6]), 0.0)

    p["seed"] = seed
    return p


# -------------------------- Ref mapping & CLI --------------------------
def build_ref_index(
    ref_roots: Optional[List[Path]],
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    by_name, by_stem = {}, {}
    if not ref_roots:
        return by_name, by_stem
    for p in list_images_recursively(ref_roots):
        by_name.setdefault(p.name, p)
        by_stem.setdefault(p.stem, []).append(p)
    return by_name, by_stem


def find_corresponding_ref(
    gt_path: Path, by_name: Dict[str, Path], by_stem: Dict[str, List[Path]]
) -> Optional[Path]:
    if not by_name and not by_stem:
        return None
    if gt_path.name in by_name:
        return by_name[gt_path.name]
    lst = by_stem.get(gt_path.stem, [])
    return lst[0] if lst else None


def parse_args():
    ap = argparse.ArgumentParser(
        description="Banding generator (scale-first; dark/black; sensor noise; perpendicular-edge jitter; global curvature)"
    )
    ap.add_argument("--gt_dirs", nargs="+", required=True)
    ap.add_argument("--ref_dirs", nargs="+", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json_name", default="index.json")
    return ap.parse_args()


# -------------------------- Main --------------------------
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    gt_roots = [Path(d) for d in args.gt_dirs]
    for r in gt_roots:
        if not r.exists():
            raise FileNotFoundError(f"GT folder not found: {r}")
    ref_roots = [Path(d) for d in args.ref_dirs] if args.ref_dirs else None
    if ref_roots:
        for r in ref_roots:
            if not r.exists():
                raise FileNotFoundError(f"REF folder not found: {r}")

    gt_list = list_images_recursively(gt_roots)
    if not gt_list:
        raise FileNotFoundError(f"No images found under gt_dirs: {args.gt_dirs}")

    ref_by_name, ref_by_stem = build_ref_index(ref_roots)

    out_dir = Path(args.out_dir)
    ref_dir, deg_dir, bmsk_dir = (
        out_dir / "reference",
        out_dir / "banding",
        out_dir / "banding_mask",
    )
    for d in (ref_dir, deg_dir, bmsk_dir):
        d.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / args.json_name
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            if not isinstance(index, dict):
                index = {}
        except Exception:
            index = {}
    else:
        index = {}

    N = int(args.num)
    scale = float(args.scale)
    if scale <= 0:
        raise ValueError("--scale must be > 0")

    for i in range(1, N + 1):
        gt_path = random.choice(gt_list)
        gt_img = imread_safe(str(gt_path), cv2.IMREAD_COLOR)
        if gt_img is None:
            print(f"[WARN] read GT fail: {gt_path}")
            continue

        ref_src_path = (
            find_corresponding_ref(gt_path, ref_by_name, ref_by_stem)
            if ref_roots
            else None
        )
        if ref_src_path is not None:
            ref_src = imread_safe(str(ref_src_path), cv2.IMREAD_COLOR)
            if ref_src is None:
                print(f"[WARN] read REF fail: {ref_src_path}, fallback to GT")
                ref_src_path = None
        if ref_src_path is None:
            ref_src = gt_img

        if ref_src.shape[:2] != gt_img.shape[:2]:
            ref_src = resize_to(ref_src, gt_img.shape[:2])

        # 先 scale，再模拟
        ref_scaled = resize_by_scale(ref_src, scale)
        p = sample_runtime_params(cfg, seed=(args.seed + i) if args.seed else None)
        sim, band_mask = simulate_banding_on_ref(ref_scaled, p)

        tag = f"{i:06d}_{gt_path.stem}"
        ref_p = str(ref_dir / f"{tag}.png")
        deg_p = str(deg_dir / f"{tag}.png")
        bmsk_p = str(bmsk_dir / f"{tag}.png")
        imwrite_safe(ref_p, ref_scaled)
        imwrite_safe(deg_p, sim)
        imwrite_safe(bmsk_p, (np.clip(band_mask, 0, 1) * 255).astype(np.uint8))

        index[tag] = {
            "reference": str(Path(ref_p).resolve()),
            "banding": str(Path(deg_p).resolve()),
            "banding_mask": str(Path(bmsk_p).resolve()),
            "params": p,
            "source_gt": str(Path(gt_path).resolve()),
            "source_ref": str(Path(ref_src_path).resolve()) if ref_src_path else None,
            "scale": scale,
            "size_ref_scaled": list(ref_scaled.shape[:2]),
        }
        if i % 20 == 0 or i == N:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        src_note = f"(ref={ref_src_path.name})" if ref_src_path else "(ref=gt)"
        print(
            f"[{i}/{N}] {gt_path.name} -> {tag}.png {src_note}  ({ref_scaled.shape[1]}x{ref_scaled.shape[0]})"
        )

    print("\nDone.")
    print(f"  reference:    {ref_dir}")
    print(f"  banding:      {deg_dir}")
    print(f"  banding_mask: {bmsk_dir}")
    print(f"  index json:   {json_path}")


if __name__ == "__main__":
    main()
