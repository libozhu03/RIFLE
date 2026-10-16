"""Restore images with RIFLE; use screen masks when provided and tile on CUDA OOM."""

from rifle.model import RIFLEInference
from rifle.pretrained import resolve_pretrained_model_path
from rifle.prior import extract_banding_prior

INFERENCE_DEFAULTS = {
    "process_mode": "full",
    "mask_blend_radius": 0,
    "prior_fast_scale": 1.0,
    "use_latent_tiling": False,
    "use_vae_tiling": True,
    "vae_encoder_tiled_size": 512,
    "vae_decoder_tiled_size": 64,
    "enable_xformers_memory_efficient_attention": False,
    "resize_to": 0,
    "upscale": 1.0,
    "coarse_tile_fuse_sigma": 12.0,
    "coarse_tile_fuse_luma_strength": 1.0,
    "coarse_tile_fuse_chroma_strength": 0.15,
    "suffix": "",
    "edge_reflect_pad": 96,
    "pad_to_multiple": 8,
    "auto_tile_by_resolution": True,
}
import argparse
import copy
import gc
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RIFLE flicker-banding removal.")
    parser.set_defaults(**INFERENCE_DEFAULTS)
    parser.add_argument(
        "--input_path", type=str, required=True, help="Input image path or directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Optional screen masks; omit to restore the full image.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Threshold applied to grayscale mask images in [0,255].",
    )
    parser.add_argument(
        "--mask_bbox_pad",
        type=int,
        default=64,
        help="Extra pixels to pad around the detected mask bbox before restoration.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Checkpoint path: model_xxx.pkl",
    )
    parser.add_argument(
        "--pretrained_model_path", type=str, default="preset/models/sd21"
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument(
        "--coarse_fallback_max_dim",
        type=int,
        default=1024,
        help="Long-side size used by the coarse global pass before fusing tile detail on fallback.",
    )
    args = parser.parse_args(argv)
    args.process_mode = "mask_image_segmented" if args.mask_dir else "full"
    return args


def _load_ckpt_cfg(pretrained_path: str):
    try:
        sd = torch.load(pretrained_path, map_location="cpu")
    except Exception as e:
        print(f"[Warn] failed to read checkpoint for config: {e}")
        return {}
    cfg = sd.get("train_cfg", {})
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _list_inputs(input_path: str):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = [
        x
        for x in sorted(p.rglob("*"))
        if x.is_file()
        and x.suffix.lower() in IMAGE_EXTS
        and (not x.name.startswith("."))
    ]
    if not files:
        raise RuntimeError(f"No image files found in: {input_path}")
    return files


def _load_rgb(path: Path, upscale: float = 1.0, resize_to: int = 0) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if upscale is not None and upscale > 1.0:
        w, h = img.size
        img = img.resize(
            (int(round(w * upscale)), int(round(h * upscale))), Image.BICUBIC
        )
    if resize_to is not None and resize_to > 0:
        img = img.resize((resize_to, resize_to), Image.BICUBIC)
    return img


def _pad_pil_to_multiple(img: Image.Image, multiple: int):
    if multiple is None or multiple <= 1:
        return (img, (0, 0))
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return (img, (0, 0))
    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return (Image.fromarray(arr), (pad_h, pad_w))


def _pad_pil_with_reflect_halo(img: Image.Image, halo: int):
    halo = int(halo)
    if halo <= 0:
        return (img, (0, 0, 0, 0))
    arr = np.asarray(img)
    arr = np.pad(arr, ((halo, halo), (halo, halo), (0, 0)), mode="reflect")
    return (Image.fromarray(arr), (halo, halo, halo, halo))


def _crop_pil_border(img: Image.Image, crop_ltrb):
    left, top, right, bottom = [int(x) for x in crop_ltrb]
    if left == 0 and top == 0 and (right == 0) and (bottom == 0):
        return img
    w, h = img.size
    return img.crop((left, top, max(left + 1, w - right), max(top + 1, h - bottom)))


def _load_mask_gray(mask_path: Path, target_hw=None) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    if target_hw is not None and mask.shape[:2] != tuple(target_hw):
        target_h, target_w = [int(x) for x in target_hw]
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask


def _find_matching_mask(
    img_path: Path, input_root: Path | None, mask_root: str
) -> Path | None:
    if not mask_root:
        return None
    mask_root_path = Path(mask_root)
    candidates = []
    if input_root is not None:
        rel = img_path.relative_to(input_root)
        candidates.append(mask_root_path / rel)
        candidates.append((mask_root_path / rel).with_suffix(".png"))
        candidates.append((mask_root_path / rel).with_suffix(".jpg"))
    candidates.append(mask_root_path / f"{img_path.stem}.png")
    candidates.append(mask_root_path / f"{img_path.stem}.jpg")
    candidates.append(mask_root_path / f"{img_path.stem}.jpeg")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    if input_root is not None:
        rel_parent = img_path.relative_to(input_root).parent
        search_dir = mask_root_path / rel_parent
        if search_dir.is_dir():
            for ext in IMAGE_EXTS:
                candidate = search_dir / f"{img_path.stem}{ext}"
                if candidate.exists():
                    return candidate
    return None


def _bbox_from_mask(mask_u8: np.ndarray, threshold: int, pad: int = 0):
    active = mask_u8 > int(threshold)
    ys, xs = np.where(active)
    if ys.size == 0 or xs.size == 0:
        return None
    y1 = max(0, int(ys.min()) - int(pad))
    y2 = min(mask_u8.shape[0] - 1, int(ys.max()) + int(pad))
    x1 = max(0, int(xs.min()) - int(pad))
    x2 = min(mask_u8.shape[1] - 1, int(xs.max()) + int(pad))
    return (x1, y1, x2, y2)


def _crop_pil_by_bbox(img: Image.Image, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return img.crop((x1, y1, x2 + 1, y2 + 1))


def _composite_restored_crop(
    base_img: Image.Image,
    restored_crop: Image.Image,
    full_mask_u8: np.ndarray,
    bbox,
    threshold: int,
    blend_radius: int = 0,
):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    base_np = np.asarray(base_img).astype(np.float32)
    restored_np = np.asarray(restored_crop).astype(np.float32)
    mask_crop = (full_mask_u8[y1 : y2 + 1, x1 : x2 + 1] > int(threshold)).astype(
        np.float32
    )
    if restored_np.shape[:2] != mask_crop.shape[:2]:
        restored_np = cv2.resize(
            restored_np,
            (mask_crop.shape[1], mask_crop.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    if int(blend_radius) > 0:
        mask_crop = cv2.GaussianBlur(
            mask_crop, (0, 0), sigmaX=float(blend_radius), sigmaY=float(blend_radius)
        )
        mask_crop = np.clip(mask_crop, 0.0, 1.0)
    mask_crop = mask_crop[..., None]
    region = base_np[y1 : y2 + 1, x1 : x2 + 1]
    region = restored_np * mask_crop + region * (1.0 - mask_crop)
    base_np[y1 : y2 + 1, x1 : x2 + 1] = region
    return Image.fromarray(np.clip(base_np, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _img_to_tensor_m1p1(img: Image.Image) -> torch.Tensor:
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t * 2.0 - 1.0


def _tensor_to_pil_01(x: torch.Tensor) -> Image.Image:
    x = x[0].detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
    return transforms.ToPILImage()(x)


def _pil_to_np01(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _np01_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _resize_down_to_max_dim(img: Image.Image, max_dim: int):
    max_dim = int(max_dim)
    if max_dim <= 0:
        return (img, 1.0)
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_dim:
        return (img, 1.0)
    scale = float(max_dim) / float(max(1, long_side))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return (img.resize((new_w, new_h), Image.BICUBIC), scale)


def _dtype_bytes_from_precision(mixed_precision: str) -> float:
    return 2.0 if str(mixed_precision).lower() in ("fp16", "bf16") else 4.0


def _estimate_fullres_no_tile_gib(args, img: Image.Image) -> float:
    w, h = img.size
    area_mp = float(w) * float(h) / 1000000.0
    latent_tokens = float((w + 7) // 8 * ((h + 7) // 8))
    attn_term = (latent_tokens / 10000.0) ** 2 * 4.0
    linear_term = area_mp * 3.0
    base_term = 2.5 + 1.4
    dtype_scale = (
        1.0
        if _dtype_bytes_from_precision(getattr(args, "mixed_precision", "fp16")) <= 2.0
        else 1.35
    )
    return (base_term + linear_term + attn_term) * dtype_scale


def _cuda_memory_snapshot(device) -> dict | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    gib = float(1024**3)
    return {
        "free_gib": float(free_bytes) / gib,
        "total_gib": float(total_bytes) / gib,
        "reserved_gib": float(torch.cuda.memory_reserved(device)) / gib,
        "allocated_gib": float(torch.cuda.memory_allocated(device)) / gib,
    }


def _adaptive_reserve_gib(snapshot: dict | None) -> float:
    if not snapshot:
        return 4.0
    return max(4.0, float(snapshot["total_gib"]) * 0.08)


def _build_coarse_dim_candidates(
    img: Image.Image, requested_max_dim: int, minimum_max_dim: int = 640
):
    long_side = max(img.size)
    start = max(minimum_max_dim, min(int(requested_max_dim), int(long_side)))
    dims = []
    cur = start
    while cur >= minimum_max_dim:
        dims.append(cur)
        if cur == minimum_max_dim:
            break
        cur = max(minimum_max_dim, cur - 128)
        if dims and cur == dims[-1]:
            break
    if not dims:
        dims = [max(minimum_max_dim, start)]
    return dims


def _fuse_with_global_baseline(
    fine_pil: Image.Image,
    coarse_pil: Image.Image,
    *,
    luma_strength: float,
    chroma_strength: float,
    sigma: float,
):
    if coarse_pil.size != fine_pil.size:
        coarse_pil = coarse_pil.resize(fine_pil.size, Image.BICUBIC)
    fine = _pil_to_np01(fine_pil)
    coarse = _pil_to_np01(coarse_pil)
    fine_ycc = cv2.cvtColor(fine, cv2.COLOR_RGB2YCrCb)
    coarse_ycc = cv2.cvtColor(coarse, cv2.COLOR_RGB2YCrCb)
    sigma = max(0.1, float(sigma))
    out = fine_ycc.copy()
    strengths = (
        float(np.clip(luma_strength, 0.0, 1.5)),
        float(np.clip(chroma_strength, 0.0, 1.5)),
        float(np.clip(chroma_strength, 0.0, 1.5)),
    )
    for ch, strength in enumerate(strengths):
        if strength <= 1e-06:
            continue
        fine_lp = cv2.GaussianBlur(
            fine_ycc[..., ch], (0, 0), sigmaX=sigma, sigmaY=sigma
        )
        coarse_lp = cv2.GaussianBlur(
            coarse_ycc[..., ch], (0, 0), sigmaX=sigma, sigmaY=sigma
        )
        out[..., ch] = fine_ycc[..., ch] + strength * (coarse_lp - fine_lp)
    out_rgb = cv2.cvtColor(np.clip(out, 0.0, 1.0), cv2.COLOR_YCrCb2RGB)
    return _np01_to_pil(out_rgb)


def _build_prior_tensor(
    img: Image.Image,
    fast_scale: float,
    prior_type: str,
    device: str,
    dtype: torch.dtype,
):
    rgb_np = np.array(img).astype(np.float32) / 255.0
    h, w = rgb_np.shape[:2]
    if fast_scale is not None and fast_scale < 1.0:
        sw = max(32, int(round(w * fast_scale)))
        sh = max(32, int(round(h * fast_scale)))
        small = img.resize((sw, sh), Image.BICUBIC)
        prior_small = extract_banding_prior(np.array(small).astype(np.float32) / 255.0)
        prior_t = torch.from_numpy(prior_small).unsqueeze(0)
        prior_t = F.interpolate(
            prior_t, size=(h, w), mode="bilinear", align_corners=False
        )
    else:
        prior = extract_banding_prior(rgb_np)
        prior_t = torch.from_numpy(prior).unsqueeze(0)
    return prior_t.to(device=device, dtype=dtype)


def _build_retry_profiles(args, allow_stricter_tiling: bool = True):
    profiles = [
        {
            "name": "default",
            "vae_encoder_tiled_size": int(args.vae_encoder_tiled_size),
            "vae_decoder_tiled_size": int(args.vae_decoder_tiled_size),
            "use_latent_tiling": bool(args.use_latent_tiling),
            "latent_tiled_size": int(args.latent_tiled_size),
            "latent_tiled_overlap": int(args.latent_tiled_overlap),
        }
    ]
    if allow_stricter_tiling:
        profiles.extend(
            [
                {
                    "name": "retry_tile_128",
                    "vae_encoder_tiled_size": min(
                        int(args.vae_encoder_tiled_size), 1024
                    ),
                    "vae_decoder_tiled_size": min(
                        int(args.vae_decoder_tiled_size), 128
                    ),
                    "use_latent_tiling": True,
                    "latent_tiled_size": min(int(args.latent_tiled_size), 96),
                    "latent_tiled_overlap": min(int(args.latent_tiled_overlap), 32),
                },
                {
                    "name": "retry_tile_96",
                    "vae_encoder_tiled_size": min(
                        int(args.vae_encoder_tiled_size), 768
                    ),
                    "vae_decoder_tiled_size": 96,
                    "use_latent_tiling": True,
                    "latent_tiled_size": 64,
                    "latent_tiled_overlap": 16,
                },
                {
                    "name": "retry_tile_64",
                    "vae_encoder_tiled_size": min(
                        int(args.vae_encoder_tiled_size), 512
                    ),
                    "vae_decoder_tiled_size": 64,
                    "use_latent_tiling": True,
                    "latent_tiled_size": 48,
                    "latent_tiled_overlap": 12,
                },
            ]
        )
    dedup = []
    seen = set()
    for p in profiles:
        key = (
            p["vae_encoder_tiled_size"],
            p["vae_decoder_tiled_size"],
            p["use_latent_tiling"],
            p["latent_tiled_size"],
            p["latent_tiled_overlap"],
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _pick_initial_profile_idx(args, img: Image.Image, profiles):
    if not bool(getattr(args, "auto_tile_by_resolution", True)):
        return 0
    w, h = img.size
    max_dim = max(w, h)
    area = w * h
    if max_dim >= 1800 or area >= 1600 * 1200:
        return min(2, len(profiles) - 1)
    if max_dim >= 1200 or area >= 1100 * 900:
        return min(1, len(profiles) - 1)
    return 0


def _make_runtime_args(base_args, profile):
    runtime_args = copy.deepcopy(base_args)
    runtime_args.vae_encoder_tiled_size = int(profile["vae_encoder_tiled_size"])
    runtime_args.vae_decoder_tiled_size = int(profile["vae_decoder_tiled_size"])
    runtime_args.use_latent_tiling = bool(profile["use_latent_tiling"])
    runtime_args.latent_tiled_size = int(profile["latent_tiled_size"])
    runtime_args.latent_tiled_overlap = int(profile["latent_tiled_overlap"])
    return runtime_args


def _create_model(runtime_args):
    model = RIFLEInference(runtime_args)
    model.set_eval()
    return model


def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _release_model_memory(model):
    if model is not None:
        model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _switch_model_profile(model, base_args, retry_profiles, profile_idx: int):
    _release_model_memory(model)
    try:
        del model
    except UnboundLocalError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()
    runtime_args = _make_runtime_args(base_args, retry_profiles[profile_idx])
    model = _create_model(runtime_args)
    return (model, runtime_args)


def _decode_model_output(out: torch.Tensor, halo_img: Image.Image, halo_ltrb, pad_hw):
    output_pil = _tensor_to_pil_01(out)
    if pad_hw != (0, 0):
        orig_w, orig_h = halo_img.size
        output_pil = output_pil.crop((0, 0, orig_w, orig_h))
    if halo_ltrb != (0, 0, 0, 0):
        output_pil = _crop_pil_border(output_pil, halo_ltrb)
    return output_pil


def _run_with_retry_profiles(
    model,
    base_args,
    retry_profiles,
    active_profile_idx: int,
    img: Image.Image,
    work_img: Image.Image,
    img_label: str,
):
    profile_start_idx = max(
        active_profile_idx, _pick_initial_profile_idx(base_args, img, retry_profiles)
    )
    if profile_start_idx != active_profile_idx:
        model, runtime_args = _switch_model_profile(
            model, base_args, retry_profiles, profile_start_idx
        )
        active_profile_idx = profile_start_idx
    else:
        runtime_args = _make_runtime_args(base_args, retry_profiles[active_profile_idx])
    infer_time = None
    out = None
    last_exc = None
    for profile_idx in range(profile_start_idx, len(retry_profiles)):
        profile = retry_profiles[profile_idx]
        if profile_idx != active_profile_idx:
            model, runtime_args = _switch_model_profile(
                model, base_args, retry_profiles, profile_idx
            )
        device = model.device
        dtype = model.weight_dtype
        x = _img_to_tensor_m1p1(work_img).to(device=device, dtype=dtype)
        prior_t = None
        prior_t = _build_prior_tensor(
            work_img,
            runtime_args.prior_fast_scale,
            str("gp_band"),
            device=device,
            dtype=dtype,
        )
        try:
            with torch.no_grad():
                infer_time, out = model(
                    c_t=x, prompt=runtime_args.prompt, prior=prior_t
                )
            active_profile_idx = profile_idx
            if profile_idx > profile_start_idx:
                print(
                    f"[Retry] {img_label} succeeded with profile={profile['name']} (enc={profile['vae_encoder_tiled_size']}, dec={profile['vae_decoder_tiled_size']}, latent={profile['latent_tiled_size']}, overlap={profile['latent_tiled_overlap']})"
                )
            return (model, runtime_args, active_profile_idx, infer_time, out)
        except RuntimeError as exc:
            err_msg = str(exc)
            if not _is_cuda_oom(exc) or profile_idx == len(retry_profiles) - 1:
                raise RuntimeError(err_msg)
            last_exc = RuntimeError(err_msg)
            print(
                f"[Warn] CUDA OOM on {img_label} with profile={profile['name']}; retrying with stricter tiling."
            )
            _release_model_memory(locals().get("model"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            gc.collect()
            continue
    raise RuntimeError(f"Failed to infer {img_label} after retries: {last_exc}")


def _run_single_pass(
    model,
    args,
    retry_profiles,
    active_profile_idx: int,
    img: Image.Image,
    img_label: str,
):
    halo_img, halo_ltrb = _pad_pil_with_reflect_halo(
        img, int(getattr(args, "edge_reflect_pad", 0))
    )
    work_img, pad_hw = _pad_pil_to_multiple(
        halo_img, int(getattr(args, "pad_to_multiple", 0))
    )
    model, runtime_args, active_profile_idx, infer_time, out = _run_with_retry_profiles(
        model, args, retry_profiles, active_profile_idx, img, work_img, img_label
    )
    output_pil = _decode_model_output(out, halo_img, halo_ltrb, pad_hw)
    return (model, runtime_args, active_profile_idx, float(infer_time), output_pil)


def _make_raw_no_tile_args(base_args):
    args = copy.deepcopy(base_args)
    args.use_vae_tiling = False
    args.use_latent_tiling = False
    return args


def _make_raw_tile_args(base_args):
    args = copy.deepcopy(base_args)
    args.use_vae_tiling = True
    args.use_latent_tiling = True
    return args


def _build_infer_context(args, *, allow_stricter_tiling: bool):
    retry_profiles = _build_retry_profiles(
        args, allow_stricter_tiling=allow_stricter_tiling
    )
    runtime_args = _make_runtime_args(args, retry_profiles[0])
    model = _create_model(runtime_args)
    return {
        "args": args,
        "retry_profiles": retry_profiles,
        "active_profile_idx": 0,
        "model": model,
    }


def _reset_infer_context(ctx):
    old_model = ctx.get("model")
    if old_model is not None:
        _release_model_memory(old_model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()
    ctx["active_profile_idx"] = 0
    runtime_args = _make_runtime_args(ctx["args"], ctx["retry_profiles"][0])
    ctx["model"] = _create_model(runtime_args)
    return ctx


def _release_infer_context(ctx):
    if not ctx:
        return
    model = ctx.get("model")
    if model is not None:
        _release_model_memory(model)
    ctx["model"] = None


def _run_ctx_on_image(ctx, img: Image.Image, img_label: str):
    model, runtime_args, active_profile_idx, infer_time, output_pil = _run_single_pass(
        ctx["model"],
        ctx["args"],
        ctx["retry_profiles"],
        ctx["active_profile_idx"],
        img,
        img_label,
    )
    ctx["model"] = model
    ctx["active_profile_idx"] = active_profile_idx
    return (output_pil, infer_time)


def _run_adaptive_fullres_or_coarse_tile(args):
    os.makedirs(args.output_dir, exist_ok=True)
    inputs = _list_inputs(args.input_path)
    input_root = Path(args.input_path) if Path(args.input_path).is_dir() else None
    total_time = 0.0
    full_ctx = _build_infer_context(
        _make_raw_no_tile_args(args), allow_stricter_tiling=False
    )
    tile_ctx = None
    try:
        for idx, img_path in enumerate(inputs, start=1):
            full_img = _load_rgb(img_path, args.upscale, args.resize_to)
            img = full_img
            save_img = None
            seg_bbox = None
            full_mask_u8 = None
            if args.process_mode == "mask_image_segmented":
                mask_path = _find_matching_mask(img_path, input_root, args.mask_dir)
                if mask_path is None:
                    print(
                        f"[Warn] no matching mask for {img_path.name}; fallback to full-image inference."
                    )
                else:
                    full_mask_u8 = _load_mask_gray(
                        mask_path, target_hw=(full_img.size[1], full_img.size[0])
                    )
                    seg_bbox = _bbox_from_mask(
                        full_mask_u8,
                        threshold=int(args.mask_threshold),
                        pad=int(args.mask_bbox_pad),
                    )
                    if seg_bbox is None:
                        print(
                            f"[Warn] empty mask for {img_path.name}; fallback to full-image inference."
                        )
                    else:
                        img = _crop_pil_by_bbox(full_img, seg_bbox)
            full_snapshot = _cuda_memory_snapshot(
                getattr(full_ctx.get("model"), "device", None)
            )
            full_est_gib = _estimate_fullres_no_tile_gib(full_ctx["args"], img)
            reserve_gib = _adaptive_reserve_gib(full_snapshot)
            should_try_full = True
            if full_snapshot is not None:
                should_try_full = (
                    float(full_snapshot["free_gib"]) >= full_est_gib + reserve_gib
                )
                plan = "try_full_no_tile" if should_try_full else "skip_to_coarse_tile"
                print(
                    f"[MemPlan] {img_path.name}: free={full_snapshot['free_gib']:.2f}GiB reserved={full_snapshot['reserved_gib']:.2f}GiB est_full_no_tile={full_est_gib:.2f}GiB -> {plan}"
                )
            strategy_used = "full_no_tile"
            output_pil = None
            total_infer_time = 0.0
            full_no_tile_exc = None
            if should_try_full:
                try:
                    output_pil, total_infer_time = _run_ctx_on_image(
                        full_ctx, img, f"{img_path.name}::full_no_tile"
                    )
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    full_no_tile_exc = exc
                    print(
                        f"[Strategy] full-resolution no-tile OOM on {img_path.name}; falling back to coarse+tile-detail."
                    )
                    full_ctx = _reset_infer_context(full_ctx)
            if output_pil is None:
                strategy_used = "coarse_plus_tile_detail"
                coarse_snapshot = _cuda_memory_snapshot(
                    getattr(full_ctx.get("model"), "device", None)
                )
                coarse_dims = _build_coarse_dim_candidates(
                    img, int(getattr(args, "coarse_fallback_max_dim", 1024))
                )
                coarse_output_pil = None
                coarse_infer_time = 0.0
                chosen_coarse_img = None
                chosen_ratio = 1.0
                last_oom = full_no_tile_exc
                for coarse_max_dim in coarse_dims:
                    coarse_img, downscale_ratio = _resize_down_to_max_dim(
                        img, coarse_max_dim
                    )
                    coarse_est_gib = _estimate_fullres_no_tile_gib(
                        full_ctx["args"], coarse_img
                    )
                    if coarse_snapshot is not None:
                        fits = float(
                            coarse_snapshot["free_gib"]
                        ) >= coarse_est_gib + _adaptive_reserve_gib(coarse_snapshot)
                        print(
                            f"[MemPlan] {img_path.name}: coarse_candidate={coarse_img.size[0]}x{coarse_img.size[1]} est={coarse_est_gib:.2f}GiB free={coarse_snapshot['free_gib']:.2f}GiB fit={('yes' if fits else 'no')}"
                        )
                        if not fits and coarse_max_dim != coarse_dims[-1]:
                            continue
                    try:
                        coarse_output_pil, coarse_infer_time = _run_ctx_on_image(
                            full_ctx, coarse_img, f"{img_path.name}::coarse_no_tile"
                        )
                        chosen_coarse_img = coarse_img
                        chosen_ratio = downscale_ratio
                        break
                    except RuntimeError as exc:
                        if not _is_cuda_oom(exc):
                            raise
                        last_oom = exc
                        full_ctx = _reset_infer_context(full_ctx)
                        continue
                if coarse_output_pil is None or chosen_coarse_img is None:
                    raise (
                        last_oom
                        if last_oom is not None
                        else RuntimeError(f"Failed coarse fallback for {img_path.name}")
                    )
                if chosen_ratio < 1.0 - 1e-06:
                    print(
                        f"[Strategy] {img_path.name}: coarse fallback {img.size[0]}x{img.size[1]} -> {chosen_coarse_img.size[0]}x{chosen_coarse_img.size[1]}"
                    )
                if tile_ctx is None:
                    tile_ctx = _build_infer_context(
                        _make_raw_tile_args(args), allow_stricter_tiling=True
                    )
                tile_output_pil, tile_infer_time = _run_ctx_on_image(
                    tile_ctx, img, f"{img_path.name}::tile_detail"
                )
                output_pil = _fuse_with_global_baseline(
                    tile_output_pil,
                    coarse_output_pil,
                    luma_strength=float(
                        getattr(args, "coarse_tile_fuse_luma_strength", 1.0)
                    ),
                    chroma_strength=float(
                        getattr(args, "coarse_tile_fuse_chroma_strength", 0.35)
                    ),
                    sigma=float(getattr(args, "coarse_tile_fuse_sigma", 12.0)),
                )
                total_infer_time = float(coarse_infer_time) + float(tile_infer_time)
            total_time += total_infer_time
            stem = img_path.stem
            suffix = f"_{args.suffix}" if args.suffix else ""
            save_name = f"{stem}{suffix}.png"
            if input_root is not None:
                rel = img_path.relative_to(input_root)
                save_path = Path(args.output_dir) / rel.parent / save_name
            else:
                save_path = Path(args.output_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if seg_bbox is not None and full_mask_u8 is not None:
                save_img = _composite_restored_crop(
                    full_img,
                    output_pil,
                    full_mask_u8,
                    seg_bbox,
                    int(args.mask_threshold),
                    int(getattr(args, "mask_blend_radius", 0)),
                )
            else:
                save_img = output_pil
            save_img.save(save_path)
            print(
                f"[{idx}/{len(inputs)}] saved: {save_path} | {strategy_used} | {total_infer_time:.4f}s"
            )
    finally:
        _release_infer_context(full_ctx)
        _release_infer_context(tile_ctx)
    print(
        f"Done. {len(inputs)} images, avg infer time: {total_time / max(len(inputs), 1):.4f}s"
    )


def main(args=None):
    args = parse_args() if args is None else args
    args.pretrained_model_path = resolve_pretrained_model_path(
        args.pretrained_model_path, cwd=str(Path(__file__).parent)
    )
    checkpoint_config = _load_ckpt_cfg(args.pretrained_path)
    args.prior_fast_scale = float(checkpoint_config.get("prior_fast_scale", 1.0))
    return _run_adaptive_fullres_or_coarse_tile(args)


if __name__ == "__main__":
    main()
