"""Online Y-banding training data and matching-image validation data."""

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as F

from .prior import extract_banding_prior
from .simulation import (
    generate_stripe_occupancy,
    sample_runtime_params,
    simulate_banding_on_ref,
)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(directory: Path) -> list[str]:
    """Discover visible images recursively, excluding metadata and hidden folders."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        str(path)
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _IMG_EXTS
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def _list_paired_files(lq_dir: Path, gt_dir: Path):
    def visible_images(directory):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Validation image directory not found: {directory}"
            )
        return {
            p.name: p
            for p in directory.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in _IMG_EXTS
        }

    lq_files, gt_files = visible_images(lq_dir), visible_images(gt_dir)
    if not lq_files or not gt_files:
        raise ValueError(
            f"Validation directories must contain images: LQ={lq_dir}, GT={gt_dir}"
        )
    if lq_files.keys() != gt_files.keys():
        missing_gt = sorted(lq_files.keys() - gt_files.keys())
        missing_lq = sorted(gt_files.keys() - lq_files.keys())
        raise ValueError(
            f"Validation filename mismatch: missing GT={missing_gt}, missing LQ={missing_lq}"
        )
    names = sorted(lq_files)
    return [str(lq_files[n]) for n in names], [str(gt_files[n]) for n in names]


def _rescale_pil(img, scale):
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return img
    new_w = max(1, round(img.width * scale))
    new_h = max(1, round(img.height * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _to_mask_image(mask_np: np.ndarray) -> Image.Image:
    mask_np = np.asarray(mask_np, dtype=np.float32)
    mask_np = np.clip(mask_np, 0.0, 1.0)
    return Image.fromarray((mask_np * 255.0).astype(np.uint8), mode="L")


def _rgb_to_luma_np(rgb_np: np.ndarray) -> np.ndarray:
    rgb_np = np.asarray(rgb_np, dtype=np.float32)
    if rgb_np.max() > 1.5:
        rgb_np = rgb_np / 255.0
    return (
        0.299 * rgb_np[..., 0] + 0.587 * rgb_np[..., 1] + 0.114 * rgb_np[..., 2]
    ).astype(np.float32)


def _sample_train_scale(args):
    scale_min = float(getattr(args, "train_scale_min", 1.0))
    scale_max = float(getattr(args, "train_scale_max", 1.0))
    scale_prob = float(getattr(args, "train_scale_prob", 1.0))
    bucket_step = float(getattr(args, "train_scale_bucket_step", 0.0))

    if scale_min > scale_max:
        scale_min, scale_max = scale_max, scale_min
    if abs(scale_max - scale_min) < 1e-6 or random.random() > max(
        0.0, min(1.0, scale_prob)
    ):
        return 1.0

    scale = random.uniform(scale_min, scale_max)
    if bucket_step > 1e-6:
        scale = round(scale / bucket_step) * bucket_step
        scale = min(scale_max, max(scale_min, scale))
    return max(0.25, float(scale))


def _sample_context_crop_scale(args):
    scale_min = float(getattr(args, "train_context_scale_min", 1.0))
    scale_max = float(getattr(args, "train_context_scale_max", 1.0))
    scale_prob = float(getattr(args, "train_context_scale_prob", 0.0))
    bucket_step = float(getattr(args, "train_context_scale_bucket_step", 0.0))

    if scale_min > scale_max:
        scale_min, scale_max = scale_max, scale_min
    scale_min = max(1.0, scale_min)
    scale_max = max(scale_min, scale_max)
    if abs(scale_max - scale_min) < 1e-6 or random.random() > max(
        0.0, min(1.0, scale_prob)
    ):
        return 1.0

    scale = random.uniform(scale_min, scale_max)
    if bucket_step > 1e-6:
        scale = round(scale / bucket_step) * bucket_step
        scale = min(scale_max, max(scale_min, scale))
    return max(1.0, float(scale))


def _sample_crop_params(
    width: int, height: int, crop_h: int, crop_w: int, do_random: bool = True
):
    if height < crop_h or width < crop_w:
        return {
            "i": 0,
            "j": 0,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "pre_resize": True,
        }
    max_i = height - crop_h
    max_j = width - crop_w
    if do_random:
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
    else:
        i = max_i // 2
        j = max_j // 2
    return {
        "i": i,
        "j": j,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "pre_resize": False,
    }


def _crop_image_to_params(img: Image.Image, crop_params):
    if crop_params["pre_resize"]:
        return img.resize((crop_params["crop_w"], crop_params["crop_h"]), Image.BICUBIC)
    return F.crop(
        img,
        crop_params["i"],
        crop_params["j"],
        crop_params["crop_h"],
        crop_params["crop_w"],
    )


def _crop_mask_to_params(img: Image.Image, crop_params):
    if crop_params["pre_resize"]:
        return img.resize((crop_params["crop_w"], crop_params["crop_h"]), Image.NEAREST)
    return F.crop(
        img,
        crop_params["i"],
        crop_params["j"],
        crop_params["crop_h"],
        crop_params["crop_w"],
    )


def _sample_mask_guided_crop_params(
    mask_np: np.ndarray,
    width: int,
    height: int,
    crop_h: int,
    crop_w: int,
    *,
    rng: random.Random,
    threshold: int = 127,
):
    if height < crop_h or width < crop_w:
        return {
            "i": 0,
            "j": 0,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "pre_resize": True,
        }

    mask_np = np.asarray(mask_np)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    pos = np.argwhere(mask_np >= int(threshold))
    if pos.size == 0:
        return _sample_crop_params(width, height, crop_h, crop_w, do_random=False)

    ys = pos[:, 0]
    xs = pos[:, 1]
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cy = 0.5 * (y_min + y_max)
    cx = 0.5 * (x_min + x_max)
    bbox_h = max(1, y_max - y_min + 1)
    bbox_w = max(1, x_max - x_min + 1)

    jitter_y = min(max(8, bbox_h // 4), max(8, crop_h // 6))
    jitter_x = min(max(8, bbox_w // 4), max(8, crop_w // 6))
    y = int(round(cy + rng.randint(-jitter_y, jitter_y)))
    x = int(round(cx + rng.randint(-jitter_x, jitter_x)))
    max_i_full = height - crop_h
    max_j_full = width - crop_w
    min_i = max(0, int(y) - crop_h + 1)
    max_i = min(int(y), max_i_full)
    min_j = max(0, int(x) - crop_w + 1)
    max_j = min(int(x), max_j_full)

    if min_i <= max_i:
        i = rng.randint(min_i, max_i)
    else:
        i = min(max(0, int(y) - crop_h // 2), max_i_full)
    if min_j <= max_j:
        j = rng.randint(min_j, max_j)
    else:
        j = min(max(0, int(x) - crop_w // 2), max_j_full)

    return {
        "i": i,
        "j": j,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "pre_resize": False,
    }


class YBandingSimulator:
    """Render the release's Y-channel banding profiles and aligned occupancy masks."""

    def __init__(self, source_cfg):
        config_path = source_cfg.get("simulate_config")
        if not config_path:
            raise ValueError("Banding source requires 'simulate_config'.")
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self._generate_occ = generate_stripe_occupancy
        self._sample_runtime_params = sample_runtime_params
        self._simulate_banding = simulate_banding_on_ref

    def _recover_occ_from_band_mask(
        self, band_mask: np.ndarray, params: dict
    ) -> np.ndarray:
        band_mask = np.asarray(band_mask, dtype=np.float32)
        if band_mask.ndim == 3:
            band_mask = band_mask[..., 0]
        stripe_mode = str(params.get("stripe_mode", "dark")).lower()
        if stripe_mode == "black":
            export_v = float(params.get("export_black_mask_value", 0.25))
        else:
            export_v = float(
                params.get(
                    "export_dark_mask_value", params.get("dark_value_unified", 0.8)
                )
            )
        denom = max(1e-6, 1.0 - export_v)
        occ = (1.0 - band_mask) / denom
        return np.clip(occ, 0.0, 1.0).astype(np.float32)

    def degrade(self, gt_np: np.ndarray, seed: int):
        params = self._sample_runtime_params(self.cfg, seed=seed)
        gt_bgr_u8 = cv2.cvtColor(
            (np.clip(gt_np, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        sim_bgr_u8, band_mask = self._simulate_banding(gt_bgr_u8, params)
        occ_render = self._recover_occ_from_band_mask(band_mask, params)
        occ_analytic = self._generate_occ(gt_bgr_u8.shape[:2], params).astype(
            np.float32
        )
        sim_rgb = cv2.cvtColor(sim_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        debug = {
            "occ": np.clip(occ_render, 0.0, 1.0),
            "occ_render": np.clip(occ_render, 0.0, 1.0),
            "occ_analytic": np.clip(occ_analytic, 0.0, 1.0),
        }
        return sim_rgb, debug


class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_prior = bool(getattr(args, "use_prior", True))
        self.prior_fast_scale = float(getattr(args, "prior_fast_scale", 1.0))
        self.tgt_size = (args.resolution_tgt, args.resolution_tgt)

    def _build_example_from_pair(
        self,
        lq_img: Image.Image,
        gt_img: Image.Image,
        *,
        mask_img: Optional[Image.Image] = None,
        do_flip: bool = False,
        base_name: Optional[str] = None,
    ):
        if do_flip:
            lq_img = F.hflip(lq_img)
            gt_img = F.hflip(gt_img)
            if mask_img is not None:
                mask_img = F.hflip(mask_img)

        lq_img = F.resize(lq_img, self.tgt_size, interpolation=Image.BICUBIC)
        gt_img = F.resize(gt_img, self.tgt_size, interpolation=Image.BICUBIC)
        if mask_img is not None:
            mask_img = F.resize(mask_img, self.tgt_size, interpolation=Image.NEAREST)

        img_t = F.to_tensor(lq_img)
        img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        output_t = F.to_tensor(gt_img)
        output_t = F.normalize(output_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        example = {
            "neg_prompt": getattr(self.args, "neg_prompt_csd", ""),
            "null_prompt": "",
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
        }

        if self.use_prior:
            prior_t = self._build_prior_tensor(lq_img)
            example["prior"] = prior_t
        if mask_img is not None:
            example["mask_pixel_values"] = F.to_tensor(mask_img).clamp(0.0, 1.0)
        if base_name is not None:
            example["base_name"] = base_name
        return example

    def _build_prior_tensor(self, lq_img: Image.Image):
        rgb = np.array(lq_img).astype(np.float32) / 255.0
        height, width = rgb.shape[:2]
        if self.prior_fast_scale < 1.0:
            small = cv2.resize(
                rgb,
                (
                    max(32, int(round(width * self.prior_fast_scale))),
                    max(32, int(round(height * self.prior_fast_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            prior = extract_banding_prior(small)
            prior = np.stack(
                [
                    cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
                    for channel in prior
                ],
                axis=0,
            ).astype(np.float32)
        else:
            prior = extract_banding_prior(rgb)
        return torch.from_numpy(prior)


class ValidationDataset(_BaseDataset):
    """Resize full matching LQ/GT images to the validation target resolution."""

    def __init__(self, lq_dir, gt_dir, args):
        super().__init__(args)
        self.lr_list, self.gt_list = _list_paired_files(Path(lq_dir), Path(gt_dir))

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, idx):
        lq_path = Path(self.lr_list[idx])
        gt_path = Path(self.gt_list[idx])
        lq_img = Image.open(lq_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        return self._build_example_from_pair(lq_img, gt_img, base_name=lq_path.name)


class OnlineBandingDataset(_BaseDataset):
    """Simulate full clean images, then apply joint scale, context crop, and flip."""

    def __init__(self, source_cfg, args):
        super().__init__(args)
        self.crop_size = (args.resolution_ori, args.resolution_ori)
        self.downsample_scale = float(source_cfg.get("downsample_scale", 1.0))
        if self.downsample_scale <= 0:
            raise ValueError("downsample_scale must be positive.")
        self.train_mask_guided_context_prob = float(
            getattr(args, "train_mask_guided_context_prob", 0.0)
        )
        name = source_cfg.get("name", "<unnamed>")
        if not source_cfg.get("hq_dir"):
            raise ValueError(f"Banding source '{name}' requires 'hq_dir'.")
        self.hq_list = _list_images(Path(source_cfg["hq_dir"]))
        if not self.hq_list:
            raise ValueError(
                f"Banding source '{name}' contains no images: {source_cfg['hq_dir']}"
            )
        self.runtime = YBandingSimulator(source_cfg)

    def __len__(self):
        return len(self.hq_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.hq_list[idx]).convert("RGB")
        base_scale = self.downsample_scale
        if abs(base_scale - 1.0) > 1e-6:
            gt_img = _rescale_pil(gt_img, base_scale)
        lq_img, debug = self._degrade_full_image(gt_img)
        mask_img = self._make_mask_image(
            debug, np.asarray(gt_img), gt_img=gt_img, lq_img=lq_img
        )
        lq_img, gt_img, mask_img = self._apply_joint_scale(
            lq_img,
            gt_img,
            mask_img,
        )
        return self._finalize_pair(lq_img, gt_img, mask_img=mask_img)

    def _resolve_context_crop_hw(self, width: int, height: int):
        base_crop_h, base_crop_w = self.crop_size
        context_scale = _sample_context_crop_scale(self.args)
        max_context_scale = max(
            1.0,
            min(
                float(width) / float(max(1, base_crop_w)),
                float(height) / float(max(1, base_crop_h)),
            ),
        )
        context_scale = max(1.0, min(context_scale, max_context_scale))
        crop_h = max(base_crop_h, int(round(base_crop_h * context_scale)))
        crop_w = max(base_crop_w, int(round(base_crop_w * context_scale)))
        return crop_h, crop_w

    def _sample_pair_crop_params(
        self,
        width: int,
        height: int,
        mask_img: Optional[Image.Image] = None,
    ):
        crop_h, crop_w = self._resolve_context_crop_hw(width, height)
        use_mask_guided = (
            self.train_mask_guided_context_prob > 0.0
            and random.random() < self.train_mask_guided_context_prob
        )
        if not use_mask_guided:
            return _sample_crop_params(width, height, crop_h, crop_w, do_random=True)
        return _sample_mask_guided_crop_params(
            np.array(mask_img),
            width,
            height,
            crop_h,
            crop_w,
            rng=random,
            threshold=127,
        )

    def _apply_joint_scale(
        self,
        lq_img: Image.Image,
        gt_img: Image.Image,
        mask_img: Optional[Image.Image] = None,
    ):
        scale = _sample_train_scale(self.args)
        if abs(scale - 1.0) <= 1e-6:
            return lq_img, gt_img, mask_img

        lq_img = _rescale_pil(lq_img, scale)
        gt_img = _rescale_pil(gt_img, scale)
        if mask_img is not None:
            new_w = max(1, round(mask_img.width * scale))
            new_h = max(1, round(mask_img.height * scale))
            mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)
        return lq_img, gt_img, mask_img

    def _finalize_pair(
        self,
        lq_img: Image.Image,
        gt_img: Image.Image,
        mask_img: Optional[Image.Image] = None,
        base_name: Optional[str] = None,
    ):
        crop_params = self._sample_pair_crop_params(
            lq_img.width,
            lq_img.height,
            mask_img=mask_img,
        )

        lq_img = _crop_image_to_params(lq_img, crop_params)
        gt_img = _crop_image_to_params(gt_img, crop_params)
        if mask_img is not None:
            mask_img = _crop_mask_to_params(mask_img, crop_params)

        flip_flag = random.random() < 0.5
        example = self._build_example_from_pair(
            lq_img,
            gt_img,
            mask_img=mask_img,
            do_flip=flip_flag,
            base_name=base_name,
        )
        return example

    def _make_mask_image(
        self,
        debug,
        ref_shape,
        *,
        gt_img: Optional[Image.Image] = None,
        lq_img: Optional[Image.Image] = None,
    ):
        if hasattr(ref_shape, "shape"):
            ref_shape = ref_shape.shape
        occ = np.array(
            debug.get("occ", np.zeros(ref_shape[:2], dtype=np.float32)),
            dtype=np.float32,
        )
        occ = np.clip(occ, 0.0, 1.0)
        if gt_img is not None and lq_img is not None:
            gt_y = _rgb_to_luma_np(np.array(gt_img).astype(np.float32))
            lq_y = _rgb_to_luma_np(np.array(lq_img).astype(np.float32))
            delta_y = np.clip(gt_y - lq_y, 0.0, 1.0)
            valid = occ > 1e-4
            if np.any(valid):
                scale = float(np.percentile(delta_y[valid], 95))
            else:
                scale = float(np.percentile(delta_y, 95))
            scale = max(scale, 1e-4)
            severity = np.clip(delta_y / scale, 0.0, 1.0)
            # Keep stripe support from occ, but weight deep stripes more heavily.
            occ = np.clip(occ * (0.20 + 0.80 * np.power(severity, 1.5)), 0.0, 1.0)
        return _to_mask_image(occ)

    def _degrade_full_image(self, gt_img: Image.Image):
        gt_np = np.array(gt_img).astype(np.float32) / 255.0
        lq_np, debug = self.runtime.degrade(gt_np, seed=random.randint(0, 2**31 - 1))
        lq_img = Image.fromarray(
            (np.clip(lq_np, 0, 1) * 255.0).astype(np.uint8)
        ).convert("RGB")
        return lq_img, debug


def build_training_dataset(config_path, args):
    """Give each source its configured sampling mass, independent of image count."""
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    active_names = getattr(args, "active_train_sources", None)
    active_names = set(active_names) if active_names else None
    datasets, weights, summaries = [], [], []
    for source in cfg.get("sources", []):
        if not bool(source.get("enabled", True)):
            continue
        if active_names is not None and source.get("name") not in active_names:
            continue
        ratio = float(source.get("sample_ratio", 1.0))
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                f"Source '{source.get('name', '<unnamed>')}' sample_ratio must be positive and finite."
            )
        dataset = OnlineBandingDataset(source, args)
        datasets.append(dataset)
        weights.extend([ratio / len(dataset)] * len(dataset))
        summaries.append((source.get("name", "<unnamed>"), len(dataset), ratio))
    if not datasets:
        raise ValueError(f"No active sources found in data config: {config_path}")
    total_ratio = sum(ratio for _, _, ratio in summaries)
    for name, length, ratio in summaries:
        print(
            f"[DataConfig] {name}: {length} images, effective_prob={ratio / total_ratio:.3f}"
        )
    combined = torch.utils.data.ConcatDataset(datasets)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined),
        replacement=True,
    )
    return combined, sampler
