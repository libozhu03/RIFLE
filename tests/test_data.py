"""CPU-only coverage of the release's training data and banding prior."""

from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import yaml

from rifle.data import (
    OnlineBandingDataset,
    ValidationDataset,
    YBandingSimulator,
    build_training_dataset,
)
from rifle.prior import BandingPriorFiLM, extract_banding_prior
from rifle.simulation import list_images_recursively

ROOT = Path(__file__).resolve().parents[1]


def data_args(**overrides):
    values = dict(
        resolution_ori=64,
        resolution_tgt=48,
        use_prior=True,
        neg_prompt_csd="banding",
        train_scale_min=0.8,
        train_scale_max=1.2,
        train_scale_bucket_step=0.1,
        train_context_scale_min=1.2,
        train_context_scale_max=1.8,
        train_context_scale_prob=1.0,
        train_mask_guided_context_prob=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def source(folder, profile="screenfocus", **overrides):
    values = dict(
        name=profile,
        hq_dir=str(folder),
        downsample_scale=0.8,
        simulate_config=str(ROOT / f"configs/simulation/{profile}.yaml"),
    )
    values.update(overrides)
    return values


def save_clean(folder, name="clean.png", seed=7):
    pixels = np.random.default_rng(seed).integers(
        40, 231, (160, 192, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(Path(folder) / name)
    return pixels


class BandingDataTests(unittest.TestCase):
    def test_simulator_is_deterministic_with_aligned_nonempty_masks(self):
        clean = (
            np.random.default_rng(7).uniform(0.2, 0.9, (128, 160, 3)).astype(np.float32)
        )
        for profile in ("screenfocus", "screenmix"):
            with self.subTest(profile=profile):
                renderer = YBandingSimulator(source("unused", profile))
                rendered, masks = renderer.degrade(clean, seed=42)
                repeated, repeated_masks = renderer.degrade(clean, seed=42)
                np.testing.assert_array_equal(rendered, repeated)
                self.assertEqual(rendered.shape, clean.shape)
                self.assertTrue(np.isfinite(rendered).all())
                self.assertGreater(float(np.abs(rendered - clean).max()), 0.05)
                for name in ("occ", "occ_render", "occ_analytic"):
                    mask = masks[name]
                    self.assertEqual(mask.shape, clean.shape[:2])
                    self.assertTrue(np.isfinite(mask).all())
                    self.assertGreater(float(mask.max()), 0)
                    self.assertGreaterEqual(float(mask.min()), 0)
                    self.assertLessEqual(float(mask.max()), 1)
                    np.testing.assert_array_equal(mask, repeated_masks[name])

    def test_online_crop_scale_and_prior_are_deterministic_and_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            save_clean(temp)
            dataset = OnlineBandingDataset(source(temp), data_args())
            random.seed(91)
            sample = dataset[0]
            random.seed(91)
            repeated = dataset[0]
        for key, channels in (
            ("conditioning_pixel_values", 3),
            ("output_pixel_values", 3),
            ("mask_pixel_values", 1),
            ("prior", 1),
        ):
            tensor = sample[key]
            self.assertEqual(tuple(tensor.shape), (channels, 48, 48))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.isfinite(tensor).all())
            torch.testing.assert_close(tensor, repeated[key], rtol=0, atol=0)
            self.assertGreaterEqual(float(tensor.min()), -1 if channels == 3 else 0)
            self.assertLessEqual(float(tensor.max()), 1)
        self.assertGreater(float(sample["mask_pixel_values"].max()), 0)
        rgb = (sample["conditioning_pixel_values"].numpy().transpose(1, 2, 0) + 1) / 2
        np.testing.assert_allclose(
            sample["prior"].numpy(), extract_banding_prior(rgb), atol=2e-6
        )

    def test_severity_mask_weights_deep_stripes(self):
        # Equal occupancy with two depths should retain support and emphasize the deeper stripe.
        dataset = OnlineBandingDataset.__new__(OnlineBandingDataset)
        gt = Image.fromarray(np.full((16, 16, 3), 200, np.uint8))
        lq_pixels = np.full((16, 16, 3), 200, np.uint8)
        lq_pixels[:, :8] = 180
        lq_pixels[:, 8:] = 100
        lq = Image.fromarray(lq_pixels)
        mask = np.asarray(
            dataset._make_mask_image(
                {"occ": np.ones((16, 16), np.float32)}, (16, 16), gt_img=gt, lq_img=lq
            )
        )
        self.assertGreater(int(mask[:, :8].min()), 0)
        self.assertGreater(float(mask[:, 8:].mean()), 3 * float(mask[:, :8].mean()))

    def test_source_probability_is_independent_of_source_length(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            first, second = folder / "first", folder / "second"
            first.mkdir()
            second.mkdir()
            save_clean(first)
            for index in range(3):
                save_clean(second, f"{index}.png", seed=index)
            config = folder / "data.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            source(first, name="first", sample_ratio=0.25),
                            source(second, name="second", sample_ratio=0.75),
                        ]
                    }
                )
            )
            dataset, sampler = build_training_dataset(config, data_args())
        self.assertEqual([len(ds) for ds in dataset.datasets], [1, 3])
        probabilities = sampler.weights / sampler.weights.sum()
        self.assertAlmostEqual(float(probabilities[0]), 0.25)
        self.assertAlmostEqual(float(probabilities[1:].sum()), 0.75)
        self.assertEqual(sampler.num_samples, 4)
        self.assertTrue(sampler.replacement)

    def test_validation_resizes_whole_images_and_matches_names(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            lq, gt = folder / "lq", folder / "gt"
            lq.mkdir()
            gt.mkdir()
            pixels = save_clean(lq, "pair.png")
            save_clean(gt, "pair.png", seed=9)
            dataset = ValidationDataset(lq, gt, data_args(use_prior=False))
            sample = dataset[0]
            expected = F.to_tensor(
                F.resize(Image.fromarray(pixels), (48, 48), interpolation=Image.BICUBIC)
            )
            torch.testing.assert_close(
                sample["conditioning_pixel_values"], expected * 2 - 1, rtol=0, atol=0
            )
            self.assertEqual(sample["base_name"], "pair.png")
            self.assertNotIn("prior", sample)
            save_clean(lq, "unmatched.png")
            with self.assertRaisesRegex(ValueError, "filename mismatch.*unmatched.png"):
                ValidationDataset(lq, gt, data_args())

    def test_hidden_files_and_directories_are_excluded_and_bad_sources_are_clear(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_clean(folder)
            (folder / "._clean.png").write_bytes(b"AppleDouble metadata")
            (folder / ".hidden").mkdir()
            save_clean(folder / ".hidden", "invisible.png")
            dataset = OnlineBandingDataset(source(folder), data_args())
            self.assertEqual([Path(p).name for p in dataset.hq_list], ["clean.png"])
            self.assertEqual(
                [p.name for p in list_images_recursively([folder])], ["clean.png"]
            )
            with self.assertRaisesRegex(FileNotFoundError, "Image directory not found"):
                OnlineBandingDataset(source(folder / "missing"), data_args())
            empty = folder / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "contains no images"):
                OnlineBandingDataset(source(empty), data_args())

    def test_prior_is_one_channel_and_film_preserves_checkpoint_keys(self):
        rgb = np.random.default_rng(17).uniform(0, 1, (40, 48, 3)).astype(np.float32)
        prior = extract_banding_prior(rgb)
        self.assertEqual(prior.shape, (1, 40, 48))
        self.assertEqual(prior.dtype, np.float32)
        self.assertGreater(float(prior.max()), 0.9)
        np.testing.assert_array_equal(
            extract_banding_prior(np.zeros_like(rgb)), np.zeros_like(prior)
        )
        film = BandingPriorFiLM(1, [4], hidden=8)
        self.assertEqual(
            set(film.state_dict()),
            {
                "dir_reduce.0.weight",
                "dir_reduce.0.bias",
                "shared.0.weight",
                "shared.0.bias",
                "heads.0.weight",
                "heads.0.bias",
            },
        )
        feature = torch.ones(1, 4, 5, 6)
        result = film(torch.from_numpy(prior)[None], [feature])[0]
        self.assertEqual(result.shape, feature.shape)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
