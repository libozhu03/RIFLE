"""Retained runner behavior without downloading a diffusion backbone."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch


class RunnerTests(unittest.TestCase):
    def test_resume_selects_semantic_adapters_without_resetting_warmup(self):
        from train import _set_training_phase

        model = SimpleNamespace(
            unet=SimpleNamespace(set_adapter=Mock()),
            set_train_pix=Mock(),
            set_train_sem=Mock(),
        )
        _set_training_phase(model, 4000, 1000)
        model.set_train_sem.assert_called_once()
        self.assertEqual(len(model.unet.set_adapter.call_args.args[0]), 6)
        _set_training_phase(model, 500, 1000)
        model.set_train_pix.assert_called_once()
        self.assertEqual(len(model.unet.set_adapter.call_args.args[0]), 3)

    def test_masked_inference_preserves_pixels_outside_screen(self):
        import infer

        test = self

        class FakeRIFLE(torch.nn.Module):
            device = torch.device("cpu")
            weight_dtype = torch.float32

            def forward(self, c_t, prompt="", prior=None):
                test.assertEqual(prior.shape[1], 1)
                return 0.0, (c_t + 0.2).clamp(-1, 1)

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / "lq").mkdir()
            (temp / "masks").mkdir()
            pixels = np.full((96, 112, 3), 128, dtype=np.uint8)
            mask = np.zeros((96, 112), dtype=np.uint8)
            mask[20:60, 30:80] = 255
            Image.fromarray(pixels).save(temp / "lq/image.png")
            Image.fromarray(mask).save(temp / "masks/image.png")
            args = infer.parse_args(
                [
                    "--input_path",
                    str(temp / "lq"),
                    "--output_dir",
                    str(temp / "out"),
                    "--pretrained_path",
                    "unused.pkl",
                    "--mask_dir",
                    str(temp / "masks"),
                ]
            )
            with (
                patch.object(infer, "_create_model", return_value=FakeRIFLE()),
                patch.object(infer, "_load_ckpt_cfg", return_value={}),
            ):
                infer.main(args)
            restored = np.asarray(Image.open(temp / "out/image.png"))
            np.testing.assert_array_equal(restored[mask == 0], pixels[mask == 0])
            self.assertGreater(int(restored[mask > 0].mean()), 128)


if __name__ == "__main__":
    unittest.main()
