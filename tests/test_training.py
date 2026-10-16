"""Run warmup, semantic refinement, validation and resume with a tiny local model."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml


class TrainingLoopTests(unittest.TestCase):
    def test_train_and_resume_keep_the_semantic_phase(self):
        import train
        from rifle.config import parse_train_args

        phases = []
        distillation_calls = []

        class TinyUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_pix = torch.nn.Parameter(torch.tensor(0.8))
                self.lora_sem = torch.nn.Parameter(torch.tensor(0.01))

            def set_adapter(self, adapters):
                self.adapters = adapters

        class TinyRIFLE(torch.nn.Module):
            use_prior_ctrl = True

            def __init__(self, args):
                super().__init__()
                self.unet = TinyUNet()
                self.prior_film = torch.nn.Conv2d(1, 1, 1)
                if args.resume_ckpt:
                    self.load_state_dict(
                        torch.load(args.resume_ckpt, map_location="cpu")["model"]
                    )

            def set_train_pix(self):
                self.unet.lora_pix.requires_grad_(True)
                self.unet.lora_sem.requires_grad_(False)
                phases.append("pixel")

            def set_train_sem(self):
                self.unet.lora_pix.requires_grad_(False)
                self.unet.lora_sem.requires_grad_(True)
                phases.append("semantic")

            def forward(self, c_t, c_tgt=None, batch=None, args=None):
                prior = self.prior_film(batch["prior"].to(c_t.device))
                output = c_t * (self.unet.lora_pix + self.unet.lora_sem) + prior * 0.01
                embedding = torch.zeros(c_t.shape[0], 2, 4)
                return output, output, embedding, embedding

            def save_model(self, path, global_step=None):
                torch.save(
                    {"global_step": global_step, "model": self.state_dict()}, path
                )

        class TinyLPIPS(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)

        class TinyDistillation(torch.nn.Module):
            def __init__(self, args, accelerator):
                super().__init__()

            def cal_csd(self, latent, positive, negative, args):
                self_test.assertEqual(args.min_dm_step_ratio, 0.02)
                self_test.assertEqual(args.max_dm_step_ratio, 0.98)
                distillation_calls.append(args.pix_steps)
                return latent.square().mean()

        self_test = self
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hq = root / "hq"
            hq.mkdir()
            pixels = np.random.default_rng(12).integers(
                40, 220, (96, 96, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(hq / "image.png")
            config = root / "data.yaml"
            profile = (
                Path(train.__file__).resolve().parent
                / "configs/simulation/screenfocus.yaml"
            )
            config.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "name": "tiny",
                                "hq_dir": str(hq),
                                "simulate_config": str(profile),
                                "sample_ratio": 1.0,
                            }
                        ]
                    }
                )
            )
            args = parse_train_args([])
            args.data_config = str(config)
            args.output_dir = str(root / "run")
            args.val_sets = [{"name": "tiny", "lq_dir": str(hq), "gt_dir": str(hq)}]
            args.resolution_ori = args.resolution_tgt = 32
            args.max_train_steps = 3
            args.pix_steps = 2
            args.train_batch_size = 1
            args.dataloader_num_workers = args.eval_num_workers = 0
            args.mixed_precision = "no"
            args.report_to = None
            args.checkpointing_steps = args.eval_freq = 1
            accelerator = train.Accelerator
            with (
                patch.object(
                    train,
                    "Accelerator",
                    side_effect=lambda **kw: accelerator(cpu=True, **kw),
                ),
                patch.object(train, "RIFLE", TinyRIFLE),
                patch.object(train.lpips, "LPIPS", return_value=TinyLPIPS()),
                patch.object(train, "DistillationLoss", TinyDistillation),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                train.main(args)
                checkpoint = root / "run/checkpoints/model_3.pkl"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(distillation_calls, [2])
                args.resume_ckpt = str(checkpoint)
                args.max_train_steps = 4
                train.main(args)
            self.assertEqual(args.pix_steps, 2)
            self.assertEqual(distillation_calls, [2, 2])
            self.assertEqual(phases[-1], "semantic")
            self.assertTrue((root / "run/checkpoints/model_4.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
