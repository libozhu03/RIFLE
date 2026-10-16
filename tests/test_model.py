"""CPU coverage for the released model and its checkpoint contract."""

import copy
import inspect
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import AutoencoderKL as NativeVAE
from diffusers import UNet2DConditionModel as NativeUNet
from diffusers import DDPMScheduler
from peft import LoraConfig

from rifle.diffusion import AutoencoderKL, UNet2DConditionModel
from rifle.model import (
    DistillationLoss,
    RIFLE,
    RIFLEInference,
    validate_checkpoint_config,
)
from rifle.vae import TiledVAE


def tiny_unet(cls=UNet2DConditionModel):
    return cls(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        cross_attention_dim=8,
        attention_head_dim=2,
        norm_num_groups=4,
    )


def tiny_vae(cls=AutoencoderKL):
    return cls(
        in_channels=3,
        out_channels=3,
        block_out_channels=(8,),
        latent_channels=4,
        norm_num_groups=4,
        sample_size=8,
    )


class TinyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, tokens):
        return (self.embedding(tokens),)


class TinyTokenizer:
    model_max_length = 3

    def __call__(self, caption, **kwargs):
        return SimpleNamespace(input_ids=torch.zeros(1, 3, dtype=torch.long))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_released_and_historical_disabled_metadata(self):
        validate_checkpoint_config(
            {
                "train_cfg": {
                    "prior_ctrl_mode": "film",
                    "prior_channels": "0",
                    "film_prior_type": "gp_band",
                    "use_prior": True,
                    "use_1f2_prior": False,
                    "use_detail_color_module": False,
                    "use_frequency_preprocess": False,
                    "tune_vae_decoder": False,
                }
            }
        )

    def test_unsupported_checkpoint_rejected_before_loading_base(self):
        cases = [
            {"prior_ctrl_mode": "controlnet"},
            {"prior_ctrl_mode": "latent_concat"},
            {"use_1f2_prior": True},
            {"use_detail_color_module": True},
            {"use_frequency_preprocess": True},
            {"tune_vae_decoder": True},
            {"prior_channels": "0,1"},
            {"film_prior_type": "edge_mix"},
            {"use_prior": False},
        ]
        for cfg in cases:
            with (
                self.subTest(cfg=cfg),
                patch("rifle.model.torch.load", return_value={"train_cfg": cfg}),
                patch("rifle.model.AutoTokenizer.from_pretrained") as base,
            ):
                with self.assertRaisesRegex(ValueError, "released RIFLE"):
                    RIFLEInference(SimpleNamespace(pretrained_path="unsupported.pkl"))
                base.assert_not_called()
        for branch in (
            "state_dict_vae",
            "controlnet",
            "prior_text_adapter",
            "detail_color_module",
        ):
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                validate_checkpoint_config({branch: {"weight": torch.ones(1)}})

    def test_public_forward_signatures(self):
        self.assertEqual(
            list(inspect.signature(RIFLE.forward).parameters),
            ["self", "c_t", "c_tgt", "batch", "args"],
        )
        self.assertEqual(
            list(inspect.signature(RIFLEInference.forward).parameters),
            ["self", "c_t", "prompt", "prior"],
        )

    def test_native_forward_and_config_signature_unchanged(self):
        model = tiny_unet().eval()
        native = tiny_unet(NativeUNet).eval()
        native.load_state_dict(model.state_dict())
        x, prompt = torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8)
        with torch.no_grad():
            torch.testing.assert_close(
                model(x, 1, prompt).sample, native(x, 1, prompt).sample, rtol=0, atol=0
            )
        self.assertEqual(
            inspect.signature(UNet2DConditionModel.__init__),
            inspect.signature(NativeUNet.__init__),
        )
        self.assertEqual(
            inspect.signature(AutoencoderKL.__init__),
            inspect.signature(NativeVAE.__init__),
        )
        vae, native_vae = tiny_vae().eval(), tiny_vae(NativeVAE).eval()
        native_vae.load_state_dict(vae.state_dict())
        rgb = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(
                vae(rgb).sample, native_vae(rgb).sample, rtol=0, atol=0
            )

    def test_merge_matches_active_pixel_then_semantic_lora(self):
        model = tiny_unet().eval()
        x, prompt = torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8)
        for name in ("pix", "sem"):
            model.add_adapter(
                LoraConfig(r=2, target_modules=["to_q", "conv_in"]), adapter_name=name
            )
            for key, parameter in model.named_parameters():
                if "lora_B" in key and f".{name}." in key:
                    torch.nn.init.normal_(parameter, std=0.02)
            model.set_adapter(name)
            with torch.no_grad():
                expected = model(x, 1, prompt).sample
                model.merge_and_unload()
                actual = model(x, 1, prompt).sample
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertFalse(any("lora" in key for key in model.state_dict()))

    def test_training_checkpoint_and_cpu_inference_round_trip(self):
        args = SimpleNamespace(
            pretrained_model_path="local-only",
            lora_rank_unet_pix=2,
            lora_rank_unet_sem=2,
            timesteps1=1,
            null_text_ratio=0,
            prior_inject_layers="down0,down1,mid",
            mixed_precision="no",
        )
        base_unet, base_vae, base_text = tiny_unet(), tiny_vae(), TinyTextEncoder()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("rifle.model.get_optimal_device", return_value=torch.device("cpu")),
            patch(
                "rifle.model.AutoTokenizer.from_pretrained",
                return_value=TinyTokenizer(),
            ),
            patch(
                "rifle.model.CLIPTextModel.from_pretrained",
                side_effect=lambda *a, **k: copy.deepcopy(base_text),
            ),
            patch(
                "rifle.model.UNet2DConditionModel.from_pretrained",
                side_effect=lambda *a, **k: copy.deepcopy(base_unet),
            ),
            patch(
                "rifle.model.AutoencoderKL.from_pretrained",
                side_effect=lambda *a, **k: copy.deepcopy(base_vae),
            ),
        ):
            model = RIFLE(args)
            model.unet.set_adapter(
                [
                    f"default_{group}_{phase}"
                    for phase in ("pix", "sem")
                    for group in ("encoder", "decoder", "others")
                ]
            )
            for name, parameter in model.unet.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.02)
            model.set_train_pix()
            self.assertTrue(
                all(
                    not p.requires_grad
                    for n, p in model.unet.named_parameters()
                    if "sem" in n
                )
            )
            model.set_train_sem()
            self.assertTrue(
                all(
                    not p.requires_grad
                    for n, p in model.unet.named_parameters()
                    if "pix" in n
                )
            )
            rgb, prior = torch.randn(1, 3, 8, 8), torch.rand(1, 1, 8, 8)
            model.eval()
            torch.manual_seed(42)
            image, latent, positive, negative = model(rgb, batch={"prior": prior})
            self.assertEqual(tuple(latent.shape), (1, 4, 8, 8))
            self.assertEqual(positive.shape, negative.shape)
            args.pretrained_path = f"{directory}/rifle.pkl"
            model.save_model(args.pretrained_path, global_step=7)
            checkpoint = torch.load(args.pretrained_path)
            validate_checkpoint_config(checkpoint)
            self.assertEqual(checkpoint["train_cfg"]["global_step"], 7)
            self.assertEqual(
                set(checkpoint["prior_film"]), set(model.prior_film.state_dict())
            )
            args.resume_ckpt = args.pretrained_path
            resumed = RIFLE(args)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    resumed.state_dict()[key], tensor, rtol=0, atol=0
                )
            inference = RIFLEInference(args)
            inference.set_eval()
            with patch(
                "torch.cuda.synchronize",
                side_effect=AssertionError("CPU must not synchronize CUDA"),
            ):
                torch.manual_seed(42)
                elapsed, output = inference(rgb, prior=prior)
            self.assertGreaterEqual(elapsed, 0)
            torch.testing.assert_close(output, image, rtol=2e-5, atol=2e-6)
            self.assertIsNone(inference._prior_film_cond)

    def test_tiled_vae_small_input_preserves_forward_and_dtype(self):
        vae = tiny_vae().eval()
        encoder = vae.encoder
        encoder.original_forward = encoder.forward
        image = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            expected = encoder(image)
            hook = TiledVAE(encoder, 32, is_decoder=False, to_gpu=False)
            actual = hook(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(next(encoder.parameters()).dtype, image.dtype)

    def test_vae_dtype_casts_at_quantization_boundaries(self):
        vae = tiny_vae().eval()
        rgb, z = torch.randn(1, 3, 8, 8), torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            encoded = vae.encoder(rgb)
            expected = vae.encode(rgb).latent_dist.parameters
            with patch.object(vae.encoder, "forward", return_value=encoded.double()):
                torch.testing.assert_close(
                    vae.encode(rgb).latent_dist.parameters, expected, rtol=0, atol=0
                )
            torch.testing.assert_close(
                vae.decode(z.double()).sample, vae.decode(z).sample, rtol=0, atol=0
            )

    def test_actual_vae_tiling_uses_global_attention_and_norm(self):
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            block_out_channels=(32,) * 4,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            layers_per_block=2,
            latent_channels=4,
            norm_num_groups=32,
            sample_size=96,
        ).eval()
        for net, size, decoder, image in (
            (vae.encoder, 16, False, torch.randn(1, 3, 96, 96)),
            (vae.decoder, 8, True, torch.randn(1, 4, 32, 32)),
        ):
            net.original_forward = lambda x: self.fail(
                "Expected actual tiled execution"
            )
            hook = TiledVAE(net, size, is_decoder=decoder)
            with (
                patch.object(
                    hook, "_run_global_attention", wraps=hook._run_global_attention
                ) as attention,
                torch.no_grad(),
            ):
                result = hook(image)
            attention.assert_called_once()
            expected_shape = (1, 3, 256, 256) if decoder else (1, 8, 12, 12)
            self.assertEqual(tuple(result.shape), expected_shape)
            self.assertTrue(torch.isfinite(result).all())

    def test_phase_two_distillation_produces_finite_latent_gradients(self):
        args = SimpleNamespace(
            pretrained_model_path_csd="local-only",
            enable_xformers_memory_efficient_attention=False,
            min_dm_step_ratio=0.1,
            max_dm_step_ratio=0.9,
            cfg_csd=2.5,
        )
        accelerator = SimpleNamespace(device=torch.device("cpu"), mixed_precision="no")
        with (
            patch(
                "rifle.model.UNet2DConditionModel.from_pretrained",
                return_value=tiny_unet(),
            ),
            patch(
                "rifle.model.DDPMScheduler.from_pretrained",
                return_value=DDPMScheduler(num_train_timesteps=20),
            ),
        ):
            loss = DistillationLoss(args, accelerator)
        latents = torch.randn(1, 4, 8, 8, requires_grad=True)
        value = loss.cal_csd(latents, torch.randn(1, 3, 8), torch.randn(1, 3, 8), args)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(latents.grad.abs().sum().item(), 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in loss.unet_fix.parameters())
        )


if __name__ == "__main__":
    unittest.main()
