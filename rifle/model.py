"""Released RIFLE: pixel/semantic LoRA restoration with banding-prior FiLM."""

import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig

from rifle.devices import get_optimal_device
from rifle.diffusion import AutoencoderKL, UNet2DConditionModel
from rifle.prior import BandingPriorFiLM
from rifle.vae import TiledVAE


_PHASES = ("pix", "sem")
_GROUPS = ("encoder", "decoder", "others")
_LAYERS = "down0,down1,down2,down3,mid"


def validate_checkpoint_config(sd):
    """Reject removed network variants before loading any base-model components.

    Disabled historical switches in released checkpoints remain valid metadata.
    """
    if not isinstance(sd, dict):
        raise ValueError("Expected a released RIFLE checkpoint dictionary.")
    cfg = sd.get("train_cfg", {})
    if not isinstance(cfg, dict):
        raise ValueError(
            "Expected released RIFLE train_cfg metadata to be a dictionary."
        )
    expected = {
        "prior_ctrl_mode": "film",
        "prior_channels": "0",
        "film_prior_type": "gp_band",
    }
    for key, value in expected.items():
        if key in cfg and str(cfg[key]) != value:
            raise ValueError(
                f"Unsupported {key}={cfg[key]!r}; released RIFLE requires {value!r}."
            )
    for key in (
        "use_1f2_prior",
        "use_prior_text",
        "use_prior_latent_concat",
        "use_controlnet",
        "use_detail_color_module",
        "use_frequency_preprocess",
        "tune_vae_decoder",
    ):
        if cfg.get(key, False) or sd.get(key, False):
            raise ValueError(
                f"Unsupported enabled {key}; use the released RIFLE FiLM architecture."
            )
    for key in (
        "use_prior",
        "use_prior_ctrl",
        "use_frequency_prior",
        "use_film_frequency_injection",
    ):
        if key in cfg and not cfg[key]:
            raise ValueError(
                f"Unsupported disabled {key}; released RIFLE requires banding-prior FiLM."
            )
    for key in (
        "controlnet",
        "prior_adapter",
        "prior_text_adapter",
        "detail_color_module",
        "state_dict_vae",
    ):
        if key in sd and len(sd[key]):
            raise ValueError(
                f"Unsupported {key} weights; use the released RIFLE checkpoint."
            )
    if "prior_film" in sd:
        weights = sd["prior_film"]
        if "shared.0.weight" in weights and weights["shared.0.weight"].shape[1] != 3:
            raise ValueError(
                "Unsupported prior channels; released RIFLE uses one banding-prior channel."
            )


def _adapter_name(group, phase):
    return f"default_{group}_{phase}"


def _adapter_key(group, phase):
    return f"unet_lora_{group}_modules_{phase}"


def _module_attr(group, phase):
    return (
        f"lora_unet_{group}_{phase}"
        if group == "others"
        else f"lora_unet_modules_{group}_{phase}"
    )


def _target_modules(unet):
    groups = {group: [] for group in _GROUPS}
    patterns = (
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
    )
    for name, _ in unet.named_parameters():
        if (
            "bias" in name
            or "norm" in name
            or not any(pattern in name for pattern in patterns)
        ):
            continue
        group = (
            "encoder"
            if "down_blocks" in name or "conv_in" in name
            else ("decoder" if "up_blocks" in name or "conv_out" in name else "others")
        )
        groups[group].append(name.replace(".weight", ""))
    return groups


def _add_adapters(unet, sd, phase):
    for group in _GROUPS:
        unet.add_adapter(
            LoraConfig(
                r=sd[f"lora_rank_unet_{phase}"],
                init_lora_weights="gaussian",
                target_modules=sd[_adapter_key(group, phase)],
            ),
            adapter_name=_adapter_name(group, phase),
        )


def _load_adapters(unet, sd):
    for name, parameter in unet.named_parameters():
        if "lora" in name:
            parameter.data.copy_(sd["state_dict_unet"][name])


class _PriorConditioning:
    def _init_prior(self, sd=None):
        layers = getattr(self.args, "prior_inject_layers", _LAYERS)
        if sd is not None:
            layers = sd.get("train_cfg", {}).get("prior_inject_layers", layers)
        self.prior_inject_layers = [layer for layer in str(layers).split(",") if layer]
        self.use_prior_ctrl = True
        self.prior_ctrl_mode = "film"
        self._prior_film_cond = None
        self._prior_hooks = []
        channels = list(self.unet.config.block_out_channels)
        blocks = {f"down{i}": block for i, block in enumerate(self.unet.down_blocks)}
        blocks["mid"] = self.unet.mid_block
        feature_channels = {f"down{i}": channel for i, channel in enumerate(channels)}
        feature_channels["mid"] = channels[-1]
        if any(layer not in blocks for layer in self.prior_inject_layers):
            raise ValueError(
                "Unknown FiLM injection layer in released RIFLE configuration."
            )
        self.prior_film = BandingPriorFiLM(
            in_ch=1,
            feat_ch_list=[feature_channels[x] for x in self.prior_inject_layers],
        ).to(self.device)
        if sd is not None:
            self.prior_film.load_state_dict(sd["prior_film"], strict=True)
        for index, layer in enumerate(self.prior_inject_layers):

            def hook(_module, _inputs, output, head=index):
                if self._prior_film_cond is None:
                    return output
                if isinstance(output, tuple):
                    return (
                        self.prior_film.apply(self._prior_film_cond, output[0], head),
                    ) + output[1:]
                return self.prior_film.apply(self._prior_film_cond, output, head)

            self._prior_hooks.append(blocks[layer].register_forward_hook(hook))

    def _build_prior_cond(self, prior):
        if prior is None:
            return None
        if prior.ndim != 4 or prior.shape[1] != 1:
            raise ValueError("RIFLE prior must have shape [batch, 1, height, width].")
        parameter = next(self.prior_film.parameters())
        return prior.to(device=parameter.device, dtype=parameter.dtype).clamp(0, 1)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            embeddings = [
                self.text_encoder(
                    self.tokenizer(
                        caption,
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.cat(embeddings, dim=0)


class RIFLE(_PriorConditioning, nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        validate_checkpoint_config({"train_cfg": vars(args)})
        sd = (
            torch.load(args.resume_ckpt, map_location="cpu")
            if getattr(args, "resume_ckpt", None)
            else None
        )
        if sd is not None:
            validate_checkpoint_config(sd)
        self.device = get_optimal_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_path, subfolder="text_encoder"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder="unet"
        )
        self.unet.requires_grad_(False)
        if sd is None:
            targets = _target_modules(self.unet)
            sd_adapters = {
                f"lora_rank_unet_{phase}": getattr(args, f"lora_rank_unet_{phase}")
                for phase in _PHASES
            }
            sd_adapters.update(
                {
                    _adapter_key(group, phase): targets[group]
                    for phase in _PHASES
                    for group in _GROUPS
                }
            )
        else:
            sd_adapters = sd
        for phase in _PHASES:
            setattr(
                self, f"lora_rank_unet_{phase}", sd_adapters[f"lora_rank_unet_{phase}"]
            )
            for group in _GROUPS:
                setattr(
                    self,
                    _module_attr(group, phase),
                    sd_adapters[_adapter_key(group, phase)],
                )
            _add_adapters(self.unet, sd_adapters, phase)
        if sd is not None:
            _load_adapters(self.unet, sd)
        self.unet.to(self.device)
        self.vae_fix = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder="vae"
        ).to(self.device)
        self.text_encoder.requires_grad_(False).eval()
        self.vae_fix.requires_grad_(False).eval()
        self.timesteps1 = torch.tensor([args.timesteps1], device=self.device).long()
        self._init_prior(sd)

    def _set_train_phase(self, phase):
        self.unet.train()
        for name, parameter in self.unet.named_parameters():
            parameter.requires_grad_("lora" in name and f"_{phase}." in name)
        self.prior_film.train().requires_grad_(True)

    def set_train_pix(self):
        self._set_train_phase("pix")

    def set_train_sem(self):
        self._set_train_phase("sem")

    def forward(self, c_t, c_tgt=None, batch=None, args=None):
        args = args or self.args
        batch = batch or {}
        encoded = (
            self.vae_fix.encode(c_t).latent_dist.sample()
            * self.vae_fix.config.scaling_factor
        )
        count = c_t.shape[0]
        prompt = self.encode_prompt(batch.get("prompt", [""] * count))
        negative = self.encode_prompt(batch.get("neg_prompt", [""] * count))
        null = self.encode_prompt(batch.get("null_prompt", [""] * count))
        positive = null if random.random() < args.null_text_ratio else prompt
        self._prior_film_cond = self._build_prior_cond(batch.get("prior"))
        try:
            prediction = self.unet(
                encoded,
                self.timesteps1,
                encoder_hidden_states=positive.to(torch.float32),
            ).sample
        finally:
            self._prior_film_cond = None
        denoised = encoded - prediction
        image = self.vae_fix.decode(
            denoised / self.vae_fix.config.scaling_factor
        ).sample.clamp(-1, 1)
        return image, denoised, positive, negative

    def save_model(self, outf, global_step=None):
        sd = {
            "state_dict_unet": {
                key: value
                for key, value in self.unet.state_dict().items()
                if "lora" in key
            },
            "prior_film": self.prior_film.state_dict(),
            "stripe_direction_aware": True,
            "degradation_aware": False,
            "tune_vae_decoder": False,
        }
        for phase in _PHASES:
            sd[f"lora_rank_unet_{phase}"] = getattr(self, f"lora_rank_unet_{phase}")
            for group in _GROUPS:
                sd[_adapter_key(group, phase)] = getattr(
                    self, _module_attr(group, phase)
                )
        metadata = (
            "timesteps1",
            "pix_steps",
            "null_text_ratio",
            "pos_prompt_csd",
            "neg_prompt_csd",
            "use_color_consistency_loss",
            "lambda_color_consistency",
            "lambda_color_hist",
            "color_hist_bins",
            "lora_rank_unet_pix",
            "lora_rank_unet_sem",
            "learning_rate",
            "max_train_steps",
        )
        cfg = {
            key: getattr(self.args, key) for key in metadata if hasattr(self.args, key)
        }
        cfg.update(
            prior_ctrl_mode="film",
            prior_channels="0",
            film_prior_type="gp_band",
            use_prior=True,
            use_frequency_prior=True,
            use_film_frequency_injection=True,
            prior_inject_layers=",".join(self.prior_inject_layers),
        )
        if global_step is not None:
            sd["global_step"] = cfg["global_step"] = int(global_step)
        sd["train_cfg"] = cfg
        torch.save(sd, outf)


class RIFLEInference(_PriorConditioning, nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        validate_checkpoint_config({"train_cfg": vars(args)})
        sd = torch.load(args.pretrained_path, map_location="cpu")
        validate_checkpoint_config(sd)
        self.device = get_optimal_device()
        self.weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            getattr(args, "mixed_precision", "no"), torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_path, subfolder="text_encoder"
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_path, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path, subfolder="unet"
        )
        self._init_prior(sd)
        # The released model first merges pixel adapters, then semantic adapters.
        for phase in _PHASES:
            _add_adapters(self.unet, sd, phase)
            _load_adapters(self.unet, sd)
            names = [_adapter_name(group, phase) for group in _GROUPS]
            set_weights_and_activate_adapters(self.unet, names, [1.0] * len(names))
            self.unet.merge_and_unload()
        if getattr(args, "use_vae_tiling", False):
            self._init_tiled_vae(
                args.vae_encoder_tiled_size, args.vae_decoder_tiled_size
            )
        for model in (self.vae, self.unet, self.text_encoder, self.prior_film):
            model.to(self.device, dtype=self.weight_dtype).requires_grad_(False)
        self.timesteps1 = torch.tensor([1], device=self.device).long()

    def set_eval(self):
        for model in (self.unet, self.vae, self.text_encoder, self.prior_film):
            model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, c_t, prompt="", prior=None):
        if c_t.device.type == "cuda":
            torch.cuda.synchronize(c_t.device)
        start = time.time()
        c_t = c_t.to(dtype=self.weight_dtype)
        embeddings = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded = (
            self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        )
        self._prior_film_cond = self._build_prior_cond(prior)
        try:
            prediction = self._process_latents(encoded, embeddings)
        finally:
            self._prior_film_cond = None
        image = self.vae.decode(
            (encoded - prediction) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        if c_t.device.type == "cuda":
            torch.cuda.synchronize(c_t.device)
        return time.time() - start, image

    def _process_latents(self, encoded, embeddings):
        h, w = encoded.shape[-2:]
        tile_size = getattr(self.args, "latent_tiled_size", 96)
        if (
            not getattr(self.args, "use_latent_tiling", False)
            or h * w <= tile_size * tile_size
        ):
            return self.unet(
                encoded, self.timesteps1, encoder_hidden_states=embeddings
            ).sample
        return self._predict_with_tiling(
            encoded, embeddings, tile_size, self.args.latent_tiled_overlap
        )

    def _predict_with_tiling(self, encoded, embeddings, tile_size, overlap):
        _, _, h, w = encoded.shape
        tile_size = min(tile_size, h, w)
        stride = max(tile_size - overlap, 1)
        ys = list(range(0, max(h - tile_size, 0) + 1, stride))
        xs = list(range(0, max(w - tile_size, 0) + 1, stride))
        if ys[-1] != h - tile_size:
            ys.append(h - tile_size)
        if xs[-1] != w - tile_size:
            xs.append(w - tile_size)
        prediction = torch.zeros_like(encoded)
        contributors = torch.zeros_like(encoded)
        full_prior = self._prior_film_cond
        try:
            for y in ys:
                for x in xs:
                    y2, x2 = y + tile_size, x + tile_size
                    self._prior_film_cond = self._crop_prior_cond_for_tile(
                        full_prior, y, y2, x, x2, h, w
                    )
                    output = self.unet(
                        encoded[:, :, y:y2, x:x2],
                        self.timesteps1,
                        encoder_hidden_states=embeddings,
                    ).sample
                    weights = self._gaussian_weights(
                        tile_size,
                        tile_size,
                        1,
                        channels=output.shape[1],
                        dtype=output.dtype,
                    )
                    prediction[:, :, y:y2, x:x2] += output * weights
                    contributors[:, :, y:y2, x:x2] += weights
        finally:
            self._prior_film_cond = full_prior
        return prediction / contributors.clamp_min(1e-6)

    def _crop_prior_cond_for_tile(self, prior_cond, y0, y1, x0, x1, latent_h, latent_w):
        """Map a latent tile back to the corresponding prior-image crop for local FiLM conditioning."""
        if prior_cond is None:
            return None

        _, _, prior_h, prior_w = prior_cond.shape
        scale_y = prior_h / max(latent_h, 1)
        scale_x = prior_w / max(latent_w, 1)

        py0 = max(0, min(prior_h - 1, int(round(y0 * scale_y))))
        py1 = max(py0 + 1, min(prior_h, int(round(y1 * scale_y))))
        px0 = max(0, min(prior_w - 1, int(round(x0 * scale_x))))
        px1 = max(px0 + 1, min(prior_w, int(round(x1 * scale_x))))
        return prior_cond[:, :, py0:py1, px0:px1]

    def _gaussian_weights(
        self, tile_width, tile_height, nbatches, channels=None, dtype=None
    ):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [
            exp(-((x - midpoint_x) ** 2) / (2 * (tile_width**2) * 0.01))
            / sqrt(2 * pi * 0.01)
            for x in range(tile_width)
        ]
        y_probs = [
            exp(-((y - midpoint_y) ** 2) / (2 * (tile_height**2) * 0.01))
            / sqrt(2 * pi * 0.01)
            for y in range(tile_height)
        ]

        weights = np.outer(y_probs, x_probs)
        ch = self.unet.config.in_channels if channels is None else channels
        out = torch.tensor(weights, device=self.device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out.repeat(nbatches, ch, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256):
        for net, size, decoder in (
            (self.vae.encoder, encoder_tile_size, False),
            (self.vae.decoder, decoder_tile_size, True),
        ):
            if not hasattr(net, "original_forward"):
                net.original_forward = net.forward
            net.forward = TiledVAE(net, size, is_decoder=decoder, to_gpu=True)


class DistillationLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.sched = DDPMScheduler.from_pretrained(
            args.pretrained_model_path_csd, subfolder="scheduler"
        )
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif accelerator.device.type == "cpu":
            weight_dtype = torch.float32
        self.weight_dtype = weight_dtype

        self.unet_fix = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_path_csd, subfolder="unet"
        )

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available, please install it by running `pip install xformers`"
                )

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        noise_pred = model(
            latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(
            self.sched.config.num_train_timesteps * args.min_dm_step_ratio
        )
        max_dm_step = int(
            self.sched.config.num_train_timesteps * args.max_dm_step_ratio
        )

        timestep = torch.randint(
            min_dm_step, max_dm_step, (bsz,), device=latents.device
        ).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=self.weight_dtype),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (
                noise_pred_text - noise_pred_uncond
            )
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(
                self.sched, noise_pred, noisy_latents, timestep
            )
            pred_fake_latents = self.eps_to_mu(
                self.sched, noise_pred_uncond, noisy_latents, timestep
            )

        weighting_factor = torch.abs(latents - pred_real_latents).mean(
            dim=[1, 2, 3], keepdim=True
        )

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()
