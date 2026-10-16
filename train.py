"""Train the released RIFLE model with masked reconstruction and semantic refinement."""

from rifle.model import RIFLE, DistillationLoss
from rifle.data import ValidationDataset, build_training_dataset
from rifle.pretrained import resolve_pretrained_model_path
from rifle.config import parse_train_args
import os
import gc
import json
import re
from pathlib import Path
import sys
import lpips
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_val_loaders(args):
    loaders = []
    for item in args.val_sets:
        dataset = ValidationDataset(item["lq_dir"], item["gt_dir"], args)
        options = dict(
            batch_size=1,
            shuffle=False,
            num_workers=args.eval_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if args.eval_num_workers > 0:
            options.update(
                persistent_workers=True,
                prefetch_factor=max(2, args.train_prefetch_factor),
            )
        loaders.append((item["name"], torch.utils.data.DataLoader(dataset, **options)))
    return loaders


def _sanitize_tracker_config(values):
    sanitized = {}
    for key, value in values.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = "None"
        else:
            sanitized[key] = json.dumps(
                value, ensure_ascii=False, sort_keys=True, default=str
            )
    return sanitized


def _tensor_to_pil(x):
    return transforms.ToPILImage()(_to_01(x.detach().cpu()))


def _save_eval_sample(base_dir, base_name, pred, lq=None, gt=None, save_lq_gt=True):
    pred_dir = os.path.join(base_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    _tensor_to_pil(pred).save(os.path.join(pred_dir, base_name))
    if not save_lq_gt:
        return
    lq_dir = os.path.join(base_dir, "lq")
    gt_dir = os.path.join(base_dir, "gt")
    os.makedirs(lq_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    if lq is not None:
        _tensor_to_pil(lq).save(os.path.join(lq_dir, base_name))
    if gt is not None:
        _tensor_to_pil(gt).save(os.path.join(gt_dir, base_name))


def _evaluate_round(model, accelerator, val_loaders, args, prompt, eval_dir):
    eval_logs = {}
    save_limit = int(getattr(args, "eval_save_limit", 0))
    model = accelerator.unwrap_model(model)
    model.eval()
    with torch.no_grad():
        for val_name, dl_val in val_loaders:
            set_dir = os.path.join(eval_dir, val_name)
            os.makedirs(set_dir, exist_ok=True)
            count = 0
            mse_sum = 0.0
            for batch_idx, batch_val in enumerate(dl_val):
                x_src = batch_val["conditioning_pixel_values"].to(accelerator.device)
                x_tgt = batch_val["output_pixel_values"].to(accelerator.device)
                x_basename = batch_val["base_name"][0]
                batch_val["prompt"] = [prompt]
                x_tgt_pred, _, _, _ = model(x_src, x_tgt, batch=batch_val, args=args)
                pred_01 = _to_01(x_tgt_pred.float())
                gt_01 = _to_01(x_tgt.float())
                mse = F.mse_loss(pred_01, gt_01, reduction="mean").item()
                mse_sum += mse
                count += 1
                if save_limit <= 0 or batch_idx < save_limit:
                    _save_eval_sample(
                        set_dir,
                        x_basename,
                        x_tgt_pred[0],
                        lq=x_src[0],
                        gt=x_tgt[0],
                        save_lq_gt=bool(getattr(args, "save_eval_lq_gt", True)),
                    )
                del x_src, x_tgt, x_tgt_pred, pred_01, gt_01, batch_val
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_mse = mse_sum / max(count, 1)
            avg_psnr = -10.0 * torch.log10(torch.tensor(max(avg_mse, 1e-12))).item()
            eval_logs[f"val_{val_name}_psnr"] = avg_psnr
            eval_logs[f"val_{val_name}_count"] = count
    model.train()
    return eval_logs


def _to_01(x):
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def _collate_optional_mask(batch):
    has_mask = any(("mask_pixel_values" in sample for sample in batch))
    if not has_mask:
        return default_collate(batch)
    for sample in batch:
        if "mask_pixel_values" not in sample:
            h, w = sample["conditioning_pixel_values"].shape[-2:]
            sample["mask_pixel_values"] = torch.zeros(
                (1, h, w), dtype=sample["conditioning_pixel_values"].dtype
            )
    return default_collate(batch)


def _should_trigger_periodic(step: int, period: int) -> bool:
    if period is None or int(period) <= 0:
        return False
    return step > 0 and step % int(period) == 0


def _infer_resume_global_step(resume_ckpt: str) -> int:
    ckpt_path = Path(resume_ckpt)
    matched = re.search("model_(\\d+)\\.pkl$", ckpt_path.name)
    if matched:
        return int(matched.group(1))
    if ckpt_path.name == "model_latest.pkl" and ckpt_path.parent.is_dir():
        numbered = []
        for cand in ckpt_path.parent.glob("model_*.pkl"):
            m = re.search("model_(\\d+)\\.pkl$", cand.name)
            if m:
                numbered.append(int(m.group(1)))
        if numbered:
            return max(numbered)
    return 0


def _rgb_to_ycbcr_torch(rgb):
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y, cb, cr], dim=1)


def _soft_histogram(x, bins=32):
    b = x.shape[0]
    x = x.reshape(b, -1, 1).clamp(0.0, 1.0)
    if x.shape[1] > 65536:
        step = max(1, x.shape[1] // 65536)
        x = x[:, ::step, :]
    nb = max(8, int(bins))
    centers = torch.linspace(0.0, 1.0, nb, device=x.device, dtype=x.dtype).view(
        1, 1, nb
    )
    delta = 1.0 / max(nb - 1, 1)
    weights = torch.relu(1.0 - torch.abs(x - centers) / (delta + 1e-08))
    hist = weights.sum(dim=1)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-06)
    return hist


def color_consistency_loss(pred, gt, src, bins=32):
    pred_ycc = _rgb_to_ycbcr_torch(_to_01(pred.float()))
    gt_ycc = _rgb_to_ycbcr_torch(_to_01(gt.float()))
    src_ycc = _rgb_to_ycbcr_torch(_to_01(src.float()))
    pred_cbcr = pred_ycc[:, 1:3]
    gt_cbcr = gt_ycc[:, 1:3]
    src_cbcr = src_ycc[:, 1:3]
    loss_chroma = F.l1_loss(pred_cbcr, gt_cbcr)
    pred_stat = torch.cat(
        [pred_cbcr.mean(dim=[2, 3]), pred_cbcr.std(dim=[2, 3])], dim=1
    )
    src_stat = torch.cat([src_cbcr.mean(dim=[2, 3]), src_cbcr.std(dim=[2, 3])], dim=1)
    loss_stat = F.smooth_l1_loss(pred_stat, src_stat)
    hist_loss = torch.tensor(0.0, device=pred.device)
    for ch in range(2):
        h_pred = _soft_histogram(pred_cbcr[:, ch : ch + 1], bins=bins)
        h_gt = _soft_histogram(gt_cbcr[:, ch : ch + 1], bins=bins)
        h_src = _soft_histogram(src_cbcr[:, ch : ch + 1], bins=bins)
        hist_loss = (
            hist_loss + 0.5 * F.l1_loss(h_pred, h_gt) + 0.5 * F.l1_loss(h_pred, h_src)
        )
    hist_loss = hist_loss / 2.0
    return (loss_chroma + 0.5 * loss_stat, hist_loss)


def masked_recon_loss(pred, target, mask):
    pred = pred.float()
    target = target.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask[:, :1].float().clamp(0.0, 1.0)
    per_pixel = (pred - target).abs().mean(dim=1, keepdim=True)
    masked = per_pixel * mask
    mask_sum = mask.sum(dim=[1, 2, 3])
    masked_denom = mask_sum.clamp_min(1.0)
    masked_loss = masked.sum(dim=[1, 2, 3]) / masked_denom
    global_loss = per_pixel.mean(dim=[1, 2, 3])
    use_global = (mask_sum <= 0.5).float()
    return (masked_loss * (1.0 - use_global) + global_loss * use_global).mean()


def _build_optimizer_params(model, args):
    layers_to_opt = []
    for name, p in model.unet.named_parameters():
        if "lora" in name:
            p.requires_grad_(True)
            layers_to_opt.append(p)
    if hasattr(model, "prior_film"):
        for _, p in model.prior_film.named_parameters():
            p.requires_grad_(True)
            layers_to_opt.append(p)
    return list({id(p): p for p in layers_to_opt}.values())


def _save_checkpoint(model, accelerator, output_dir: str, global_step: int):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model = accelerator.unwrap_model(model)
    model.save_model(
        os.path.join(ckpt_dir, f"model_{global_step}.pkl"), global_step=global_step
    )
    model.save_model(
        os.path.join(ckpt_dir, "model_latest.pkl"), global_step=global_step
    )


def _set_training_phase(network, step, pix_steps):
    model = network.module if hasattr(network, "module") else network
    adapters = ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
    if step >= pix_steps:
        adapters += ["default_encoder_sem", "default_decoder_sem", "default_others_sem"]
        model.unet.set_adapter(adapters)
        model.set_train_sem()
    else:
        model.unet.set_adapter(adapters)
        model.set_train_pix()


def main(args):
    print(
        f"RIFLE training: {args.resolution_ori}px, FiLM prior + masked reconstruction"
    )
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    model = RIFLE(args)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    use_csd = float(args.lambda_csd) > 0
    net_csd = None
    use_lpips = float(args.lambda_lpips) > 0
    net_lpips = None
    model.unet.set_adapter(
        ["default_encoder_pix", "default_decoder_pix", "default_others_pix"]
    )
    model.set_train_pix()
    opt_params = _build_optimizer_params(model, args)
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    train_sampler = None
    dataset_train, train_sampler = build_training_dataset(args.data_config, args)
    val_loaders = _build_val_loaders(args)
    dl_train_kwargs = dict(
        batch_size=args.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_optional_mask,
    )
    if args.dataloader_num_workers > 0:
        dl_train_kwargs["persistent_workers"] = bool(
            getattr(args, "train_persistent_workers", True)
        )
        dl_train_kwargs["prefetch_factor"] = max(
            2, int(getattr(args, "train_prefetch_factor", 2))
        )
    dl_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    model, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        model, optimizer, dl_train, lr_scheduler
    )
    if accelerator.is_main_process:
        tracker_config = _sanitize_tracker_config(dict(vars(args)))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    resume_global_step = 0
    if args.resume_ckpt is not None:
        resume_global_step = _infer_resume_global_step(args.resume_ckpt)
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = resume_global_step
    _set_training_phase(model, global_step, args.pix_steps)
    semantic_phase_active = False
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0.0
    lambda_csd = 0.0
    while global_step < args.max_train_steps:
        for _step, batch in enumerate(dl_train):
            with accelerator.accumulate(model):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                bsz = x_tgt.shape[0]
                prompt = args.pos_prompt_csd if args.pos_prompt_csd is not None else ""
                batch["prompt"] = [prompt for _ in range(bsz)]
                if global_step >= args.pix_steps and (not semantic_phase_active):
                    _set_training_phase(model, global_step, args.pix_steps)
                    semantic_phase_active = True
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd = args.lambda_csd
                    if use_lpips and net_lpips is None:
                        net_lpips = lpips.LPIPS(net="vgg").to(accelerator.device)
                        net_lpips.requires_grad_(False)
                        net_lpips.eval()
                    if use_csd and net_csd is None:
                        net_csd = DistillationLoss(args=args, accelerator=accelerator)
                        net_csd.requires_grad_(False)
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = model(
                    x_src, x_tgt, batch=batch, args=args
                )
                use_full_loss = global_step >= args.pix_steps
                loss_l2 = (
                    F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                    * lambda_l2
                )
                if use_full_loss and net_lpips is not None:
                    loss_lpips = (
                        net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                        * lambda_lpips
                    )
                    loss = loss_l2 + loss_lpips
                else:
                    loss_lpips = torch.tensor(0.0, device=loss_l2.device)
                    loss = loss_l2
                loss_mask = torch.tensor(0.0, device=loss.device)
                if "mask_pixel_values" in batch:
                    loss_mask = masked_recon_loss(
                        x_tgt_pred.float(),
                        x_tgt.float(),
                        batch["mask_pixel_values"].to(x_tgt.device),
                    ) * getattr(args, "mask_loss_lambda", 0.2)
                    loss = loss + loss_mask
                loss_color_consistency = torch.tensor(0.0, device=loss.device)
                loss_color_hist = torch.tensor(0.0, device=loss.device)
                if use_full_loss:
                    loss_color_consistency, loss_color_hist = color_consistency_loss(
                        x_tgt_pred.float(),
                        x_tgt.float(),
                        x_src.float(),
                        bins=getattr(args, "color_hist_bins", 32),
                    )
                    loss_color_consistency = loss_color_consistency * getattr(
                        args, "lambda_color_consistency", 0.06
                    )
                    loss_color_hist = loss_color_hist * getattr(
                        args, "lambda_color_hist", 0.04
                    )
                    loss = loss + loss_color_consistency + loss_color_hist
                if net_csd is not None:
                    loss_csd = (
                        net_csd.cal_csd(
                            latents_pred, prompt_embeds, neg_prompt_embeds, args
                        )
                        * lambda_csd
                    )
                else:
                    loss_csd = torch.tensor(0.0, device=loss.device)
                loss = loss + loss_csd
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(opt_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {
                        "loss_csd": loss_csd.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                    }
                    logs["loss_mask"] = loss_mask.detach().item()
                    logs["loss_color_consistency"] = (
                        loss_color_consistency.detach().item()
                    )
                    logs["loss_color_hist"] = loss_color_hist.detach().item()
                    progress_bar.set_postfix(**logs)
                    should_save_ckpt = (
                        _should_trigger_periodic(global_step, args.checkpointing_steps)
                        or global_step >= args.max_train_steps
                    )
                    if should_save_ckpt:
                        _save_checkpoint(
                            model, accelerator, args.output_dir, global_step
                        )
                    if _should_trigger_periodic(global_step, args.eval_freq):
                        del (
                            loss,
                            loss_csd,
                            loss_l2,
                            loss_lpips,
                            x_src,
                            x_tgt,
                            x_tgt_pred,
                            latents_pred,
                            prompt_embeds,
                            neg_prompt_embeds,
                            batch,
                        )
                        del loss_mask
                        del loss_color_consistency, loss_color_hist
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        eval_dir = os.path.join(
                            args.output_dir, "eval", f"fid_{global_step}"
                        )
                        os.makedirs(eval_dir, exist_ok=True)
                        eval_logs = _evaluate_round(
                            model, accelerator, val_loaders, args, prompt, eval_dir
                        )
                        logs.update(eval_logs)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)
                    accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_train_args()
    args.pretrained_model_path = resolve_pretrained_model_path(
        args.pretrained_model_path, cwd=os.path.dirname(__file__)
    )
    args.pretrained_model_path_csd = resolve_pretrained_model_path(
        args.pretrained_model_path_csd, cwd=os.path.dirname(__file__)
    )
    main(args)
