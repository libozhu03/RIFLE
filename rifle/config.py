"""Load the single released training recipe with a small set of CLI overrides."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs/train.yaml"


def _read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return config


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the released RIFLE model.")
    parser.add_argument(
        "--config", default="configs/train.yaml", help="Training YAML configuration."
    )
    parser.add_argument(
        "--output_dir", help="Override the experiment output directory."
    )
    parser.add_argument("--resume_ckpt", help="Resume from a RIFLE checkpoint.")
    parser.add_argument(
        "--max_train_steps", type=int, help="Override the number of optimizer steps."
    )
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int)
    parser.add_argument("--seed", type=int)
    overrides = vars(parser.parse_args(argv))

    config = _read_config(REFERENCE_CONFIG)
    custom_path = Path(overrides["config"])
    if custom_path.resolve() != REFERENCE_CONFIG:
        custom = _read_config(custom_path)
        unknown = sorted(set(custom) - set(config))
        if unknown:
            raise ValueError(
                f"Unknown training configuration keys: {', '.join(unknown)}"
            )
        config.update(custom)
    config.update({key: value for key, value in overrides.items() if value is not None})
    for key in (
        "max_train_steps",
        "resolution",
        "train_batch_size",
        "gradient_accumulation_steps",
    ):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    # The backbone and teacher share the same local SD2.1 model.
    config["pretrained_model_path_csd"] = config["pretrained_model_path"]
    config["resolution_ori"] = config["resolution_tgt"] = config.pop("resolution")
    config["pos_prompt_csd"] = config.pop("prompt")
    config["neg_prompt_csd"] = config.pop("negative_prompt")
    config.setdefault("resume_ckpt", None)
    return SimpleNamespace(**config)
