import os
from pathlib import Path


def _has_any_file(path: Path, filenames):
    return any((path / name).is_file() for name in filenames)


def _is_valid_sd21_root(path_str: str | None) -> bool:
    if not path_str:
        return False
    path = Path(path_str).expanduser()
    if not path.is_dir():
        return False
    if not (path / "tokenizer").is_dir():
        return False
    if not (path / "scheduler").is_dir():
        return False
    if not _has_any_file(
        path / "text_encoder", ["model.safetensors", "pytorch_model.bin"]
    ):
        return False
    if not _has_any_file(
        path / "unet",
        ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
    ):
        return False
    if not _has_any_file(
        path / "vae",
        ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"],
    ):
        return False
    return True


def resolve_pretrained_model_path(
    path_str: str | None, *, cwd: str | None = None
) -> str | None:
    path_str = path_str or os.environ.get("PRETRAINED_MODEL_PATH")
    if not path_str:
        return path_str
    raw = Path(path_str).expanduser()
    candidates = [raw]
    if cwd:
        candidates.append((Path(cwd) / raw).resolve())
    for candidate in candidates:
        if _is_valid_sd21_root(str(candidate)):
            return str(candidate)

    path_name = raw.name.lower()
    if "sd21" not in path_name:
        return path_str

    for fallback in (os.environ.get("PRETRAINED_MODEL_PATH"),):
        if _is_valid_sd21_root(fallback):
            return fallback
    return path_str
