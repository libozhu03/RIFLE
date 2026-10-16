"""Portable release entrypoints; no models, GPU or external data required."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rifle.pretrained import resolve_pretrained_model_path

ROOT = Path(__file__).resolve().parents[1]


def make_sd21(root):
    for name in ("tokenizer", "scheduler", "text_encoder", "unet", "vae"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "text_encoder/model.safetensors").touch()
    for name in ("unet", "vae"):
        (root / name / "diffusion_pytorch_model.safetensors").touch()
    return root


class ModelPathTests(unittest.TestCase):
    def test_explicit_valid_path_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            explicit = make_sd21(Path(temp) / "explicit")
            fallback = make_sd21(Path(temp) / "fallback")
            with patch.dict(os.environ, {"PRETRAINED_MODEL_PATH": str(fallback)}):
                self.assertEqual(
                    resolve_pretrained_model_path(str(explicit)), str(explicit)
                )

    def test_default_sd21_path_uses_exported_model(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_sd21(Path(temp) / "base-model")
            with patch.dict(os.environ, {"PRETRAINED_MODEL_PATH": str(model)}):
                self.assertEqual(
                    resolve_pretrained_model_path("preset/models/sd21"), str(model)
                )

    def test_unset_path_uses_exported_model(self):
        with tempfile.TemporaryDirectory() as temp:
            model = make_sd21(Path(temp) / "base-model")
            with patch.dict(os.environ, {"PRETRAINED_MODEL_PATH": str(model)}):
                self.assertEqual(resolve_pretrained_model_path(None), str(model))

    def test_custom_missing_path_is_preserved(self):
        self.assertEqual(resolve_pretrained_model_path("custom-model"), "custom-model")


class ShellWrapperTests(unittest.TestCase):
    def run_wrapper(self, script, args, overrides=None):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            logger = temp / "python"
            logger.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "with open(os.environ['RIFLE_TEST_LOG'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            logger.chmod(0o755)
            env = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "INFER_BIN",
                    "PYTHON_BIN",
                    "PRETRAINED_MODEL_PATH",
                }
            }
            env.update(
                PATH=str(temp) + os.pathsep + env.get("PATH", ""),
                RIFLE_TEST_LOG=str(temp / "calls.jsonl"),
            )
            if overrides:
                env.update(
                    {
                        k: str(logger) if v == "LOGGER" else v
                        for k, v in overrides.items()
                    }
                )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / script), *args],
                cwd=temp,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return [
                json.loads(line)
                for line in (temp / "calls.jsonl").read_text().splitlines()
            ]

    def test_segmented_wrapper_uses_active_python(self):
        calls = self.run_wrapper("infer.sh", ["weight.pkl", "lq", "masks", "out"])
        self.assertEqual(calls[0][0], "infer.py")
        self.assertEqual(
            calls[0][calls[0].index("--pretrained_model_path") + 1],
            "preset/models/sd21",
        )

    def test_inference_wrapper_forwards_python_and_model_path(self):
        calls = self.run_wrapper(
            "infer.sh",
            ["weight.pkl", "lq", "masks", "out"],
            {"PYTHON_BIN": "LOGGER", "PRETRAINED_MODEL_PATH": "/tmp/custom sd21"},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][calls[0].index("--pretrained_model_path") + 1], "/tmp/custom sd21"
        )


if __name__ == "__main__":
    unittest.main()
