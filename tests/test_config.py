"""The release has one training configuration and a small explicit CLI."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest


class ConfigTests(unittest.TestCase):
    def test_reference_config_and_cli_override(self):
        from rifle.config import parse_train_args

        args = parse_train_args(
            ["--config", "configs/train.yaml", "--max_train_steps", "12"]
        )
        self.assertEqual(args.max_train_steps, 12)
        self.assertEqual(args.data_config, "configs/data.yaml")
        self.assertEqual(args.lora_rank_unet_pix, 32)
        self.assertEqual(args.mask_loss_lambda, 0.16)

    def test_unknown_yaml_key_is_rejected(self):
        from rifle.config import parse_train_args

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.yaml"
            path.write_text("controlnet_model: historical\n")
            with self.assertRaisesRegex(ValueError, "Unknown.*controlnet_model"):
                parse_train_args(["--config", str(path)])

    def test_historical_cli_mode_is_rejected(self):
        from rifle.config import parse_train_args

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train_args(["--train_mode", "wavelet_only"])


if __name__ == "__main__":
    unittest.main()
