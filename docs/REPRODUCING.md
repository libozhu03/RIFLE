# Reproducing RIFLE

## Released implementation

The source has one training/restoration path: a single-channel GP banding prior,
FiLM injection, masked reconstruction and pixel/semantic LoRA refinement.
There is one simulator, one prior implementation and one tiled VAE implementation.
Native Diffusers backbones are extended only where RIFLE needs adapter merging
and VAE dtype conversion.

Historical ControlNet, latent concatenation, text priors, detail-color modules,
wavelet/SSIM ablations, ISP simulation and adjustable pixel/semantic inference
modes are not included. The checkpoint loader accepts the published FiLM model
and explicitly rejects unsupported historical variants. Its tensor keys and
adapter names remain unchanged.

## Data layout

**Dataset release: TBD.** The paths below describe the expected local layout.

```text
data/
  LSDIR/HR/
  UHDM/gt/
  validation/lq/
  validation/gt/
  testset/lq/
  testset/gt/
  testset/masks/
```

Edit HQ folders in [configs/data.yaml](../configs/data.yaml) and validation
folders in [configs/train.yaml](../configs/train.yaml). The four source sampling
ratios remain 0.45/0.15 for LSDIR focus/screen profiles and 0.25/0.15 for UHDM.
Simulation and stripe-severity supervision are generated online.

Validation images must have matching visible filenames in LQ and GT. Inference
images and masks are paired by filename stem, for example `Banding01_1.png`.
Use aligned original binary masks; inference uses threshold 127 on uint8.
Hidden image files, including macOS `._*` metadata, are ignored. Training and
simulation also exclude hidden directories.

## Inference modes

Without `--mask_dir`, `infer.py` restores the entire image. With `--mask_dir`,
it restores the screen region and preserves the pixels outside the aligned
binary screen mask. See the [two inference examples](../README.md#-inference).

## Simulation

The simulator is available independently of training. See the
[simulation command and output layout](../README.md#simulation).

## Training and resume

Download the SD2.1 base model and RIFLE checkpoint from the
[model download links](../weights/README.md#downloads).

Set `PRETRAINED_MODEL_PATH` to a local SD2.1 base directory, and prepare the HQ and
validation images before training. The default output folder is `runs/RIFLE`.
All hyperparameters are in one flat YAML file; CLI flags cover common overrides.

```bash
python train.py --config configs/train.yaml
python train.py --config configs/train.yaml \
  --resume_ckpt runs/RIFLE/checkpoints/model_4000.pkl
```

Resume selects the pixel/semantic phase from the checkpoint step and preserves
the configured warmup boundary. For Accelerate-based distributed launching:

```bash
accelerate launch train.py --config configs/train.yaml
```

## Reference checkpoint

The original source selected `model_4000.pkl` from the screenfocus line before
its full 50,000-step sweep completed. The current checkpoint content is preserved;
the public filename is simplified to `RIFLE.pkl`.

```bash
bash scripts/infer.sh \
  weights/RIFLE.pkl data/testset/lq data/testset/masks \
  results/reference_predictions
```

## Local verification

```bash
python -m unittest discover -s tests -v
python train.py --help
python infer.py --help
python simulate.py --help
```

Regression tests use small local components, without downloading SD2.1. They cover
checkpoint compatibility, numerical parity, crop/mask/prior alignment, training
phase resume and image compositing. Full CUDA restoration/training and
convergence must still be validated on the intended hardware before publishing
new reproduction claims.
