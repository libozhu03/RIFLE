# RIFLE checkpoint

## Downloads

| Model | Download | Extraction code | Local path |
| --- | --- | --- | --- |
| Stable Diffusion 2.1 base (`sd21.zip`) | [Baidu Netdisk](https://pan.baidu.com/s/15GuxzqXme4yycgWdqsen8g?pwd=ca6v) | `ca6v` | `preset/models/sd21/` |
| RIFLE checkpoint (`RIFLE.pkl`) | [Baidu Netdisk](https://pan.baidu.com/s/1EIUj_0phH5XWK6w8_7xAhw?pwd=w99g) | `w99g` | `weights/RIFLE.pkl` |

Download both assets. Extract `sd21.zip` and place the SD2.1 model directory at
`preset/models/sd21/`,
with `tokenizer/`, `text_encoder/`, `scheduler/`, `unet/` and `vae/` directly inside.
Place `RIFLE.pkl` in `weights/`. To use a different SD2.1 location:

```bash
export PRETRAINED_MODEL_PATH=/path/to/sd21
```

## Checkpoint verification

Place the published checkpoint at `weights/RIFLE.pkl`. Its expected SHA256 is:

```text
be938afbdd5ea221bf4af640158d42c0024e259f52b71e88e4a947cb39b959e6
```

The checkpoint contents are unchanged; previous filenames work when passed
explicitly with `--pretrained_path`. The SD2.1 base model is a separate asset,
selected using `PRETRAINED_MODEL_PATH` or `--pretrained_model_path`.

If you have the three original split files, name them
`RIFLE.pkl.part-000`, `RIFLE.pkl.part-001`, `RIFLE.pkl.part-002`, place them in this
folder and run:

```bash
bash scripts/assemble_weights.sh
```

Model binaries are ignored by Git.
