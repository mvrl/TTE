# Training

```bash
python train/train.py --config train/config.yaml --devices 2
```

The frozen [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) MAE ViT-L/16 weights download automatically on first run. Sites are initialized from the shipped land-concentrated lattice `sites/sites_4096.npz`.

## Data

We train on the SatCLIP **S2-100K** dataset — [`davanstrien/satclip`](https://huggingface.co/datasets/davanstrien/satclip): 100K 13-band Sentinel-2 patches with `lon`/`lat`. It ships as GeoTIFFs, so preprocess it into a HuggingFace Arrow dataset (`datasets.save_to_disk`) of `{image, lon, lat}` records — `image` a 13-band reflectance tensor — and set `data.hf_dataset_path` in `config.yaml`. Set `data.preprocessed: true` if the imagery is already reflectance-normalized (otherwise it is divided by 10000).

## Files

- `train.py` — entry point (reads `config.yaml`, builds the model, runs Lightning).
- `pretrain.py` — `TTEPretrainModel`, the contrastive image-location model (training only).
- `lightning_module.py` — training/optimization loop.
- `losses.py` — contrastive + reconstruction + alignment.
- `datamodule.py` — the Sentinel-2 HuggingFace dataset.
- `config.yaml` — the recipe.

Checkpoints (every improving epoch + `last.ckpt`) are written to `paths.output_dir`. Load one with `tte.load_tte_model("path/to/epoch_XXX....ckpt")`.
