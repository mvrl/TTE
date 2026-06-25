# Evaluation

The RANGE evaluation data follows the [RANGE benchmark](https://github.com/mvrl/RANGE).

> Constants are kept identical to the original RANGE protocol (`RANDOM_SEED=42`,
> ridge alphas `(0.1, 1.0, 10.0)`, 80/20 split).

## Data

| Script | Data | Provide via |
| --- | --- | --- |
| `eval_range.py` | RANGE benchmark eval set | `--data_dir` or `RANGE_EVAL_DATA` |
| `eval_inat.py` | iNaturalist-2018 NPZs (`inat2018_train.npz`, `inat2018_val.npz`: `lon, lat, classes`) | `--data_dir` or `INAT_DATA` |
| `range.py` | a HuggingFace image+coords dataset | `--data_dir` (required) |



## Run

```bash
# Geospatial benchmark (classification + regression). Defaults to the mean over 5
# random splits (seeds 42-46), matching paper Table 1; use --n_seeds 1 for the
# original single fixed-split eval (~0.3-1.5 pts lower).
python eval/eval_range.py --model_path MVRL/TTE --data_dir /path/to/range_eval_data --device cuda:0

# iNaturalist-2018 species classification (location prior + image classifier)
python eval/eval_inat.py  --model_path MVRL/TTE --data_dir /path/to/inat --device cuda:0

# RANGE-style retrieval augmentation: build a DB, then evaluate
python eval/range.py generate --model_path MVRL/TTE --data_dir /path/to/hf_dataset
python eval/range.py evaluate --model_path MVRL/TTE
```

`--model_path` accepts the HuggingFace id `MVRL/TTE` or a local `.ckpt`/`.pt`.
Results are written under `outputs/eval/<model_name>/`.
