#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `import tte`
sys.path.insert(0, str(Path(__file__).resolve().parent))          # train/ for siblings

from datamodule import HFMultispectralDataModule
from lightning_module import TTELightningModule
from pretrain import TTEPretrainModel


class SaveOnImprovement(Callback):
    """Save a checkpoint each time val/loss improves (keeps every improving epoch)."""

    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.best = float("inf")

    def on_validation_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get("val/loss")
        if val is not None and val.item() < self.best:
            self.best = val.item()
            self.dirpath.mkdir(parents=True, exist_ok=True)
            path = self.dirpath / f"epoch_{trainer.current_epoch:03d}_val_loss_{self.best:.4f}.ckpt"
            trainer.save_checkpoint(str(path))
            print(f"  ✓ new best val/loss={self.best:.4f} -> {path.name}")


def parse_args():
    p = argparse.ArgumentParser(description="Train TTE (contrastive image-location pretraining).")
    p.add_argument("--config", default="train/config.yaml")
    p.add_argument("--devices", default="1", help="num GPUs or comma-separated ids, e.g. '2' or '0,1'")
    p.add_argument("--precision", default="32", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--accumulate_grad_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="override per-GPU batch size")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    m, d, t, o, l = cfg["model"], cfg["data"], cfg["training"], cfg["optimizer"], cfg["loss"]
    out_dir = cfg.get("paths", {}).get("output_dir", "./checkpoints/tte")

    devices = [int(x) for x in args.devices.split(",")] if "," in args.devices else int(args.devices)
    n_gpus = len(devices) if isinstance(devices, list) else devices
    strategy = DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto"

    init_sites = None
    if m.get("init_sites"):
        init_sites = torch.from_numpy(np.load(m["init_sites"])["sites"].astype("float32"))

    model = TTEPretrainModel(
        output_dim=m["output_dim"],
        image_resolution=m.get("image_resolution", 224),
        n_sites=m["n_sites"],
        tau_init=m["tau_init"],
        site_embed_dim=m["site_embed_dim"],
        hidden_dim=m["hidden_dim"],
        n_reslayers=m["n_reslayers"],
        dropout=m.get("dropout", 0.5),
        init_sites=init_sites,
        n_registers=m["n_registers"],
        register_temperature=m["register_temperature"],
        img_temperature=m.get("img_temperature", 0.05),
        register_fixed_gate=m["register_fixed_gate"],
        register_dropout=m.get("register_dropout", 0.1),
        temperature_anneal_end=m.get("temperature_anneal_end", 0.2),
        use_cosine_attention=m.get("use_cosine_attention", False),
    )

    lit = TTELightningModule(
        model=model,
        lambda_reconstruction=l["lambda_reconstruction"],
        lambda_alignment=l["lambda_alignment"],
        lr_location=o["lr_location"],
        lr_tau=o.get("lr_tau"),
        lr_directions=o.get("lr_directions"),
        lr_registers=o.get("lr_registers"),
        lr_image_proj=o["lr_image_proj"],
        lr_logit_scale=o["lr_logit_scale"],
        weight_decay=o["weight_decay"],
        max_epochs=t["max_epochs"],
        warmup_epochs=o.get("warmup_epochs", 0),
    )

    data = HFMultispectralDataModule(
        dataset_path=d["hf_dataset_path"],
        batch_size=args.batch_size or t["batch_size"],
        num_workers=d.get("num_workers", 8),
        crop_size=d.get("crop_size", 224),
        val_split=d.get("val_split", 0.1),
        preprocessed=d.get("preprocessed", False),
    )

    trainer = pl.Trainer(
        max_epochs=t["max_epochs"],
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches or t.get("accumulate_grad_batches", 1),
        gradient_clip_val=t.get("grad_clip_norm", 1.0),
        callbacks=[
            SaveOnImprovement(out_dir),
            ModelCheckpoint(dirpath=out_dir, filename="last", save_last=True, save_top_k=0),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=CSVLogger(out_dir),
        default_root_dir=out_dir,
    )
    trainer.fit(lit, datamodule=data, ckpt_path=args.resume)
    print(f"Done. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
