from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from pretrain import TTEPretrainModel
from losses import TotalLoss


class TTELightningModule(pl.LightningModule):
    """Lightning wrapper around TTEPretrainModel (contrastive image-location training)."""

    def __init__(
        self,
        model: TTEPretrainModel,
        lambda_reconstruction: float = 0.0,
        lambda_alignment: float = 0.0,
        lr_location: float = 1e-4,
        lr_tau: Optional[float] = None,
        lr_directions: Optional[float] = None,
        lr_registers: Optional[float] = None,
        lr_image_proj: float = 1e-5,
        lr_logit_scale: float = 1e-5,
        weight_decay: float = 0.01,
        max_epochs: int = 300,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = TotalLoss(lambda_reconstruction=lambda_reconstruction,
                                 lambda_alignment=lambda_alignment)

    def _shared_step(self, batch, prefix: str) -> Dict[str, torch.Tensor]:
        images, coords = batch["image"], batch["point"]

        # One ViT forward gives logits and the L2-normalized image features; we
        # reuse those features for the semantic-token losses (no second pass).
        logits_per_image, logits_per_location, _, image_features = self.model(
            images, coords, return_weights=True, return_image_features=True,
        )
        image_features = image_features.detach()  # token losses update only token params

        loc_attention = img_attention = img_attended = None
        use_token_losses = (self.hparams.lambda_reconstruction > 0
                            or self.hparams.lambda_alignment > 0)
        if use_token_losses:
            img_attended, img_attention = self.model.forward_image_registers(image_features)
            loc_attention = self.model.get_semantic_loss_components()["loc_attention"]

        loss_dict = self.loss_fn(
            logits_per_image=logits_per_image,
            logits_per_location=logits_per_location,
            loc_attention=loc_attention,
            img_attention=img_attention,
            img_attended=img_attended,
            raw_image_features=image_features,
        )

        for key, value in loss_dict.items():
            self.log(f"{prefix}/{key}", value, prog_bar=(key == "loss"), sync_dist=True)
        if prefix == "train":
            self.log("train/logit_scale", self.model.logit_scale.exp(), sync_dist=False)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], sync_dist=False)
            if self.global_step % 100 == 0:
                for key, value in self.model.location.get_register_stats().items():
                    self.log(f"train/register_{key}", value, sync_dist=True)

        if use_token_losses:
            self.model.clear_semantic_cache()
        return loss_dict

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")["loss"]

    def on_after_backward(self):
        # Keep sites on S² and clamp log τ after each step.
        self.model.location.normalize_sites()

    def on_train_epoch_start(self):
        # Linearly anneal the location-attention temperature over training.
        sr = self.model.location.semantic_registers
        if sr.temperature_anneal_end is not None:
            sr.set_anneal_progress(self.current_epoch / max(1, self.hparams.max_epochs - 1))

    def configure_optimizers(self):
        param_groups = self.model.get_trainable_param_groups(
            lr_location=self.hparams.lr_location,
            lr_tau=self.hparams.lr_tau,
            lr_directions=self.hparams.lr_directions,
            lr_registers=self.hparams.lr_registers,
            lr_image_proj=self.hparams.lr_image_proj,
            lr_logit_scale=self.hparams.lr_logit_scale,
            weight_decay=self.hparams.weight_decay,
        )
        optimizer = AdamW(param_groups)

        warmup, total = self.hparams.warmup_epochs, self.hparams.max_epochs
        if warmup > 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6),
                ],
                milestones=[warmup],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total, eta_min=1e-6)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
