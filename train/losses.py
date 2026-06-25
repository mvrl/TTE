from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Symmetric CLIP-style cross-entropy over image↔location similarity logits."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_per_image, logits_per_location):
        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        loss_i2l = F.cross_entropy(logits_per_image, labels, label_smoothing=self.label_smoothing)
        loss_l2i = F.cross_entropy(logits_per_location, labels, label_smoothing=self.label_smoothing)
        return (loss_i2l + loss_l2i) / 2


class ReconstructionLoss(nn.Module):
    """MSE between image-attended semantic tokens and the raw image features —
    pushes the tokens to span the image feature space."""

    def forward(self, img_attended, image_features):
        return F.mse_loss(img_attended, image_features)


class AlignmentLoss(nn.Module):
    """KL(loc_attention || img_attention.detach()) — anchors the location-side
    token attention to the image (teacher) attention."""

    def forward(self, loc_attention, img_attention):
        img_attention = img_attention.detach()
        eps = 1e-8
        kl = loc_attention * (torch.log(loc_attention + eps) - torch.log(img_attention + eps))
        return kl.sum(dim=-1).mean()


class TotalLoss(nn.Module):
    """L = L_contrastive + λ_recon · L_recon + λ_align · L_align."""

    def __init__(self, lambda_reconstruction=0.0, lambda_alignment=0.0, label_smoothing=0.0):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_alignment = lambda_alignment
        self.contrastive_loss = ContrastiveLoss(label_smoothing)
        self.reconstruction_loss = ReconstructionLoss()
        self.alignment_loss = AlignmentLoss()

    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_location: torch.Tensor,
        loc_attention: Optional[torch.Tensor] = None,
        img_attention: Optional[torch.Tensor] = None,
        img_attended: Optional[torch.Tensor] = None,
        raw_image_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = logits_per_image.device
        loss_contrastive = self.contrastive_loss(logits_per_image, logits_per_location)

        if self.lambda_reconstruction > 0 and img_attended is not None:
            loss_reconstruction = self.reconstruction_loss(img_attended, raw_image_features)
        else:
            loss_reconstruction = torch.tensor(0.0, device=device)

        if self.lambda_alignment > 0 and loc_attention is not None:
            loss_alignment = self.alignment_loss(loc_attention, img_attention)
        else:
            loss_alignment = torch.tensor(0.0, device=device)

        total = (loss_contrastive
                 + self.lambda_reconstruction * loss_reconstruction
                 + self.lambda_alignment * loss_alignment)
        return {
            "loss": total,
            "loss_contrastive": loss_contrastive,
            "loss_reconstruction": loss_reconstruction,
            "loss_alignment": loss_alignment,
        }
