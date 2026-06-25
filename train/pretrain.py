from typing import Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from tte.location_encoder import VoronoiLocationEncoder

# Frozen MAE ViT-L/16 encoder pretrained on 13-band Sentinel-2 by SSL4EO-S12.
# https://github.com/zhu-xlab/SSL4EO-S12
MAE_VITL16_URL = (
    "https://huggingface.co/wangyi111/SSL4EO-S12/resolve/"
    "75c72195d35201dc1fb210818993518c25da566b/B13_vitl16_mae_ep99_enc.pth"
)


class TTEPretrainModel(nn.Module):
    """CLIP-style image-location contrastive model used to pretrain TTE.

    Pairs the Spherical Voronoi location encoder with a frozen Sentinel-2 ViT and aligns them with a symmetric contrastive loss. Only the location encoder and the ViT head are trained; at inference only the location encoder is used.
    """

    def __init__(
        self,
        output_dim: int = 512,
        image_resolution: int = 224,
        # Voronoi location encoder
        n_sites: int = 4096,
        tau_init: float = 45.0,
        site_embed_dim: int = 384,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        dropout: float = 0.5,
        init_sites: Optional[torch.Tensor] = None,
        # Semantic register attention
        n_registers: int = 64,
        register_temperature: float = 0.5,
        img_temperature: float = 0.05,
        register_fixed_gate: float = 0.5,
        register_dropout: float = 0.1,
        temperature_anneal_end: Optional[float] = 0.2,
        use_cosine_attention: bool = False,
    ):
        super().__init__()
        self.visual = self._build_vit(output_dim, image_resolution)
        self.location = VoronoiLocationEncoder(
            n_sites=n_sites,
            init_sites=init_sites,
            site_embed_dim=site_embed_dim,
            tau_init=tau_init,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_reslayers=n_reslayers,
            dropout=dropout,
            n_registers=n_registers,
            register_temperature=register_temperature,
            img_temperature=img_temperature,
            register_fixed_gate=register_fixed_gate,
            register_dropout=register_dropout,
            temperature_anneal_end=temperature_anneal_end,
            use_cosine_attention=use_cosine_attention,
        )
        self.output_dim = output_dim
        self.n_voronoi_sites = self.location.n_sites
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @staticmethod
    def _build_vit(output_dim: int, image_resolution: int) -> nn.Module:
        # Band order: B1,B2,B3,B4,B5,B6,B7,B8,B8a,B9,B10,B11,B12 (13 bands).
        visual = timm.create_model(
            "vit_large_patch16_224", in_chans=13,
            num_classes=output_dim, img_size=image_resolution,
        )
        state_dict = load_state_dict_from_url(MAE_VITL16_URL, progress=True, map_location="cpu")
        visual.load_state_dict(state_dict, strict=False)
        visual.requires_grad_(False)       # frozen backbone
        visual.head.requires_grad_(True)   # train only the projection head
        return visual

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))

    def encode_location(self, coords: torch.Tensor, return_weights: bool = False):
        return self.location(coords, return_weights=return_weights)

    def forward_image_registers(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Image-side attention over the semantic tokens (training only)."""
        return self.location.forward_image_registers(image_features)

    def get_semantic_loss_components(self) -> dict:
        return self.location.get_semantic_loss_components()

    def clear_semantic_cache(self):
        self.location.clear_semantic_cache()

    def forward(
        self,
        image: torch.Tensor,
        coords: torch.Tensor,
        return_weights: bool = False,
        return_image_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Contrastive logits for a batch. ``coords`` are (lat, lon) in degrees.

        Returns ``(logits_per_image, logits_per_location[, weights][, image_features])``;
        ``image_features`` lets the training step reuse the ViT output for the
        semantic-register losses instead of a second forward pass.
        """
        image_features = self.encode_image(image)
        loc = self.encode_location(coords, return_weights=return_weights)
        location_features, weights = loc if return_weights else (loc, None)

        image_features = F.normalize(image_features, dim=-1)
        location_features = F.normalize(location_features.float(), dim=-1)

        logit_scale = self.logit_scale.clamp(max=4.6052).exp()  # cap τ at 100 (as in CLIP)
        logits_per_image = logit_scale * image_features @ location_features.t()

        out = [logits_per_image, logits_per_image.t()]
        if return_weights:
            out.append(weights)
        if return_image_features:
            out.append(image_features)
        return tuple(out)

    def get_trainable_param_groups(
        self,
        lr_location: float = 1e-4,
        lr_tau: Optional[float] = None,
        lr_directions: Optional[float] = None,
        lr_registers: Optional[float] = None,
        lr_image_proj: float = 1e-5,
        lr_logit_scale: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list:
        lr_tau = lr_tau or lr_location
        lr_registers = lr_registers or lr_location
        split_directions = lr_directions is not None

        groups = [
            {"params": self.location.get_base_params(exclude_directions=split_directions),
             "lr": lr_location, "weight_decay": weight_decay, "name": "location_encoder"},
            {"params": self.location.get_tau_params(),
             "lr": lr_tau, "weight_decay": 0.0, "name": "location_tau"},
            {"params": self.location.get_register_params(),
             "lr": lr_registers, "weight_decay": weight_decay, "name": "location_registers"},
            {"params": [self.logit_scale],
             "lr": lr_logit_scale, "weight_decay": 0.0, "name": "logit_scale"},
            {"params": [p for p in self.visual.head.parameters() if p.requires_grad],
             "lr": lr_image_proj, "weight_decay": 0.0, "name": "image_proj"},
        ]
        if split_directions:
            groups.append(
                {"params": self.location.get_direction_params(),
                 "lr": lr_directions, "weight_decay": 0.0, "name": "location_directions"})
        return groups
