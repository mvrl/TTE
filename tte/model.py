import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .location_encoder import VoronoiLocationEncoder


class TTE(nn.Module, PyTorchModelHubMixin):
    """TTE location encoder. __init__ kwargs are the saved config; from_pretrained
    rebuilds from config.json then loads model.safetensors."""

    def __init__(
        self,
        n_sites: int = 4096,
        site_embed_dim: int = 384,
        output_dim: int = 512,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        n_registers: int = 64,
        register_fixed_gate: float = 0.5,
        use_cosine_attention: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.location = VoronoiLocationEncoder(
            n_sites=n_sites,
            site_embed_dim=site_embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_reslayers=n_reslayers,
            n_registers=n_registers,
            register_fixed_gate=register_fixed_gate,
            use_cosine_attention=use_cosine_attention,
        )

    @torch.no_grad()
    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) of (lat, lon) in degrees -> (N, output_dim), L2-normalized iff self.normalize."""
        coords = torch.as_tensor(coords, dtype=torch.float32, device=self.device)
        emb = self.location(coords).float()
        return F.normalize(emb, dim=-1) if self.normalize else emb

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encode(coords)

    @property
    def device(self) -> torch.device:
        return self.location._directions.device
