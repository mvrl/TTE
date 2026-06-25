import math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticRegisterAttention(nn.Module):
    """
    Shared global semantic tokens (registers) attended over by location and image. A site embedding attends over ``n_registers`` learnable concept tokens and is gate-mixed back with a skip connection. During training the image pathway attends over the same tokens, supervising them via reconstruction + attention alignment; at inference only the location pathway runs.
    """

    def __init__(
        self,
        site_embed_dim: int,
        output_dim: int,
        n_registers: int = 32,
        temperature: float = 0.2,
        img_temperature: float = 0.05,
        gate: float = 0.5,
        register_dropout: float = 0.0,
        temperature_anneal_end: Optional[float] = None,
        use_cosine_attention: bool = False,
    ):
        super().__init__()
        self.n_registers = n_registers
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim
        self.register_dropout = register_dropout
        self.temperature_anneal_end = temperature_anneal_end
        self.use_cosine_attention = use_cosine_attention
        self._anneal_progress = 0.0  # 0 to 1, set externally
        self.gate = gate  # fixed skip-connection blend (site embed vs register-attended)

        # Semantic registers: each is an output_dim concept embedding
        # Initialize with orthogonal vectors, then normalize to unit length
        # This ensures registers can reconstruct L2-normalized image features
        registers = torch.empty(n_registers, output_dim)
        nn.init.orthogonal_(registers)
        registers = F.normalize(registers, dim=-1)  # Unit norm for reconstruction
        self.registers = nn.Parameter(registers)

        # Linear projections to register logits
        # Use moderate init to avoid collapse to 1-2 registers
        # In cosine mode, disable bias for pure directional similarity
        use_bias = not use_cosine_attention
        self.loc_to_logits = nn.Linear(site_embed_dim, n_registers, bias=use_bias)
        self.img_to_logits = nn.Linear(output_dim, n_registers, bias=use_bias)
        nn.init.normal_(self.loc_to_logits.weight, std=0.1)
        nn.init.normal_(self.img_to_logits.weight, std=0.1)
        if use_bias:
            nn.init.zeros_(self.loc_to_logits.bias)
            nn.init.zeros_(self.img_to_logits.bias)

        # Project site embeddings to output dimension for skip connection
        if site_embed_dim != output_dim:
            self.site_to_out = nn.Linear(site_embed_dim, output_dim)
        else:
            self.site_to_out = None  # Identity for skip connection

        # Asymmetric temperatures: location attention is learnable (softer),
        # image attention is fixed and low (peaked, for hard alignment targets).
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.register_buffer('_img_temperature', torch.tensor(img_temperature))

        # Layer norm for output stability
        self.norm = nn.LayerNorm(output_dim)

        # Cached during forward for the training-time semantic losses.
        self._loc_attention: Optional[torch.Tensor] = None
        self._img_attention: Optional[torch.Tensor] = None
        self._img_attended: Optional[torch.Tensor] = None

    @property
    def temperature(self) -> torch.Tensor:
        """
        Location attention temperature (learnable, always positive). If temperature_anneal_end is set, interpolates between learned temp and anneal_end based on _anneal_progress (0=start, 1=end).
        """
        base_temp = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        if self.temperature_anneal_end is not None:
            # Linear interpolation: start at base_temp, end at temperature_anneal_end
            return base_temp * (1 - self._anneal_progress) + self.temperature_anneal_end * self._anneal_progress
        return base_temp

    def set_anneal_progress(self, progress: float):
        """Set temperature annealing progress (0=start, 1=end)."""
        self._anneal_progress = max(0.0, min(1.0, progress))

    @property
    def img_temperature(self) -> torch.Tensor:
        """Image attention temperature (fixed, low for peaked attention)."""
        return self._img_temperature

    def forward_location(
        self,
        site_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhance location embeddings with semantic register information.

        Args:
            site_embeddings: (N, site_embed_dim) aggregated site embeddings

        Returns:
            output: (N, output_dim) enhanced embeddings ready for MLP
            attention: (N, n_registers) attention weights
        """
        # Compute attention logits and weights
        if self.use_cosine_attention:
            # Cosine similarity: normalize both input and weights
            # Prevents weight magnitude divergence by making attention direction-based only
            site_emb_norm = F.normalize(site_embeddings, dim=-1)
            W_norm = F.normalize(self.loc_to_logits.weight, dim=1)
            logits = (site_emb_norm @ W_norm.T) / self.temperature
            if self.loc_to_logits.bias is not None:
                logits = logits + self.loc_to_logits.bias / self.temperature
        else:
            logits = self.loc_to_logits(site_embeddings) / self.temperature  # (N, R)

        # Register dropout: randomly mask registers during training to force exploration
        if self.training and self.register_dropout > 0:
            # Create dropout mask for registers (same mask across batch)
            mask = torch.rand(self.n_registers, device=logits.device) > self.register_dropout
            # Ensure at least one register is active
            if not mask.any():
                mask[torch.randint(self.n_registers, (1,))] = True
            # Apply mask to logits (masked registers get -inf)
            logits = logits.masked_fill(~mask.unsqueeze(0), float('-inf'))

        attention = F.softmax(logits, dim=-1)  # (N, R)

        # Attend to registers
        loc_attended = attention @ self.registers  # (N, output_dim)

        # Skip connection: combine site info and register info
        gate = self.gate
        if self.site_to_out is not None:
            site_out = self.site_to_out(site_embeddings)  # Project to output_dim
        else:
            site_out = site_embeddings  # Identity (already output_dim)
        output = self.norm((1 - gate) * site_out + gate * loc_attended)  # (N, output_dim)

        self._loc_attention = attention  # cached for the alignment loss
        return output, attention

    def forward_image(
        self,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image pathway for reconstruction and alignment losses.

        Called during training to get image register weights and attended values.
        Uses a fixed low temperature for peaked attention (hard alignment targets).

        Args:
            image_features: (N, output_dim) normalized image features from ViT

        Returns:
            attended: (N, output_dim) attended register values (for reconstruction loss)
            attention: (N, n_registers) attention weights (peaked due to low temperature)
        """
        # Image pathway: predict register weights with LOW temperature (peaked attention)
        logits = self.img_to_logits(image_features) / self.img_temperature  # (N, R)
        attention = F.softmax(logits, dim=-1)  # (N, R) - peaked

        # Attend to registers
        attended = attention @ self.registers  # (N, output_dim)

        # Store for loss computation
        self._img_attention = attention
        self._img_attended = attended

        return attended, attention

    def get_loss_components(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return stored attention weights for loss computation.

        Returns:
            Dictionary with:
            - loc_attention: (N, R) location attention weights
            - img_attention: (N, R) image attention weights
            - img_attended: (N, output_dim) image-attended registers
        """
        return {
            'loc_attention': self._loc_attention,
            'img_attention': self._img_attention,
            'img_attended': self._img_attended,
        }

    def clear_cache(self):
        """Clear cached attention weights after loss computation."""
        self._loc_attention = None
        self._img_attention = None
        self._img_attended = None

    def get_register_stats(self) -> dict:
        """Get statistics about register usage and diversity."""
        stats = {}

        with torch.no_grad():
            R = F.normalize(self.registers, dim=-1)
            sim = R @ R.T
            mask = ~torch.eye(self.n_registers, dtype=torch.bool, device=sim.device)
            off_diag = sim[mask]

            stats['n_registers'] = self.n_registers
            stats['gate'] = self.gate
            stats['loc_temperature'] = self.temperature.item()
            stats['img_temperature'] = self.img_temperature.item()
            stats['register_norm_mean'] = self.registers.norm(dim=-1).mean().item()
            stats['register_sim_mean'] = off_diag.mean().item()
            stats['register_sim_max'] = off_diag.max().item()

            max_entropy = np.log(self.n_registers)

            # Location attention entropy (higher = more uniform)
            if self._loc_attention is not None:
                loc_entropy = -(self._loc_attention * (self._loc_attention + 1e-8).log()).sum(dim=-1)
                stats['loc_entropy_mean'] = loc_entropy.mean().item()
                stats['loc_entropy_normalized'] = (loc_entropy.mean() / max_entropy).item()
                # Count active registers (>5% average usage)
                loc_usage = self._loc_attention.mean(dim=0)
                stats['loc_n_active'] = (loc_usage > 0.05).sum().item()

            # Image attention entropy
            if self._img_attention is not None:
                img_entropy = -(self._img_attention * (self._img_attention + 1e-8).log()).sum(dim=-1)
                stats['img_entropy_mean'] = img_entropy.mean().item()
                stats['img_entropy_normalized'] = (img_entropy.mean() / max_entropy).item()
                img_usage = self._img_attention.mean(dim=0)
                stats['img_n_active'] = (img_usage > 0.05).sum().item()

        return stats


class ResLayer(nn.Module):
    """Residual layer matching SatCLIP's FCNet architecture."""

    def __init__(self, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y

class VoronoiLocationEncoder(nn.Module):
    """Spherical Voronoi location encoder.

    ``n_sites`` learnable sites on the unit sphere, each with a direction, a
    per-site temperature τ, and an embedding. A coordinate is encoded as the
    soft-Voronoi-weighted sum of site embeddings (``softmax(τ_k·⟨g, s_k⟩) · E``),
    which attends over shared semantic tokens and passes through a residual MLP.
    """

    # Clamp range for log(τ): exp(-0.7) ~ 0.5, exp(6.2) ~ 500
    LOG_TAU_MIN = -0.7
    LOG_TAU_MAX = 6.2

    def __init__(
        self,
        n_sites: Optional[int] = None,
        init_sites: Optional[torch.Tensor] = None,
        site_embed_dim: int = 384,
        tau_init: float = 45.0,
        output_dim: int = 512,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        dropout: float = 0.5,
        # Semantic register attention
        n_registers: int = 64,
        register_temperature: float = 0.5,
        img_temperature: float = 0.05,
        register_fixed_gate: float = 0.5,
        register_dropout: float = 0.1,
        temperature_anneal_end: Optional[float] = 0.2,
        use_cosine_attention: bool = False,
    ):
        """Construct TTE location encoder.

        Sites are placed on a Fibonacci lattice from ``n_sites`` unless
        ``init_sites`` (a (K, 3) tensor of unit-sphere positions) is given.
        """
        super().__init__()

        if init_sites is None:
            init_sites = self._fibonacci_lattice(n_sites)
        if isinstance(init_sites, np.ndarray):
            init_sites = torch.from_numpy(init_sites.astype(np.float32))
        self.n_sites = init_sites.shape[0]
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim
        self.n_registers = n_registers

        # ---- Voronoi site parameters ----
        self._directions = nn.Parameter(F.normalize(init_sites.clone(), dim=-1))
        self._log_tau = nn.Parameter(torch.full((self.n_sites,), math.log(tau_init)))
        self._embeddings = nn.Parameter(torch.randn(self.n_sites, site_embed_dim) * 0.02)

        # ---- Semantic register attention ----
        self.semantic_registers = SemanticRegisterAttention(
            site_embed_dim=site_embed_dim,
            output_dim=output_dim,
            n_registers=n_registers,
            temperature=register_temperature,
            img_temperature=img_temperature,
            gate=register_fixed_gate,
            register_dropout=register_dropout,
            temperature_anneal_end=temperature_anneal_end,
            use_cosine_attention=use_cosine_attention,
        )

        # ---- Residual MLP head ----
        layers = [nn.Linear(output_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_reslayers):
            layers.append(ResLayer(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Site / tau properties
    # ------------------------------------------------------------------

    @property
    def sites(self) -> torch.Tensor:
        """Unit-sphere site positions (K, 3)."""
        return F.normalize(self._directions, dim=-1)

    @property
    def tau(self) -> torch.Tensor:
        """Per-site temperature (K,) — exp of the learnable log-tau."""
        return torch.exp(self._log_tau)

    @property
    def scaled_sites(self) -> torch.Tensor:
        """s_k * τ_k (used directly in the soft-Voronoi logit)."""
        return self.sites * self.tau.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        coords: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode (lat, lon) coordinates into the location embedding space.

        Args:
            coords: (N, 2) tensor of (lat, lon) in degrees.
            return_weights: If True, also return the (N, K) soft Voronoi
                weight matrix.

        Returns:
            output: (N, output_dim) location embedding.
            weights (optional): (N, K) soft assignment weights.
        """
        # lat/lon -> unit sphere
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        sphere = torch.stack([x, y, z], dim=-1)  # (N, 3)

        # Soft Voronoi assignment
        logits = sphere @ self.scaled_sites.t()             # (N, K)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(logits, dim=-1)                  # (N, K)
        site_emb = weights @ self._embeddings                # (N, site_embed_dim)

        # Semantic register attention, then residual MLP
        attended, _ = self.semantic_registers.forward_location(site_emb)
        output = self.mlp(attended)

        if return_weights:
            return output, weights
        return output

    def forward_image_registers(
        self,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image-side attention over semantic registers (training only)."""
        return self.semantic_registers.forward_image(image_features)

    def get_semantic_loss_components(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.semantic_registers.get_loss_components()

    def clear_semantic_cache(self):
        self.semantic_registers.clear_cache()

    def normalize_sites(self):
        """Re-project site directions onto S² and clamp log τ (called each step)."""
        with torch.no_grad():
            self._directions.data = F.normalize(self._directions.data, dim=-1)
            self._log_tau.data.clamp_(self.LOG_TAU_MIN, self.LOG_TAU_MAX)

    # ------------------------------------------------------------------
    # Parameter-group helpers
    # ------------------------------------------------------------------

    def get_tau_params(self) -> list:
        return [self._log_tau]

    def get_direction_params(self) -> list:
        """Site position parameters. Benefit from a higher LR than the MLP."""
        return [self._directions]

    def get_register_params(self) -> list:
        return list(self.semantic_registers.parameters())

    def get_base_params(self, exclude_directions: bool = False) -> list:
        """All parameters except (registers, tau, optionally directions)."""
        exclude_ids = set(id(p) for p in self.get_register_params())
        exclude_ids.update(id(p) for p in self.get_tau_params())
        if exclude_directions:
            exclude_ids.update(id(p) for p in self.get_direction_params())
        return [p for p in self.parameters() if id(p) not in exclude_ids]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def get_register_stats(self) -> dict:
        return self.semantic_registers.get_register_stats()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fibonacci_lattice(n: int) -> torch.Tensor:
        """Generate ``n`` points on the unit sphere via a Fibonacci lattice."""
        indices = torch.arange(n, dtype=torch.float32)
        phi = (1 + np.sqrt(5)) / 2
        theta = 2 * np.pi * indices / phi
        z = 1 - (2 * indices + 1) / n
        r = torch.sqrt(1 - z**2)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def load_sites_from_npz(path: str) -> torch.Tensor:
        """Load (K, 3) initial site positions from an .npz with a ``sites`` key."""
        data = np.load(path)
        return torch.from_numpy(data['sites'].astype(np.float32))
