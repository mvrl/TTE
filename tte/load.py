import os
import torch
from .model import TTE


def _strip_prefix(state_dict: dict) -> dict:
    """Drop the Lightning ``model.`` prefix if present."""
    if any(k.startswith("model.") for k in state_dict):
        return {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    return dict(state_dict)


def _infer_config(sd: dict) -> dict:
    """Infer the TTE config from a flat state dict's tensor shapes."""
    mlp_w = [k for k in sd if k.startswith("location.mlp.") and k.endswith(".weight")]
    cfg = {
        "n_sites": sd["location._directions"].shape[0],
        "site_embed_dim": sd["location._embeddings"].shape[1],
        "hidden_dim": sd["location.mlp.0.weight"].shape[0],
        "output_dim": max(sd[k].shape[0] for k in mlp_w),
        "n_reslayers": sum(1 for k in sd if k.startswith("location.mlp.") and k.endswith(".w1.weight")),
    }
    reg = "location.semantic_registers.registers"
    if reg in sd:
        cfg["n_registers"] = sd[reg].shape[0]
    return cfg


def load_tte_model(path_or_repo: str, device: str = "cpu", normalize: bool = True) -> TTE:
    """Load a TTE location encoder from a HuggingFace repo id or a local checkpoint."""
    if os.path.exists(path_or_repo):
        ckpt = torch.load(path_or_repo, map_location="cpu", weights_only=False)
        sd = _strip_prefix(ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)))
        model = TTE(**_infer_config(sd))
        loc_sd = {k: v for k, v in sd.items() if k.startswith("location.")}
        model.load_state_dict(loc_sd, strict=False)
    else:
        model = TTE.from_pretrained(path_or_repo)
    model.normalize = normalize
    return model.to(device).eval()
