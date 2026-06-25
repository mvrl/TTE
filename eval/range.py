import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# repo root (for `import tte`) + eval/ (for `from eval_range`) on path
for _p in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_range import DEFAULT_DATA_DIR, evaluate_all_tasks, load_tte_model


def latlon_to_sphere(coords: torch.Tensor) -> torch.Tensor:
    """(N, 2) (lat, lon) in degrees -> (N, 3) unit vectors on S²."""
    lat, lon = torch.deg2rad(coords[:, 0]), torch.deg2rad(coords[:, 1])
    return torch.stack([torch.cos(lat) * torch.cos(lon),
                        torch.cos(lat) * torch.sin(lon),
                        torch.sin(lat)], dim=-1)


# ---------------------------------------------------------------------------
# Dataset wrapper for database generation
# ---------------------------------------------------------------------------

def _load_hf_dataset(data_dir: str, batch_size: int = 64, num_workers: int = 4):
    """Load HuggingFace dataset for database generation.

    Returns a DataLoader yielding dicts with:
        'image': (B, 3, 224, 224) RGB tensor, ImageNet-normalized
        'coords': (B, 2) as (lat, lon) in degrees
    """
    from datasets import DatasetDict, load_from_disk
    import torchvision.transforms as T

    hf_path = Path(data_dir)
    if not ((hf_path / "dataset_info.json").exists() or (hf_path / "data").exists()):
        raise ValueError(
            f"{data_dir} is not a HuggingFace dataset. "
            "Expected dataset_info.json or data/ directory."
        )

    dataset = load_from_disk(data_dir)

    # Combine all splits for maximum coverage
    if isinstance(dataset, DatasetDict):
        from datasets import concatenate_datasets
        all_splits = [dataset[k] for k in dataset.keys()]
        dataset = concatenate_datasets(all_splits)
        print(f"Combined {len(all_splits)} splits: {len(dataset)} total samples")
    else:
        print(f"Dataset size: {len(dataset)} samples")

    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
            self.transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]

            # Load image
            if 'image' not in item:
                raise ValueError("Dataset must have 'image' field")
            image = item['image']
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            elif hasattr(image, 'numpy'):
                image = torch.from_numpy(image.numpy()).float()
            else:
                image = torch.tensor(image).float()

            # Ensure (C, H, W)
            if image.dim() == 3 and image.shape[0] > 4:
                image = image.permute(2, 0, 1)

            # Extract RGB from multispectral: bands [3,2,1] = B4,B3,B2
            if image.shape[0] > 3:
                image = image[[3, 2, 1], :, :]

            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0 if image.max() <= 255 else image / 10000.0

            image = self.transform(image)

            # Get coordinates as (lat, lon)
            if 'lat' in item and 'lon' in item:
                coords = torch.tensor([item['lat'], item['lon']],
                                      dtype=torch.float32)
            elif 'point' in item:
                point = item['point']
                # point is (lon, lat) -> swap to (lat, lon)
                coords = torch.tensor([point[1], point[0]],
                                      dtype=torch.float32)
            elif 'latitude' in item and 'longitude' in item:
                coords = torch.tensor([item['latitude'], item['longitude']],
                                      dtype=torch.float32)
            else:
                raise ValueError("Dataset must have coordinate fields")

            return {'image': image, 'coords': coords}

    dataloader = DataLoader(
        HFDatasetWrapper(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


# ---------------------------------------------------------------------------
# Database generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_range_database(
    model_path: str,
    data_dir: str,
    output_path: str,
    image_model_type: str = "timm_vit",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> str:
    """Generate a per-image RANGE-style retrieval database.

    For each training image, stores:
      - Location coordinates (lon, lat)
      - TTE location embedding (L2-normalized)
      - ViT-L image features (1024-dim)

    Args:
        model_path: Path to TTE model checkpoint.
        data_dir: Path to HuggingFace dataset directory.
        output_path: Where to save the .npz database.
        image_model_type: Image feature extractor type.
        batch_size: Batch size for processing.
        num_workers: DataLoader workers.
        device: 'cuda' or 'cpu'.
        max_samples: Cap on number of samples (None = use all).

    Returns:
        Path to saved database.
    """
    # --- Load TTE model ---
    print(f"Loading TTE model from: {model_path}")
    model_wrapper = load_tte_model(model_path, device=device, normalize=True)

    # Detect embedding dimension
    dummy = torch.tensor([[0.0, 0.0]], device=device)
    loc_dim = model_wrapper(dummy).shape[-1]
    print(f"  Location embedding dim: {loc_dim}")

    # --- Load image model (ViT-L, 1024-dim) ---
    import timm
    print("Loading ViT-L image model...")
    image_model = timm.create_model(
        'vit_large_patch16_224', pretrained=True, num_classes=0
    )
    image_model = image_model.eval().to(device)
    for p in image_model.parameters():
        p.requires_grad = False

    img_dim = 1024
    print(f"  Image feature dim: {img_dim}")

    # --- Load dataset ---
    print(f"Loading dataset from: {data_dir}")
    dataloader = _load_hf_dataset(data_dir, batch_size, num_workers)

    # --- Iterate and collect per-image embeddings ---
    all_locs = []
    all_tte_embeddings = []
    all_image_embeddings = []
    n_processed = 0

    for batch in tqdm(dataloader, desc="Building database"):
        images = batch['image'].to(device)          # (B, 3, 224, 224)
        coords_latlon = batch['coords']              # (B, 2) as (lat, lon)

        # TTE embeddings: wrapper expects (lon, lat), so swap
        coords_lonlat = coords_latlon[:, [1, 0]].to(device)  # (B, 2) as (lon, lat)
        tte_emb = model_wrapper(coords_lonlat)                # (B, D), L2-normalized

        # Image features
        img_emb = image_model(images)                          # (B, 1024)

        # Store: locs as (lon, lat) matching RANGE convention
        all_locs.append(coords_lonlat.cpu().numpy())
        all_tte_embeddings.append(tte_emb.cpu().numpy())
        all_image_embeddings.append(img_emb.cpu().numpy())

        n_processed += images.shape[0]
        if max_samples is not None and n_processed >= max_samples:
            break

    # Concatenate
    all_locs = np.concatenate(all_locs, axis=0).astype(np.float32)
    all_tte_embeddings = np.concatenate(all_tte_embeddings, axis=0).astype(np.float32)
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype(np.float32)

    if max_samples is not None:
        all_locs = all_locs[:max_samples]
        all_tte_embeddings = all_tte_embeddings[:max_samples]
        all_image_embeddings = all_image_embeddings[:max_samples]

    print(f"\nDatabase statistics:")
    print(f"  Samples:          {all_locs.shape[0]}")
    print(f"  locs shape:       {all_locs.shape}")
    print(f"  tte_embeddings:   {all_tte_embeddings.shape}")
    print(f"  image_embeddings: {all_image_embeddings.shape}")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(
        output_path,
        locs=all_locs,                       # (N, 2) as (lon, lat) degrees
        tte_embeddings=all_tte_embeddings,   # (N, D) L2-normalized
        image_embeddings=all_image_embeddings,  # (N, 1024)
    )
    print(f"  Saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# RANGE retrieval model
# ---------------------------------------------------------------------------

class RangeRetrieval(nn.Module):
    """RANGE-style per-image retrieval using TTE location embeddings.

    Modes:
      - "range":  Semantic-only retrieval (cosine similarity, single temperature)
      - "range+": Blended semantic + geographic retrieval

    Forward input:  (B, 2) as (lon, lat) degrees  [matches eval_range.py convention]
    Forward output: (B, image_dim + location_dim)
    """

    def __init__(
        self,
        model_path: str,
        db_path: str,
        mode: str = "range",
        semantic_temperature: Optional[float] = None,
        geo_temperature: float = 40.0,
        beta: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()

        self.mode = mode
        self.device = device

        # Default temperatures depend on mode (matching RANGE reference)
        if semantic_temperature is None:
            self.semantic_temperature = 15.0 if mode == "range" else 12.0
        else:
            self.semantic_temperature = semantic_temperature
        self.geo_temperature = geo_temperature
        self.beta = beta

        # --- Load TTE model ---
        print(f"Loading TTE model from: {model_path}")
        self.model_wrapper = load_tte_model(model_path, device=device, normalize=True)

        # Detect location embedding dim
        dummy = torch.tensor([[0.0, 0.0]], device=device)
        dummy_emb = self.model_wrapper(dummy)
        self.location_dim = dummy_emb.shape[-1]

        # --- Load database ---
        print(f"Loading RANGE database from: {db_path}")
        db = np.load(db_path, allow_pickle=True)

        self.register_buffer(
            'db_tte_embeddings',
            F.normalize(torch.tensor(db['tte_embeddings'], dtype=torch.float32), dim=-1)
        )
        self.register_buffer(
            'db_image_embeddings',
            torch.tensor(db['image_embeddings'], dtype=torch.float32)
        )

        self.image_dim = self.db_image_embeddings.shape[-1]
        self.output_dim = self.image_dim + self.location_dim
        n_db = self.db_tte_embeddings.shape[0]

        print(f"  DB entries:       {n_db}")
        print(f"  Location dim:     {self.location_dim}")
        print(f"  Image dim:        {self.image_dim}")
        print(f"  Output dim:       {self.output_dim}")
        print(f"  Mode:             {self.mode}")
        print(f"  Semantic temp:    {self.semantic_temperature}")

        # --- For RANGE+: precompute geographic coordinates ---
        if self.mode == "range+":
            db_locs = db['locs']  # (N, 2) as (lon, lat)
            # latlon_to_sphere expects (lat, lon), so swap columns
            db_latlon = torch.tensor(
                db_locs[:, [1, 0]], dtype=torch.float32
            )
            self.register_buffer('db_xyz', latlon_to_sphere(db_latlon))

            print(f"  Geo temp:         {self.geo_temperature}")
            print(f"  Beta:             {self.beta}")

        self.to(device)

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) as (lon, lat) in degrees.

        Returns:
            (B, image_dim + location_dim) augmented embeddings.
        """
        coords = coords.float().to(self.device)

        # Get TTE location embedding (wrapper handles lon,lat -> lat,lon + L2-norm)
        loc_emb = self.model_wrapper(coords)  # (B, D), L2-normalized

        # Cosine similarity to all DB entries
        sim = loc_emb.float() @ self.db_tte_embeddings.t()  # (B, N)

        if self.mode == "range":
            # Semantic-only retrieval
            weights = F.softmax(sim * self.semantic_temperature, dim=-1)
            retrieved = weights @ self.db_image_embeddings  # (B, 1024)

        elif self.mode == "range+":
            # Semantic component
            sem_weights = F.softmax(sim * self.semantic_temperature, dim=-1)
            sem_retrieved = sem_weights @ self.db_image_embeddings

            # Geographic component: convert query (lon, lat) -> (lat, lon) -> xyz
            lat_lon = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
            query_xyz = latlon_to_sphere(lat_lon).float().to(self.device)

            geo_sim = query_xyz @ self.db_xyz.t()  # (B, N)
            geo_weights = F.softmax(geo_sim * self.geo_temperature, dim=-1)
            geo_retrieved = geo_weights @ self.db_image_embeddings

            # Blend
            retrieved = (1 - self.beta) * geo_retrieved + self.beta * sem_retrieved

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'range' or 'range+'.")

        # Concatenate [retrieved_features, location_embedding]
        return torch.cat([retrieved, loc_emb], dim=-1)


# ---------------------------------------------------------------------------
# Evaluation-compatible wrapper
# ---------------------------------------------------------------------------

class RangeRetrievalWrapper:
    """Wrapper matching the LocationEncoderWrapper interface for eval_range.py.

    Accepts (lon, lat) coordinates and returns augmented embeddings.
    Plugs directly into extract_embeddings() and evaluate_all_tasks().
    """

    def __init__(self, retrieval_model: RangeRetrieval):
        self.retrieval = retrieval_model
        self.output_dim = retrieval_model.output_dim

    def eval(self):
        self.retrieval.eval()
        return self

    def to(self, device):
        self.retrieval.to(device)
        return self

    @torch.no_grad()
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) as (lon, lat) from eval_range.py datasets.
        Returns:
            (N, output_dim) embeddings.
        """
        return self.retrieval(coords)


def load_range_retrieval(
    model_path: str,
    db_path: str,
    mode: str = "range",
    device: str = "cuda",
    **kwargs,
) -> RangeRetrievalWrapper:
    """Load RANGE-style retrieval model for evaluation.

    Args:
        model_path: Path to TTE model checkpoint.
        db_path: Path to RANGE database (.npz).
        mode: "range" or "range+".
        device: 'cuda' or 'cpu'.
        **kwargs: Additional args passed to RangeRetrieval
            (semantic_temperature, geo_temperature, beta).

    Returns:
        RangeRetrievalWrapper ready for evaluation.
    """
    retrieval = RangeRetrieval(
        model_path=model_path,
        db_path=db_path,
        mode=mode,
        device=device,
        **kwargs,
    )
    retrieval.eval()
    return RangeRetrievalWrapper(retrieval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RANGE-style per-image retrieval for TTE location encoders"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Generate per-image retrieval database")
    gen.add_argument("--model_path", type=str, required=True,
                     help="Path to TTE model checkpoint")
    gen.add_argument("--data_dir", type=str, required=True,
                     help="Path to HuggingFace dataset directory")
    gen.add_argument("--output", type=str, default=None,
                     help="Output .npz path (default: prototypes/<model>_range_db.npz)")
    gen.add_argument("--image_model", type=str, default="timm_vit",
                     choices=["timm_vit"],
                     help="Image feature extractor")
    gen.add_argument("--batch_size", type=int, default=64)
    gen.add_argument("--num_workers", type=int, default=4)
    gen.add_argument("--device", type=str, default="cuda")
    gen.add_argument("--max_samples", type=int, default=None,
                     help="Cap number of DB entries (default: use all)")

    # --- evaluate ---
    ev = subparsers.add_parser("evaluate", help="Evaluate with RANGE retrieval")
    ev.add_argument("--model_path", type=str, required=True,
                    help="Path to TTE model checkpoint")
    ev.add_argument("--db_path", type=str, required=True,
                    help="Path to RANGE database (.npz)")
    ev.add_argument("--mode", type=str, default="range",
                    choices=["range", "range+"])
    ev.add_argument("--data_dir", type=str, default=None,
                    help="Evaluation data directory")
    ev.add_argument("--tasks", type=str, nargs="+", default=None,
                    help="Tasks to evaluate (default: all)")
    ev.add_argument("--device", type=str, default="cuda")
    ev.add_argument("--semantic_temp", type=float, default=None,
                    help="Semantic temperature (default: 15.0 for range, 12.0 for range+)")
    ev.add_argument("--geo_temp", type=float, default=40.0,
                    help="Geographic temperature for range+ (default: 40.0)")
    ev.add_argument("--beta", type=float, default=0.5,
                    help="Blend factor for range+ (default: 0.5)")
    ev.add_argument("--batch_size", type=int, default=256)
    ev.add_argument("--num_workers", type=int, default=4)
    ev.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save results")

    args = parser.parse_args()

    if args.command == "generate":
        if args.output is None:
            model_name = Path(args.model_path).stem
            args.output = f"prototypes/{model_name}_range_db.npz"

        generate_range_database(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_path=args.output,
            image_model_type=args.image_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_samples=args.max_samples,
        )

    elif args.command == "evaluate":
        if args.data_dir is None:
            args.data_dir = DEFAULT_DATA_DIR

        # Build kwargs for RangeRetrieval
        retrieval_kwargs = {}
        if args.semantic_temp is not None:
            retrieval_kwargs['semantic_temperature'] = args.semantic_temp
        retrieval_kwargs['geo_temperature'] = args.geo_temp
        retrieval_kwargs['beta'] = args.beta

        wrapper = load_range_retrieval(
            model_path=args.model_path,
            db_path=args.db_path,
            mode=args.mode,
            device=args.device,
            **retrieval_kwargs,
        )

        results = evaluate_all_tasks(
            wrapper, args.data_dir, args.device,
            args.batch_size, args.num_workers, args.tasks,
        )

        # Save results
        if args.output_dir is None:
            model_name = Path(args.model_path).stem
            args.output_dir = f"outputs/eval/{model_name}_range_{args.mode}"

        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
