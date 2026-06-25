from typing import Any, Dict, Optional

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


def coordinate_jitter(coords: torch.Tensor, radius: float = 0.01) -> torch.Tensor:
    """Add uniform ±radius° jitter to (lat, lon) for augmentation (~1km at 0.01°)."""
    return coords + (torch.rand_like(coords) * 2 - 1) * radius


class HFSatelliteDataset(Dataset):
    """Wraps a HuggingFace Arrow dataset of 13-band Sentinel-2 images with lon/lat."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        sample = {
            "image": item["image"].float(),
            "point": torch.tensor([item["lon"], item["lat"]], dtype=torch.float32),
        }
        return self.transform(sample) if self.transform else sample


def _make_transform(crop_size: int, jitter_radius: float, normalize: bool):
    augment = T.Compose([
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(kernel_size=3),
    ])

    def transform(sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        image = sample["image"]
        if normalize:
            image = image / 10000.0          # raw Sentinel-2 reflectance
        image = augment(image)
        point = coordinate_jitter(sample["point"], radius=jitter_radius)
        point = torch.stack([point[1], point[0]])   # (lon, lat) -> (lat, lon) for the model
        return {"image": image, "point": point}

    return transform


class HFMultispectralDataModule(pl.LightningDataModule):
    """Sentinel-2 (image, lat/lon) pairs from a HuggingFace Arrow dataset."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        crop_size: int = 224,
        val_split: float = 0.1,
        preprocessed: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.val_split = val_split
        self.preprocessed = preprocessed
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        from datasets import load_from_disk

        full = load_from_disk(self.dataset_path)
        full.set_format("torch")
        split = full.train_test_split(test_size=self.val_split, seed=self.seed)
        normalize = not self.preprocessed
        self.train_dataset = HFSatelliteDataset(
            split["train"], _make_transform(self.crop_size, 0.01, normalize))
        self.val_dataset = HFSatelliteDataset(
            split["test"], _make_transform(self.crop_size, 0.0, normalize))
        print(f"Train {len(self.train_dataset)} / val {len(self.val_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
