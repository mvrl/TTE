import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import top_k_accuracy_score, make_scorer
from scipy.special import softmax

# repo root (for `import tte`) + eval/ on path
for _p in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tte import load_tte_model as _load_tte

# ==============================================================================
# Configuration
# ==============================================================================

# Default data directory — override with --data_dir or the RANGE_EVAL_DATA env var.
DEFAULT_DATA_DIR = os.environ.get("RANGE_EVAL_DATA", "data/range_eval_data")

# Source: https://github.com/mvrl/RANGE
RANDOM_SEED = 42
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# ==============================================================================
# Dataset Classes
# ==============================================================================

class BiomeDataset(Dataset):
    """Biome classification dataset from ecoregion data."""

    def __init__(self, data_dir):
        train_path = os.path.join(data_dir, 'ecoregion_train.csv')
        val_path = os.path.join(data_dir, 'ecoregion_val.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        df = pd.concat([train_df, val_df])
        df = df.dropna(subset=['BIOME_NAME']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['BIOME_NAME'])
        self.locations = df[['X', 'Y']].values
        self.num_classes = df['BIOME_NAME'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class EcoregionDataset(Dataset):
    """Ecoregion classification dataset."""

    def __init__(self, data_dir):
        train_path = os.path.join(data_dir, 'ecoregion_train.csv')
        val_path = os.path.join(data_dir, 'ecoregion_val.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        df = pd.concat([train_df, val_df])
        df = df.dropna(subset=['ECO_NAME']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['ECO_NAME'])
        self.locations = df[['X', 'Y']].values
        self.num_classes = df['ECO_NAME'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class CountryDataset(Dataset):
    """Country classification dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['country', 'lat', 'lon']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['country'])
        self.locations = df[['lon', 'lat']].values
        self.num_classes = df['country'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class OceanDataset(Dataset):
    """Ocean/land binary classification dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['land', 'lat', 'lon']).reset_index(drop=True)

        self.labels = df['land'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = df['land'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class TemperatureDataset(Dataset):
    """Mean temperature regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['meanT']).reset_index(drop=True)

        self.labels = df['meanT'].values
        self.locations = df[['Lon', 'Lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class HousingDataset(Dataset):
    """California housing prices regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['median_house_value']).reset_index(drop=True)

        self.labels = df['median_house_value'].values
        self.locations = df[['longitude', 'latitude']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class ElevationDataset(Dataset):
    """Elevation regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['elevation']).reset_index(drop=True)

        self.labels = df['elevation'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class PopulationDataset(Dataset):
    """Population regression dataset (log-transformed)."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['population']).reset_index(drop=True)

        self.labels = df['population'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        # Apply log transform: log(1 + population)
        return loc, np.log(1 + label)


class ERA5Dataset(Dataset):
    """ERA5 climate variable regression dataset."""

    def __init__(self, data_path, variable='air_temp_m'):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=[variable]).reset_index(drop=True)

        self.variable = variable
        self.labels = df[variable].values
        self.locations = df[['Longitude', 'Latitude']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


# ==============================================================================
# Checkerboard Dataset (Synthetic)
# ==============================================================================

def generate_fibonacci_lattice(n_points, n_classes=16):
    """Generate points on a sphere using Fibonacci lattice sampling."""
    n_points = n_points // 2
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    lats, lons, labels = [], [], []

    for i in np.arange(-n_points, n_points):
        lat = np.arcsin((2 * i) / (2 * n_points + 1)) * 180 / np.pi
        lon = (i % phi) * (360 / phi)

        # Wrap longitude to [-180, 180]
        if lon < -180:
            lon += 360
        if lon > 180:
            lon -= 360

        lons.append(lon)
        lats.append(lat)
        labels.append(i % n_classes)

    return np.array(lons), np.array(lats), np.array(labels)


def haversine_distance(lon1, lat1, lon2, lat2, radius=1.0):
    """Calculate pairwise Haversine distances."""
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)

    dlon = lon2[:, np.newaxis] - lon1
    dlat = lat2[:, np.newaxis] - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c


def cart2sph(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    hxy = np.hypot(x, y)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el


def get_checker_data(n_samples, n_support, n_classes, seed=0, grid=False):
    """Generate checkerboard pattern data on a sphere."""
    lons, lats, labels = generate_fibonacci_lattice(n_support, n_classes=n_classes)

    if grid:
        # Use Fibonacci lattice for evaluation grid
        lons_grid, lats_grid, _ = generate_fibonacci_lattice(n_samples)
        distances = haversine_distance(lons_grid, lats_grid, lons, lats)
        labels_grid = labels[distances.argmin(0)]

        lonlats = torch.from_numpy(np.stack([lons_grid, lats_grid])).T
        labels_out = torch.from_numpy(labels_grid)
    else:
        # Random sampling on sphere
        rng = np.random.RandomState(seed)
        x, y, z = rng.normal(size=(3, n_samples))
        az, el = cart2sph(x, y, z)
        lons_seed, lats_seed = np.rad2deg(az), np.rad2deg(el)

        distances = haversine_distance(lons_seed, lats_seed, lons, lats)
        labels_seed = labels[distances.argmin(0)]

        lonlats = torch.from_numpy(np.stack([lons_seed, lats_seed])).T
        labels_out = torch.from_numpy(labels_seed)

    return lonlats, torch.zeros_like(labels_out), labels_out


class CheckerboardDataset:
    """Synthetic checkerboard classification dataset on a sphere."""

    def __init__(self, n_samples=10000, n_classes=16, n_support=200):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_support = n_support

        # Training set (seed=0 for consistency)
        self.train_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, seed=0
        ))

        # Validation set (seed=1 for consistency)
        self.valid_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, seed=1
        ))

        # Evaluation set (grid-based)
        self.eval_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, grid=True
        ))


# ==============================================================================
# Dataset Loading
# ==============================================================================

# Task type mapping
CLASSIFICATION_TASKS = {'biome', 'ecoregion', 'country', 'ocean'}
REGRESSION_TASKS = {'temperature', 'housing', 'elevation', 'population'}


def get_dataset(task_name, data_dir=DEFAULT_DATA_DIR, batch_size=256, num_workers=4):
    """
    Load dataset for a given task.

    Args:
        task_name: Name of the evaluation task
        data_dir: Directory containing evaluation data
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, num_classes, task_type
    """
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    if task_name == 'biome':
        dataset = BiomeDataset(data_dir)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'ecoregion':
        dataset = EcoregionDataset(data_dir)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'country':
        data_path = os.path.join(data_dir, 'country.csv')
        dataset = CountryDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'ocean':
        train_path = os.path.join(data_dir, 'land_ocean_train.csv')
        val_path = os.path.join(data_dir, 'land_ocean_test.csv')
        train_ds = OceanDataset(train_path)
        val_ds = OceanDataset(val_path)
        num_classes = train_ds.num_classes
        task_type = 'classification'

    elif task_name == 'temperature':
        data_path = os.path.join(data_dir, 'temp.csv')
        dataset = TemperatureDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'housing':
        data_path = os.path.join(data_dir, 'housing.csv')
        dataset = HousingDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'elevation':
        data_path = os.path.join(data_dir, 'elevation.csv')
        dataset = ElevationDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'population':
        data_path = os.path.join(data_dir, 'population.csv')
        dataset = PopulationDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name.startswith('era5-'):
        # ERA5 tasks: era5-air_temp_m, era5-precip_m, etc.
        variable = task_name.split('-', 1)[1]
        data_path = os.path.join(data_dir, 'ERA5_Land_Clipped_2020.csv')
        dataset = ERA5Dataset(data_path, variable=variable)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name.startswith('checker_'):
        # Checkerboard tasks: checker_100, checker_200, etc.
        n_support = int(task_name.split('_')[1])
        n_classes = 16
        checker = CheckerboardDataset(n_samples=10000, n_classes=n_classes, n_support=n_support)
        train_ds = checker.train_ds
        val_ds = checker.eval_ds
        num_classes = n_classes
        task_type = 'classification'

    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False
    )

    return train_loader, val_loader, num_classes, task_type


# ==============================================================================
# Embedding Extraction
# ==============================================================================

def extract_embeddings(data_loader, model, device='cuda'):
    """
    Extract embeddings from a location encoder model.

    Args:
        data_loader: DataLoader providing (coords, labels) pairs
        model: Location encoder model
        device: Device to run inference on

    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,)
    """
    model.eval()
    model = model.to(device)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            # Handle both 2-tuple (coords, labels) and 3-tuple (coords, _, labels) formats
            if len(batch) == 2:
                coords, labels = batch
            else:
                coords, _, labels = batch  # Checkerboard dataset returns 3 values
            coords = coords.to(device)

            emb = model(coords).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return embeddings, labels


def save_embeddings(embeddings, labels, save_path):
    """Save embeddings and labels to npz file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, embeddings=embeddings, labels=labels)
    print(f"Saved embeddings to {save_path}")


def load_embeddings(load_path):
    """Load embeddings and labels from npz file."""
    data = np.load(load_path, allow_pickle=True)
    return data['embeddings'], data['labels']


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_embeddings(train_embeddings, train_labels, val_embeddings, val_labels, task_type):
    """
    Evaluate embeddings using Ridge regression/classification.

    Args:
        train_embeddings: Training set embeddings
        train_labels: Training set labels
        val_embeddings: Validation set embeddings
        val_labels: Validation set labels
        task_type: 'classification' or 'regression'

    Returns:
        score: Accuracy (classification) or R² (regression)
        model: Fitted Ridge model
    """
    # Normalize embeddings
    scaler = MinMaxScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)

    if task_type == 'classification':
        model = RidgeClassifierCV(alphas=RIDGE_ALPHAS, cv=10)
    else:
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)

    model.fit(train_embeddings, train_labels)
    score = model.score(val_embeddings, val_labels)

    return score, model


def evaluate_task(task_name, model, data_dir=DEFAULT_DATA_DIR, device='cuda',
                  batch_size=256, num_workers=4):
    """
    Full evaluation pipeline for a single task.

    Args:
        task_name: Name of the evaluation task
        model: Location encoder model
        data_dir: Directory containing evaluation data
        device: Device to run inference on
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading

    Returns:
        score: Evaluation metric (accuracy or R²)
    """
    # Load data
    train_loader, val_loader, num_classes, task_type = get_dataset(
        task_name, data_dir, batch_size, num_workers
    )

    print(f"\nEvaluating task: {task_name}")
    print(f"  Task type: {task_type}")
    if task_type == 'classification':
        print(f"  Num classes: {num_classes}")

    # Extract embeddings
    train_emb, train_labels = extract_embeddings(train_loader, model, device)
    val_emb, val_labels = extract_embeddings(val_loader, model, device)

    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")

    # Evaluate
    score, _ = evaluate_embeddings(train_emb, train_labels, val_emb, val_labels, task_type)

    metric_name = "Accuracy" if task_type == 'classification' else "R²"
    print(f"  {metric_name}: {score:.4f}")

    return score


# Tasks with predefined train/val files (no random_split) — single-shot only.
FIXED_SPLIT_TASKS = {'ocean'}


def evaluate_task_multiseed(task_name, model, n_seeds=5, base_seed=RANDOM_SEED,
                            data_dir=DEFAULT_DATA_DIR, device='cuda',
                            batch_size=256, num_workers=4):
    """
    Multi-seed evaluation pipeline for a single task. Extracts embeddings once,
    then re-shuffles the concatenated train+val pool under N different random
    splits and refits the linear probe each time.

    Tasks listed in FIXED_SPLIT_TASKS use a predefined train/val partition and
    fall back to single-shot evaluation (std=0).

    Returns:
        dict with keys {'mean', 'std', 'scores', 'n_seeds'}.
    """
    if task_name in FIXED_SPLIT_TASKS or n_seeds <= 1:
        score = evaluate_task(task_name, model, data_dir, device,
                              batch_size, num_workers)
        return {
            'mean': float(score),
            'std': 0.0,
            'scores': [float(score)],
            'n_seeds': 1,
        }

    # Build the same train/val loaders the published eval uses, then
    # concatenate so we have one pool to re-split.
    train_loader, val_loader, num_classes, task_type = get_dataset(
        task_name, data_dir, batch_size, num_workers
    )

    print(f"\nEvaluating task: {task_name} ({n_seeds} seeds)")
    print(f"  Task type: {task_type}")
    if task_type == 'classification':
        print(f"  Num classes: {num_classes}")

    train_emb, train_labels = extract_embeddings(train_loader, model, device)
    val_emb, val_labels = extract_embeddings(val_loader, model, device)

    full_emb = np.concatenate([train_emb, val_emb], axis=0)
    full_labels = np.concatenate([train_labels, val_labels], axis=0)

    n_total = len(full_labels)
    n_train = int(round(TRAIN_RATIO * n_total))

    print(f"  Pool size: {n_total} (train={n_train}, val={n_total - n_train})")

    scores = []
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        score, _ = evaluate_embeddings(
            full_emb[train_idx], full_labels[train_idx],
            full_emb[val_idx], full_labels[val_idx],
            task_type,
        )
        scores.append(float(score))
        print(f"    seed={seed}: {score:.4f}")

    mean = float(np.mean(scores))
    # Use sample std (ddof=1) to match the standard reporting convention.
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    metric_name = "Accuracy" if task_type == 'classification' else "R²"
    print(f"  {metric_name}: {mean:.4f} ± {std:.4f} (n={n_seeds})")

    return {
        'mean': mean,
        'std': std,
        'scores': scores,
        'n_seeds': n_seeds,
    }


def evaluate_all_tasks(model, data_dir=DEFAULT_DATA_DIR, device='cuda',
                       batch_size=256, num_workers=4, tasks=None,
                       n_seeds=1, base_seed=RANDOM_SEED):
    """
    Evaluate model on all available tasks.

    Args:
        model: Location encoder model
        data_dir: Directory containing evaluation data
        device: Device to run inference on
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        tasks: List of tasks to evaluate (None = all available)
        n_seeds: Number of random train/val splits to evaluate (>=1).
            n_seeds=1 runs a single train/val split.
        base_seed: First seed; subsequent seeds are base_seed+1, base_seed+2, ...

    Returns:
        results: dict mapping task names to either a float (n_seeds=1) or a dict
            with keys {'mean', 'std', 'scores', 'n_seeds'} (n_seeds>1).
    """
    if tasks is None:
        tasks = [
            # Classification tasks
            'biome', 'ecoregion', 'country', 'ocean',
            # Regression tasks
            'temperature', 'housing', 'elevation', 'population',
            # Checkerboard tasks
            'checker_100', 'checker_500', 'checker_1000', 'checker_1500', 'checker_2000',
        ]

    results = {}
    for task in tasks:
        try:
            if n_seeds > 1:
                results[task] = evaluate_task_multiseed(
                    task, model, n_seeds=n_seeds, base_seed=base_seed,
                    data_dir=data_dir, device=device,
                    batch_size=batch_size, num_workers=num_workers,
                )
            else:
                results[task] = evaluate_task(
                    task, model, data_dir, device, batch_size, num_workers
                )
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            results[task] = None

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for task, res in results.items():
        if res is None:
            print(f"  {task:20s}: FAILED")
        elif isinstance(res, dict):
            print(f"  {task:20s}: {res['mean']:.4f} ± {res['std']:.4f} (n={res['n_seeds']})")
        else:
            print(f"  {task:20s}: {res:.4f}")

    return results


# ==============================================================================
# Model Loading
# ==============================================================================

class LocationEncoderWrapper(torch.nn.Module):
    """Adapt the TTE location encoder to the (lon, lat) coordinate convention the
    eval datasets use (the encoder itself takes (lat, lon))."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coords):
        # datasets provide (lon, lat); the encoder expects (lat, lon)
        lat_lon = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
        return self.model.encode(lat_lon)


def load_tte_model(checkpoint_path: str, device: str = 'cuda', normalize: bool = True):
    """Load the TTE location encoder for evaluation.

    Accepts a HuggingFace repo id (e.g. ``MVRL/TTE``) or a local checkpoint, via
    the canonical :func:`tte.load_tte_model`. Returns a wrapper that takes
    ``(lon, lat)`` coordinates and returns L2-normalized location embeddings.
    """
    return LocationEncoderWrapper(_load_tte(checkpoint_path, device=device, normalize=normalize))


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RANGE Evaluation for Location Encoders")

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing evaluation data')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Tasks to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip L2 normalization of embeddings (matches original RANGE eval)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random train/val splits to average over '
                             '(paper protocol; seeds base_seed..base_seed+n-1). '
                             'Output JSON is {mean, std, scores, n_seeds} per task. '
                             'Use --n_seeds 1 for the original fixed-split eval.')
    parser.add_argument('--base_seed', type=int, default=RANDOM_SEED,
                        help='First random seed (default 42; seeds run base_seed..base_seed+n_seeds-1)')

    args = parser.parse_args()

    # Derive output_dir from model name if not specified
    if args.output_dir is None:
        model_path = Path(args.model_path)
        # Use checkpoint filename (without extension) as the eval name
        # e.g., epoch_150_val_loss_5.8219.pt -> epoch_150_val_loss_5.8219
        model_name = model_path.stem
        args.output_dir = f"outputs/eval/{model_name}"

    # Load model
    normalize = not args.no_normalize
    model = load_tte_model(args.model_path, args.device, normalize=normalize)
    if not normalize:
        print("Embedding normalization: DISABLED (matching original RANGE eval)")

    # Run evaluation
    results = evaluate_all_tasks(
        model, args.data_dir, args.device, args.batch_size, args.num_workers,
        args.tasks, n_seeds=args.n_seeds, base_seed=args.base_seed,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_n{args.n_seeds}" if args.n_seeds > 1 else ""
    results_path = os.path.join(args.output_dir, f'eval_results{suffix}.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
