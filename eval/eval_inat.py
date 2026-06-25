import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# repo root (for `import tte`) + eval/ (for `from eval_range`) on path
for _p in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_range import load_tte_model

DEFAULT_DATA_DIR = os.environ.get("INAT_DATA", "data/inat")   # or --data_dir
NUM_CLASSES = 8142
DEFAULT_DEVICE = "cuda"


# ==============================================================================
# Dataset
# ==============================================================================

class INat2018EmbeddingDataset(Dataset):
    """Dataset that serves pre-computed location embeddings and class labels."""

    def __init__(self, embeddings, classes):
        self.embeddings = embeddings.astype(np.float32)
        self.classes = classes.astype(np.int64)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.classes[idx]


# ==============================================================================
# Linear Probe Model
# ==============================================================================

class INat2018Probe(pl.LightningModule):
    """Linear probe that maps location embeddings to species predictions."""

    def __init__(self, embed_dim, num_classes=NUM_CLASSES, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self(x.float()).sigmoid()
        loss = -torch.log(1 - y_hat + 1e-6)
        loss[torch.arange(len(y)), y] = -2048 * torch.log(y_hat[torch.arange(len(y)), y] + 1e-6)
        return loss

    def training_step(self, batch):
        loss = self.shared_step(batch).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = self.shared_step(batch).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ==============================================================================
# Embedding Generation
# ==============================================================================

def generate_embeddings(model, lon, lat, device='cuda', batch_size=4096):
    """Generate location embeddings for all (lon, lat) pairs using TTE model."""
    model.eval()
    model = model.to(device)

    coords = np.stack([lon, lat], axis=1)  # (N, 2) as (lon, lat)
    coords_tensor = torch.from_numpy(coords).double()

    all_embeddings = []
    for i in tqdm(range(0, len(coords_tensor), batch_size), desc="Generating embeddings"):
        batch = coords_tensor[i:i + batch_size].to(device)
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)


# ==============================================================================
# Main
# ==============================================================================

def _train_and_evaluate(train_embeddings, train_classes, val_embeddings,
                        val_classes, val_predictions, device, max_epochs,
                        batch_size, lr, output_dir=None, save_probe=False):
    """Train one probe + return CNN / probe / combined top-k accuracies."""
    embed_dim = train_embeddings.shape[1]

    train_ds = INat2018EmbeddingDataset(train_embeddings, train_classes)
    val_ds = INat2018EmbeddingDataset(val_embeddings, val_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    probe = INat2018Probe(embed_dim=embed_dim, num_classes=NUM_CLASSES, lr=lr)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator='gpu' if 'cuda' in device else 'cpu',
        default_root_dir=output_dir,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(probe, train_loader, val_loader)

    if save_probe and output_dir is not None:
        probe_path = os.path.join(output_dir, 'inat_probe.ckpt')
        trainer.save_checkpoint(probe_path)
        print(f"  Saved probe to {probe_path}")

    probe.eval()
    with torch.no_grad():
        val_emb_tensor = torch.tensor(val_embeddings).float()
        location_probs = probe(val_emb_tensor).sigmoid().numpy()

    combined_preds = location_probs * val_predictions

    def top_k_accuracy(preds, labels, k):
        top_k_indices = np.argsort(preds, axis=1)[:, -k:]
        return float(np.mean([labels[i] in top_k_indices[i] for i in range(len(labels))]))

    ks = [1, 3, 5, 10]
    return {
        'cnn':      {k: top_k_accuracy(val_predictions, val_classes, k) for k in ks},
        'probe':    {k: top_k_accuracy(location_probs, val_classes, k) for k in ks},
        'combined': {k: top_k_accuracy(combined_preds, val_classes, k) for k in ks},
        'embed_dim': embed_dim,
    }


def _run_single(args, train_lon, train_lat, train_classes,
                val_lon, val_lat, val_classes, val_predictions):
    """Default mode: evaluate the model once with all sites."""
    normalize = not args.no_normalize
    tte_model = load_tte_model(args.model_path, args.device, normalize=normalize)

    print("\nGenerating TTE embeddings for train set...")
    train_embeddings = generate_embeddings(tte_model, train_lon, train_lat, args.device)
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print("Generating TTE embeddings for val set...")
    val_embeddings = generate_embeddings(tte_model, val_lon, val_lat, args.device)
    print(f"  Val embeddings shape: {val_embeddings.shape}")

    emb_path = os.path.join(args.output_dir, 'embeddings.npz')
    np.savez(emb_path,
             train_embeddings=train_embeddings, train_classes=train_classes,
             val_embeddings=val_embeddings, val_classes=val_classes)
    print(f"  Saved embeddings to {emb_path}")

    scores = _train_and_evaluate(
        train_embeddings, train_classes, val_embeddings, val_classes, val_predictions,
        device=args.device, max_epochs=args.max_epochs, batch_size=args.batch_size,
        lr=args.lr, output_dir=args.output_dir, save_probe=True,
    )

    ks = [1, 3, 5, 10]
    print(f"\n{'=' * 60}\niNat2018 RESULTS (Top-k Accuracy)\n{'=' * 60}")
    print(f"{'Method':<25} {'top-1':>8} {'top-3':>8} {'top-5':>8} {'top-10':>8}")
    print('-' * 60)
    for name, key in [('Img (CNN only)', 'cnn'),
                      ('Location probe only', 'probe'),
                      ('Img + Location', 'combined')]:
        print(f"{name:<25} "
              + " ".join(f"{scores[key][k]*100:>7.1f}%" for k in ks))

    results = {
        'cnn':      {f'top_{k}': scores['cnn'][k] for k in ks},
        'probe':    {f'top_{k}': scores['probe'][k] for k in ks},
        'combined': {f'top_{k}': scores['combined'][k] for k in ks},
        'embed_dim': scores['embed_dim'],
        'max_epochs': args.max_epochs,
        'lr': args.lr,
        'normalize': normalize,
    }
    results_path = os.path.join(args.output_dir, 'inat_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="iNat2018 Evaluation for TTE")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to TTE model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing inat2018_{train,val}.npz')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help='Device (cuda or cpu)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Max epochs for linear probe training')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for probe training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for probe training')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results and probe checkpoint')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip L2 normalization of TTE embeddings')
    args = parser.parse_args()

    # Derive output_dir from model name if not specified
    if args.output_dir is None:
        model_name = Path(args.model_path).stem
        args.output_dir = f"outputs/eval_inat/{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load iNat2018 data once (shared across both modes)
    train_data = np.load(os.path.join(args.data_dir, 'inat2018_train.npz'))
    val_data = np.load(os.path.join(args.data_dir, 'inat2018_val.npz'))
    train_lon, train_lat, train_classes = train_data['lon'], train_data['lat'], train_data['classes']
    val_lon, val_lat, val_classes = val_data['lon'], val_data['lat'], val_data['classes']
    val_predictions = val_data['prediction']
    print(f"Train: {len(train_classes)} samples")
    print(f"Val: {len(val_classes)} samples, predictions shape: {val_predictions.shape}")

    _run_single(args, train_lon, train_lat, train_classes,
                val_lon, val_lat, val_classes, val_predictions)


if __name__ == '__main__':
    main()
