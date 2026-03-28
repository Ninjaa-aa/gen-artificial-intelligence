"""
Variational Autoencoder (VAE) — Fashion-MNIST
===============================================
Covers all 6 tasks:
  1. Dataset preparation  (load CSV / torchvision, normalise, train/val/test)
  2. VAE architecture     (encoder → μ,σ → reparam → decoder)
  3. Loss function        (BCE reconstruction + KL divergence)
  4. Training             (monitor recon + KL per epoch)
  5. Generation & visualisation (random sampling + reconstruction grids)
  6. Experimental study   (latent dim sweep with tables & figures)

Kaggle:  the script auto-detects CSV files under /kaggle/input.
         Set FASHION_CSV_DIR if yours lives in a non-standard path.
Local:   falls back to torchvision.datasets.FashionMNIST (auto-download).
"""

import os
import random
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Global config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_KAGGLE = Path("/kaggle/input").exists()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()
OUTPUT_DIR = WORK_DIR / "outputs_vae"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMG_SIZE = 28
FLAT_DIM = IMG_SIZE * IMG_SIZE  # 784
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataset Preparation
# ──────────────────────────────────────────────────────────────────────────────

def _find_csv(name_hint: str) -> Path | None:
    """Search /kaggle/input (recursively) for a CSV whose name contains *name_hint*."""
    if not IS_KAGGLE:
        return None
    for p in Path("/kaggle/input").rglob("*.csv"):
        if name_hint in p.name.lower():
            return p
    return None


class FashionCSVDataset(Dataset):
    """Load Fashion-MNIST from a CSV where col-0 = label, cols 1–784 = pixels."""
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        self.labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
        pixels = df.iloc[:, 1:].values.astype(np.float32) / 255.0
        self.images = torch.tensor(pixels).view(-1, 1, IMG_SIZE, IMG_SIZE)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_datasets() -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns (train_ds, val_ds, test_ds).
    Kaggle path priority:
      1. FASHION_CSV_DIR env var
      2. /kaggle/input/datasets/hammad5114/fashion-mnist/fashionmnist/
      3. Recursive search under /kaggle/input for *train*.csv / *test*.csv
    Local fallback: torchvision FashionMNIST (auto-download).
    """
    train_csv: Path | None = None
    test_csv: Path | None = None

    # --- Kaggle CSV discovery ---
    explicit = os.environ.get("FASHION_CSV_DIR", "").strip()
    candidate_dirs = []
    if explicit:
        candidate_dirs.append(Path(explicit))
    candidate_dirs += [
        Path("/kaggle/input/datasets/hammad5114/fashion-mnist/fashionmnist"),
        Path("/kaggle/input/fashion-mnist/fashionmnist"),
        Path("/kaggle/input/fashionmnist"),
        Path("/kaggle/input/fashion-mnist"),
    ]

    for d in candidate_dirs:
        if not d.is_dir():
            continue
        csvs = sorted(d.glob("*.csv"))
        for c in csvs:
            n = c.name.lower()
            if "train" in n and train_csv is None:
                train_csv = c
            elif "test" in n and test_csv is None:
                test_csv = c
        if train_csv and test_csv:
            break

    # Fallback: recursive search
    if train_csv is None and IS_KAGGLE:
        train_csv = _find_csv("train")
    if test_csv is None and IS_KAGGLE:
        test_csv = _find_csv("test")

    if train_csv and test_csv:
        print(f"[DATA] Loading CSVs:\n  train → {train_csv}\n  test  → {test_csv}")
        full_train = FashionCSVDataset(train_csv)
        test_ds = FashionCSVDataset(test_csv)
    else:
        print("[DATA] CSVs not found — using torchvision FashionMNIST (auto-download).")
        tfm = T.Compose([T.ToTensor()])
        data_root = str(WORK_DIR / "data")
        full_train = torchvision.datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=tfm)
        test_ds = torchvision.datasets.FashionMNIST(
            root=data_root, train=False, download=True, transform=tfm)

    # 50 000 train + 10 000 val from the 60 000 training samples
    val_size = 10_000
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"[DATA] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Variational Autoencoder Architecture
# ──────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    Convolutional VAE for 28×28 grayscale images.

    Encoder
    -------
    Conv(1→32, 3, s2, p1) → BN → ReLU   → 14×14
    Conv(32→64, 3, s2, p1) → BN → ReLU  → 7×7
    Flatten → FC → μ  (latent_dim)
                → logσ² (latent_dim)

    Reparameterisation:  z = μ + σ · ε ,  ε ~ N(0, I)

    Decoder
    -------
    FC(latent_dim → 64·7·7) → ReLU → Reshape
    ConvT(64→32, 3, s2, p1, op1) → BN → ReLU  → 14×14
    ConvT(32→1, 3, s2, p1, op1) → Sigmoid       → 28×28
    """
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 64, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Loss Function  (Reconstruction BCE + KL Divergence)
# ──────────────────────────────────────────────────────────────────────────────

def vae_loss(recon_x, x, mu, logvar):
    """
    recon_loss : pixel-wise binary cross-entropy (summed over pixels, averaged over batch)
    kl_loss    : -0.5 * Σ(1 + log σ² - μ² - σ²), averaged over batch
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + kl, bce, kl

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train_vae(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    history = {k: [] for k in
               ("train_loss", "train_recon", "train_kl",
                "val_loss", "val_recon", "val_kl")}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        t_loss = t_recon = t_kl = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            recon, mu, logvar = model(imgs)
            loss, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = imgs.size(0)
            t_loss += loss.item() * b
            t_recon += recon_l.item() * b
            t_kl += kl_l.item() * b
        n = len(train_loader.dataset)
        history["train_loss"].append(t_loss / n)
        history["train_recon"].append(t_recon / n)
        history["train_kl"].append(t_kl / n)

        # --- validate ---
        model.eval()
        v_loss = v_recon = v_kl = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                recon, mu, logvar = model(imgs)
                loss, recon_l, kl_l = vae_loss(recon, imgs, mu, logvar)
                b = imgs.size(0)
                v_loss += loss.item() * b
                v_recon += recon_l.item() * b
                v_kl += kl_l.item() * b
        nv = len(val_loader.dataset)
        history["val_loss"].append(v_loss / nv)
        history["val_recon"].append(v_recon / nv)
        history["val_kl"].append(v_kl / nv)

        scheduler.step(v_loss / nv)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train[total={history['train_loss'][-1]:.2f} "
                  f"recon={history['train_recon'][-1]:.2f} "
                  f"kl={history['train_kl'][-1]:.2f}]  "
                  f"Val[total={history['val_loss'][-1]:.2f} "
                  f"recon={history['val_recon'][-1]:.2f} "
                  f"kl={history['val_kl'][-1]:.2f}]")

    return history

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Generation & Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _to_grid(imgs_tensor, nrow=10):
    """Return a numpy HW image from a batch of [N,1,28,28] tensors."""
    from torchvision.utils import make_grid
    grid = make_grid(imgs_tensor.cpu(), nrow=nrow, padding=2, normalize=False)
    return grid.permute(1, 2, 0).numpy().squeeze()


def visualize_reconstruction(model, loader, tag: str, n: int = 10):
    """Side-by-side original vs reconstruction."""
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:n].to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(imgs)

    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4))
    for i in range(n):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Recon", fontsize=12)
    plt.suptitle(f"Reconstruction — {tag}", fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / f"recon_{tag.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] Saved → {path}")


def generate_samples(model, tag: str, nrow: int = 10, ncol: int = 10):
    """Sample z ~ N(0,I) and decode."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(nrow * ncol, model.latent_dim, device=DEVICE)
        samples = model.decode(z).cpu()

    fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.5 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].imshow(samples[i * ncol + j].squeeze(), cmap="gray")
            axes[i, j].axis("off")
    plt.suptitle(f"Generated Samples — {tag}", fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / f"gen_{tag.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] Saved → {path}")


def plot_latent_space_2d(model, loader, tag: str):
    """If latent_dim == 2, scatter-plot the encoded test set coloured by class."""
    if model.latent_dim != 2:
        return
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            mu, _ = model.encode(imgs.to(DEVICE))
            zs.append(mu.cpu())
            ys.append(labels)
    z = torch.cat(zs).numpy()
    y = torch.cat(ys).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=2, alpha=0.6)
    cbar = fig.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(CLASS_NAMES)
    ax.set_title(f"Latent Space (2-D) — {tag}")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    plt.tight_layout()
    path = OUTPUT_DIR / f"latent2d_{tag.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] Saved → {path}")


def plot_loss_curves(history: dict, tag: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(history["train_loss"], label="Train Total")
    ax1.plot(history["val_loss"], label="Val Total")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Total Loss")
    ax1.legend()

    ax2.plot(history["train_recon"], label="Train Recon")
    ax2.plot(history["val_recon"], label="Val Recon")
    ax2.plot(history["train_kl"], label="Train KL", linestyle="--")
    ax2.plot(history["val_kl"], label="Val KL", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Recon & KL")
    ax2.legend()

    plt.suptitle(f"Training Curves — {tag}", fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / f"curves_{tag.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Experimental Study — latent dimension sweep
# ──────────────────────────────────────────────────────────────────────────────
LATENT_DIMS = [2, 8, 16, 32, 64]


def evaluate_reconstruction(model, loader) -> dict:
    """MSE and BCE between originals and reconstructions (per-pixel averages)."""
    model.eval()
    total_mse = total_bce = 0.0
    n = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            recon, mu, logvar = model(imgs)
            total_mse += F.mse_loss(recon, imgs, reduction="sum").item()
            total_bce += F.binary_cross_entropy(recon, imgs, reduction="sum").item()
            n += imgs.size(0)
    return {"MSE": total_mse / (n * FLAT_DIM), "BCE": total_bce / (n * FLAT_DIM)}


def run_experiment(train_ds, val_ds, test_ds):
    results = []

    for ldim in LATENT_DIMS:
        tag = f"ldim{ldim}"
        print(f"\n{'='*60}\n[EXP] Latent dim = {ldim}\n{'='*60}")

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=torch.cuda.is_available())

        model = VAE(latent_dim=ldim)
        history = train_vae(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
        plot_loss_curves(history, tag)

        metrics = evaluate_reconstruction(model, test_loader)
        print(f"  [TEST] Pixel MSE={metrics['MSE']:.6f}  Pixel BCE={metrics['BCE']:.6f}")

        visualize_reconstruction(model, test_loader, tag)
        generate_samples(model, tag)
        plot_latent_space_2d(model, test_loader, tag)

        results.append({
            "latent_dim": ldim,
            "test_MSE": metrics["MSE"],
            "test_BCE": metrics["BCE"],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_kl": history["train_kl"][-1],
            "final_val_kl": history["val_kl"][-1],
        })

    return results


def summarize_results(results: list[dict]):
    header = (f"{'Latent Dim':<12} {'Test MSE':<12} {'Test BCE':<12} "
              f"{'Train Loss':<12} {'Val Loss':<12} {'Train KL':<12} {'Val KL':<12}")
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"{r['latent_dim']:<12d} {r['test_MSE']:<12.6f} {r['test_BCE']:<12.6f} "
              f"{r['final_train_loss']:<12.2f} {r['final_val_loss']:<12.2f} "
              f"{r['final_train_kl']:<12.2f} {r['final_val_kl']:<12.2f}")
    print(sep)

    # --- Bar charts ---
    dims = [r["latent_dim"] for r in results]
    x = np.arange(len(dims))

    for metric, label in [("test_MSE", "Test Pixel MSE"), ("test_BCE", "Test Pixel BCE")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = [r[metric] for r in results]
        ax.bar(x, vals, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Latent Dimension")
        plt.tight_layout()
        path = OUTPUT_DIR / f"bar_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Saved → {path}")

    # KL comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.15, [r["final_train_kl"] for r in results], 0.3, label="Train KL")
    ax.bar(x + 0.15, [r["final_val_kl"] for r in results], 0.3, label="Val KL")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence vs Latent Dimension")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / f"bar_kl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"[DEVICE] {DEVICE}")
    if IS_KAGGLE:
        print("[INFO] Running on Kaggle")

    # Task 1
    train_ds, val_ds, test_ds = load_datasets()

    # Tasks 2–5  — baseline with latent_dim=16
    print("\n" + "=" * 60)
    print("[DEMO] Baseline VAE  (latent_dim=16)")
    print("=" * 60)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    model = VAE(latent_dim=16)
    history = train_vae(model, train_loader, val_loader)
    plot_loss_curves(history, "baseline_ldim16")

    metrics = evaluate_reconstruction(model, test_loader)
    print(f"\n[BASELINE TEST] Pixel MSE={metrics['MSE']:.6f}  "
          f"Pixel BCE={metrics['BCE']:.6f}")

    visualize_reconstruction(model, test_loader, "baseline_ldim16")
    generate_samples(model, "baseline_ldim16")

    # Task 6 — Full experiment
    print("\n" + "=" * 60)
    print("[EXP] Starting latent dimension sweep …")
    print("=" * 60)
    results = run_experiment(train_ds, val_ds, test_ds)
    summarize_results(results)

    print(f"\n[DONE] All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
