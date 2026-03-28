"""
Denoising Autoencoder for Image Denoising — BSDS300 Dataset
============================================================
Covers all 6 tasks:
  1. Dataset preparation (download + train/val/test splits)
  2. Noise injection (Gaussian & Salt-and-pepper)
  3. Convolutional Denoising Autoencoder architecture
  4. Training with MSE reconstruction loss
  5. Evaluation (MSE, PSNR, SSIM) with visualisation
  6. Experimental study (noise levels × bottleneck sizes)

Kaggle: add your dataset (BSDS300-images.tgz) via “Add data”. The script
auto-finds the tarball under /kaggle/input, extracts to /kaggle/working/BSDS300,
and writes plots to /kaggle/working/outputs.
Optional env: BSDS_TGZ_PATH=/kaggle/input/<your-dataset>/BSDS300-images.tgz
"""

import os
import random
import tarfile
import urllib.request
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# 0.  Global config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# Kaggle: input is read-only; extract dataset and save outputs under /kaggle/working
IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
    BSDS_DIR = WORK_DIR / "BSDS300"
    OUTPUT_DIR = WORK_DIR / "outputs"
else:
    WORK_DIR = _script_dir()
    BSDS_DIR = WORK_DIR / "BSDS300"
    OUTPUT_DIR = WORK_DIR / "outputs"

IMAGES_DIR = BSDS_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BSDS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128          # resize target (square)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1.  Dataset Preparation
# ---------------------------------------------------------------------------
DATASET_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
TGZ_NAME = "BSDS300-images.tgz"


def _find_kaggle_tgz() -> Path | None:
    """Locate BSDS300-images.tgz under any /kaggle/input/<dataset>/ folder."""
    explicit = os.environ.get("BSDS_TGZ_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        print(f"[WARN] BSDS_TGZ_PATH set but not a file: {explicit}")

    kaggle_in = Path("/kaggle/input")
    if not kaggle_in.is_dir():
        return None
    for p in kaggle_in.rglob(TGZ_NAME):
        if p.is_file():
            return p
    return None


def _flatten_extracted_tree():
    """Normalize layout to BSDS_DIR/images/ (train|test|*.jpg)."""
    if IMAGES_DIR.exists() and any(IMAGES_DIR.rglob("*.jpg")):
        return
    nested = BSDS_DIR / "BSDS300" / "images"
    if nested.exists() and not IMAGES_DIR.exists():
        nested.rename(IMAGES_DIR)
        return
    # Some archives use a single top folder with images inside
    for sub in BSDS_DIR.iterdir():
        if sub.is_dir() and (sub / "images").is_dir() and not IMAGES_DIR.exists():
            (sub / "images").rename(IMAGES_DIR)
            return


def prepare_dataset():
    """
    Kaggle: read BSDS300-images.tgz from /kaggle/input (or BSDS_TGZ_PATH),
    extract into BSDS_DIR under working (writable).

    Local: use BSDS300/BSDS300-images.tgz or download from Berkeley if missing.
    """
    if IMAGES_DIR.exists() and any(IMAGES_DIR.rglob("*.jpg")):
        print("[INFO] BSDS300 images already present — skipping extract.")
        _flatten_extracted_tree()
        return

    if IS_KAGGLE:
        tgz_path = _find_kaggle_tgz()
        if tgz_path is None:
            raise FileNotFoundError(
                f"Could not find {TGZ_NAME} under /kaggle/input. "
                "Add your Kaggle dataset, or set env BSDS_TGZ_PATH to the full path."
            )
        print(f"[INFO] Kaggle — using tarball: {tgz_path}")
    else:
        tgz_path = BSDS_DIR / TGZ_NAME
        if not tgz_path.is_file():
            BSDS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Downloading BSDS300 images from\n       {DATASET_URL}")
            urllib.request.urlretrieve(DATASET_URL, str(tgz_path))
            print("[INFO] Download complete.")

    print("[INFO] Extracting archive …")
    _extract_tarball(tgz_path, BSDS_DIR)
    print("[INFO] Extraction complete.")

    _flatten_extracted_tree()


def _extract_tarball(archive_path: Path, dest_dir: Path) -> None:
    """
    Open .tgz as gzip-compressed tar, or fall back to uncompressed tar.
    Raises a clear error if the file is HTML, Git LFS text, or corrupt.
    """
    size = archive_path.stat().st_size
    if size < 500_000:
        print(f"[WARN] Archive is only {size} bytes — expected ~tens of MB for BSDS300 images.")

    modes = ("r:gz", "r")  # gzip tar, then plain tar (some uploads mislabel .tar as .tgz)
    last_err: Exception | None = None
    for mode in modes:
        try:
            with tarfile.open(archive_path, mode) as tar:
                tar.extractall(str(dest_dir))
            if mode == "r":
                print("[INFO] Opened as uncompressed tar (file was not gzip despite .tgz name).")
            return
        except tarfile.ReadError as e:
            last_err = e
            continue
        except OSError as e:
            last_err = e
            continue

    with open(archive_path, "rb") as f:
        head = f.read(256)
    magic = head[:16]
    text_preview = head[:120].decode("utf-8", errors="replace").strip()

    hint = (
        "\n  Common causes:\n"
        "  • File is not the real Berkeley tarball (e.g. HTML save-as, or wrong upload).\n"
        "  • Git LFS pointer text was uploaded instead of the binary.\n"
        "  • Corrupt or truncated download — re-download from Berkeley and re-upload to Kaggle.\n"
        f"  Official URL: {DATASET_URL}"
    )
    raise RuntimeError(
        f"Could not read archive as gzip-tar or plain tar: {last_err}\n"
        f"  Path: {archive_path}\n"
        f"  First bytes (hex): {magic.hex()}\n"
        f"  First bytes (ascii preview): {text_preview!r}"
        f"{hint}"
    ) from last_err


def _read_ids(txt_path: Path) -> list[str]:
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]


def load_image_paths() -> dict[str, list[Path]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} of image paths."""
    train_dir = IMAGES_DIR / "train"
    test_dir = IMAGES_DIR / "test"
    train_paths: list[Path] = []
    test_paths: list[Path] = []

    if train_dir.is_dir() and test_dir.is_dir():
        # Official BSDS300 tarball: images/train (200), images/test (100)
        train_paths = sorted(train_dir.glob("*.jpg"))
        test_paths = sorted(test_dir.glob("*.jpg"))
    else:
        # Fallback: iids_*.txt (local copy or extra files in Kaggle dataset)
        def _find_id_file(name: str) -> Path | None:
            for d in (BSDS_DIR, _script_dir() / "BSDS300"):
                p = d / name
                if p.is_file():
                    return p
            if Path("/kaggle/input").is_dir():
                for p in Path("/kaggle/input").rglob(name):
                    if p.is_file():
                        return p
            return None

        train_txt = _find_id_file("iids_train.txt")
        test_txt = _find_id_file("iids_test.txt")
        if train_txt is None or test_txt is None:
            raise FileNotFoundError(
                "Expected images/train and images/test under extracted BSDS300, "
                "or iids_train.txt / iids_test.txt beside the dataset."
            )
        train_ids = _read_ids(train_txt)
        test_ids = _read_ids(test_txt)
        id_to_path: dict[str, Path] = {}
        for p in IMAGES_DIR.rglob("*.jpg"):
            id_to_path[p.stem] = p
        train_paths = [id_to_path[i] for i in train_ids if i in id_to_path]
        test_paths = [id_to_path[i] for i in test_ids if i in id_to_path]

    if not train_paths or not test_paths:
        raise RuntimeError(
            f"No images found. IMAGES_DIR={IMAGES_DIR} "
            f"(train={len(train_paths)}, test={len(test_paths)})."
        )

    # Split training set → 80% train + 20% validation
    random.shuffle(train_paths)
    val_split = int(0.2 * len(train_paths))
    val_paths = train_paths[:val_split]
    train_paths = train_paths[val_split:]

    print(f"[DATA] Train: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")
    return {"train": train_paths, "val": val_paths, "test": test_paths}


def verify_dataset_loaded(paths: dict[str, list[Path]], sample_read: int = 3) -> bool:
    """
    Sanity-check that paths point to real files and at least one image opens.
    Call after load_image_paths() (or use the printed [DATA] line + this in a notebook).
    """
    ok = True
    for split, plist in paths.items():
        n = len(plist)
        missing = sum(1 for p in plist if not p.is_file())
        print(f"[CHECK] {split}: {n} paths, {missing} missing on disk")
        if missing:
            ok = False
        for p in plist[:sample_read]:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                print(f"[CHECK] FAIL read {p}: {e}")
                ok = False
    # One real load (verify() is not enough for some codecs)
    if paths["train"]:
        try:
            im = Image.open(paths["train"][0]).convert("RGB")
            w, h = im.size
            print(f"[CHECK] Sample train image OK: {paths['train'][0].name}  size={w}x{h}")
        except Exception as e:
            print(f"[CHECK] FAIL open sample: {e}")
            ok = False
    print("[CHECK] Dataset OK" if ok else "[CHECK] Dataset has problems — fix paths/extract")
    return ok

# ---------------------------------------------------------------------------
# 2.  Noise Injection
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Add zero-mean Gaussian noise with given sigma (image in [0,1])."""
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Add salt-and-pepper noise (fraction = amount)."""
    out = img.copy()
    n_pixels = img.size
    # Salt
    n_salt = int(n_pixels * amount / 2)
    coords = tuple(np.random.randint(0, d, n_salt) for d in img.shape)
    out[coords] = 1.0
    # Pepper
    n_pepper = int(n_pixels * amount / 2)
    coords = tuple(np.random.randint(0, d, n_pepper) for d in img.shape)
    out[coords] = 0.0
    return out


NOISE_FN = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper_noise,
}

# ---------------------------------------------------------------------------
#  PyTorch Dataset
# ---------------------------------------------------------------------------

class NoisyImageDataset(Dataset):
    def __init__(self, paths: list[Path], noise_type: str = "gaussian",
                 noise_level: float = 0.1, img_size: int = IMG_SIZE):
        self.paths = paths
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # -> [C,H,W] in [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.transform(img)                          # [3,H,W]
        clean_np = clean.permute(1, 2, 0).numpy()           # [H,W,3]
        noisy_np = NOISE_FN[self.noise_type](clean_np, self.noise_level)
        noisy = torch.from_numpy(noisy_np).permute(2, 0, 1) # [3,H,W]
        return noisy, clean

# ---------------------------------------------------------------------------
# 3.  Denoising Autoencoder Architecture
# ---------------------------------------------------------------------------

class DenoisingAutoencoder(nn.Module):
    """
    Fully-convolutional DAE.

    Encoder:  3 → 64 → 128 → bottleneck   (stride-2 convs, BN + ReLU)
    Decoder:  bottleneck → 128 → 64 → 3    (transposed convs, BN + ReLU, Sigmoid output)

    Bottleneck channel count is configurable for the experimental study.
    """
    def __init__(self, bottleneck: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, bottleneck, 3, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------------------------------------------------------------------------
# 4.  Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            out = model(noisy)
            loss = criterion(out, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * noisy.size(0)
        train_loss = running / len(train_loader.dataset)

        # --- validate ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                out = model(noisy)
                running += criterion(out, clean).item() * noisy.size(0)
        val_loss = running / len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    return history

# ---------------------------------------------------------------------------
    # 5.  Evaluation & Visualization
# ---------------------------------------------------------------------------

def evaluate(model, loader) -> dict:
    """Return average MSE, PSNR, SSIM over the loader."""
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            out = model(noisy)
            for i in range(clean.size(0)):
                c = clean[i].cpu().numpy().transpose(1, 2, 0)
                r = out[i].cpu().numpy().transpose(1, 2, 0)
                mse_val = np.mean((c - r) ** 2)
                mse_list.append(mse_val)
                psnr_list.append(psnr(c, r, data_range=1.0))
                ssim_list.append(ssim(c, r, data_range=1.0, channel_axis=2))
    return {
        "MSE": np.mean(mse_list),
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
    }


def visualize_samples(model, loader, tag: str, n: int = 5):
    """Save a figure showing clean / noisy / reconstructed triplets."""
    model.eval()
    noisy_batch, clean_batch = next(iter(loader))
    noisy_batch, clean_batch = noisy_batch.to(DEVICE), clean_batch.to(DEVICE)
    with torch.no_grad():
        recon_batch = model(noisy_batch)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    row_labels = ["Clean", "Noisy", "Denoised"]
    for i in range(n):
        imgs = [
            clean_batch[i].cpu().numpy().transpose(1, 2, 0),
            noisy_batch[i].cpu().numpy().transpose(1, 2, 0),
            recon_batch[i].cpu().numpy().transpose(1, 2, 0),
        ]
        for r, img in enumerate(imgs):
            axes[r, i].imshow(np.clip(img, 0, 1))
            axes[r, i].axis("off")
            if i == 0:
                axes[r, i].set_ylabel(row_labels[r], fontsize=14, rotation=90,
                                      labelpad=10)
    plt.suptitle(tag, fontsize=16, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / f"vis_{tag.replace(' ', '_').replace('=','')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [VIS] Saved → {path}")


def plot_loss_curves(history: dict, tag: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Loss Curve — {tag}")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / f"loss_{tag.replace(' ', '_').replace('=','')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 6.  Experimental Study
# ---------------------------------------------------------------------------
NOISE_LEVELS = {
    "gaussian": [0.05, 0.10, 0.20],
    "salt_pepper": [0.02, 0.05, 0.10],
}
BOTTLENECKS = [64, 128, 256]
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3


def run_experiment(paths: dict):
    """Full experimental study: noise type × noise level × bottleneck."""
    results = []

    for noise_type in NOISE_FN:
        for noise_lv, bottleneck in product(NOISE_LEVELS[noise_type], BOTTLENECKS):
            tag = f"{noise_type}  σ={noise_lv}  bneck={bottleneck}"
            print(f"\n{'='*60}\n[EXP] {tag}\n{'='*60}")

            train_ds = NoisyImageDataset(paths["train"], noise_type, noise_lv)
            val_ds = NoisyImageDataset(paths["val"], noise_type, noise_lv)
            test_ds = NoisyImageDataset(paths["test"], noise_type, noise_lv)

            # Kaggle notebooks: keep workers at 0 to avoid multiprocessing issues
            _nw = 0
            _pm = bool(torch.cuda.is_available())
            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                      num_workers=_nw, pin_memory=_pm)
            val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                                    num_workers=_nw, pin_memory=_pm)
            test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                                     num_workers=_nw, pin_memory=_pm)

            model = DenoisingAutoencoder(bottleneck=bottleneck)
            history = train_model(model, train_loader, val_loader,
                                  epochs=EPOCHS, lr=LR)
            plot_loss_curves(history, tag)

            metrics = evaluate(model, test_loader)
            print(f"  [TEST] MSE={metrics['MSE']:.6f}  "
                  f"PSNR={metrics['PSNR']:.2f} dB  SSIM={metrics['SSIM']:.4f}")

            visualize_samples(model, test_loader, tag)

            results.append({
                "noise_type": noise_type,
                "noise_level": noise_lv,
                "bottleneck": bottleneck,
                "MSE": metrics["MSE"],
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
            })

    return results


def summarize_results(results: list[dict]):
    """Print summary table and save comparison bar charts."""
    # ---- Table ----
    header = f"{'Noise Type':<14} {'Level':<8} {'Bottleneck':<11} {'MSE':<10} {'PSNR (dB)':<11} {'SSIM':<8}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"{r['noise_type']:<14} {r['noise_level']:<8.2f} "
              f"{r['bottleneck']:<11d} {r['MSE']:<10.6f} "
              f"{r['PSNR']:<11.2f} {r['SSIM']:<8.4f}")
    print(sep)

    # ---- Grouped bar charts per noise type ----
    for noise_type in NOISE_FN:
        subset = [r for r in results if r["noise_type"] == noise_type]
        if not subset:
            continue

        levels = sorted(set(r["noise_level"] for r in subset))
        bnecks = sorted(set(r["bottleneck"] for r in subset))
        x = np.arange(len(levels))
        width = 0.25

        for metric in ("PSNR", "SSIM"):
            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, bn in enumerate(bnecks):
                vals = [next(r[metric] for r in subset
                             if r["noise_level"] == lv and r["bottleneck"] == bn)
                        for lv in levels]
                ax.bar(x + idx * width, vals, width, label=f"bneck={bn}")
            ax.set_xticks(x + width)
            ax.set_xticklabels([str(l) for l in levels])
            ax.set_xlabel("Noise Level")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Noise Level — {noise_type}")
            ax.legend()
            plt.tight_layout()
            path = OUTPUT_DIR / f"bar_{noise_type}_{metric}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[PLOT] Saved → {path}")

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print(f"[DEVICE] {DEVICE}")

    if IS_KAGGLE:
        print("[INFO] Running on Kaggle — input: /kaggle/input, extract & outputs: /kaggle/working")

    # Task 1 — Dataset
    prepare_dataset()
    paths = load_image_paths()
    verify_dataset_loaded(paths)

    # Tasks 2–5 — Quick demo with default settings
    print("\n" + "=" * 60)
    print("[DEMO] Training baseline model (Gaussian σ=0.1, bottleneck=256)")
    print("=" * 60)

    demo_train_ds = NoisyImageDataset(paths["train"], "gaussian", 0.1)
    demo_val_ds = NoisyImageDataset(paths["val"], "gaussian", 0.1)
    demo_test_ds = NoisyImageDataset(paths["test"], "gaussian", 0.1)

    _nw = 0
    _pm = bool(torch.cuda.is_available())
    train_loader = DataLoader(demo_train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=_nw, pin_memory=_pm)
    val_loader = DataLoader(demo_val_ds, BATCH_SIZE, shuffle=False,
                            num_workers=_nw, pin_memory=_pm)
    test_loader = DataLoader(demo_test_ds, BATCH_SIZE, shuffle=False,
                             num_workers=_nw, pin_memory=_pm)

    model = DenoisingAutoencoder(bottleneck=256)
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    plot_loss_curves(history, "baseline_gaussian_0.1_bneck256")

    metrics = evaluate(model, test_loader)
    print(f"\n[BASELINE TEST] MSE={metrics['MSE']:.6f}  "
          f"PSNR={metrics['PSNR']:.2f} dB  SSIM={metrics['SSIM']:.4f}")
    visualize_samples(model, test_loader, "Baseline Gaussian sigma=0.1 bneck=256")

    # Task 6 — Full experimental study
    print("\n" + "=" * 60)
    print("[EXP] Starting full experimental study …")
    print("=" * 60)
    results = run_experiment(paths)
    summarize_results(results)

    print(f"\n[DONE] All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
