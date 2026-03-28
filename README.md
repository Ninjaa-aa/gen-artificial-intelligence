# Generative AI

**Hammad Zahid** 
---

## Repository Structure

```
A02/
├── denoising-autoencoder/
│   ├── code/
│   │   ├── denoising_autoencoder.py          # Standalone Python script
│   │   └── denoising-autoencoders.ipynb      # Kaggle notebook
│   ├── data/
│   │   └── BSDS300/
│   │       ├── iids_train.txt                # 200 training image IDs
│   │       └── iids_test.txt                 # 100 test image IDs
│   └── outputs/                              # All generated plots & visualisations
│       ├── bar_gaussian_PSNR.png
│       ├── bar_gaussian_SSIM.png
│       ├── bar_salt_pepper_PSNR.png
│       ├── bar_salt_pepper_SSIM.png
│       ├── loss_*.png                        # 19 loss-curve plots
│       └── vis_*.png                         # 19 clean/noisy/denoised triplets
├── vae-autoencoder/
│   ├── code/
│   │   ├── vae_fashion_mnist.py              # Standalone Python script
│   │   └── variational-autoencoders.ipynb    # Kaggle notebook
│   └── outputs_vae/                          # All generated plots & visualisations
│       ├── bar_test_MSE.png
│       ├── bar_test_BCE.png
│       ├── bar_kl.png
│       ├── curves_*.png                      # 6 training-curve plots
│       ├── recon_*.png                       # 7 reconstruction grids
│       ├── gen_*.png                         # 6 generated-sample grids
│       └── latent2d_ldim2.png                # 2-D latent-space scatter
├── prompts_used_per_question.txt             # All prompts used (deliverable)
├── image.png                                 # FAST-NUCES logo
└── README.md                                 # This file
```

---

## Question 1 — Image Denoising using Denoising Autoencoder

### 1.1 Problem Statement

Design and implement a Denoising Autoencoder (DAE) capable of reconstructing clean images from noisy inputs using the **Berkeley Segmentation Dataset (BSDS300)**.

### 1.2 Dataset

| Property | Value |
|---|---|
| Source | [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/) |
| Total images | 300 (200 train + 100 test) |
| Split used | 160 train / 40 validation / 100 test |
| Preprocessing | Resize to 128 x 128, normalise to [0, 1] |

The BSDS300-images.tgz archive is uploaded as a Kaggle dataset; the script auto-detects and extracts it under `/kaggle/working/BSDS300`.

### 1.3 Noise Injection

Two noise types are applied on-the-fly (different noise each epoch = implicit augmentation):

| Noise Type | Parameter | Levels Tested |
|---|---|---|
| **Gaussian** | sigma (std dev) | 0.05, 0.10, 0.20 |
| **Salt-and-Pepper** | amount (fraction) | 0.02, 0.05, 0.10 |

### 1.4 Model Architecture

Fully-convolutional DAE with configurable bottleneck:

```
Encoder:
  Conv2d(3 → 64, k=3, s=2, p=1) → BatchNorm → ReLU     # 128 → 64
  Conv2d(64 → 128, k=3, s=2, p=1) → BatchNorm → ReLU    # 64 → 32
  Conv2d(128 → B, k=3, s=2, p=1) → BatchNorm → ReLU     # 32 → 16

Decoder:
  ConvTranspose2d(B → 128, k=3, s=2, p=1, op=1) → BN → ReLU   # 16 → 32
  ConvTranspose2d(128 → 64, k=3, s=2, p=1, op=1) → BN → ReLU  # 32 → 64
  ConvTranspose2d(64 → 3, k=3, s=2, p=1, op=1) → Sigmoid       # 64 → 128
```

Where **B** (bottleneck channels) is swept over **{64, 128, 256}**.

### 1.5 Training Details

| Hyperparameter | Value |
|---|---|
| Loss | MSE (pixel-level L2) |
| Optimiser | Adam (lr = 1e-3) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Epochs | 30 |
| Batch size | 16 |

### 1.6 Evaluation Metrics

- **MSE** — Mean Squared Error
- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index

### 1.7 Results — Experimental Study

Full grid: **2 noise types x 3 noise levels x 3 bottleneck sizes = 18 experiments**.

#### Gaussian Noise Results

| Noise Level | Bottleneck | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|
| 0.05 | 64 | 0.003488 | 25.10 | 0.7266 |
| 0.05 | 128 | 0.003013 | 25.71 | 0.7546 |
| 0.05 | **256** | **0.002974** | **25.72** | **0.7659** |
| 0.10 | 64 | 0.006003 | 22.58 | 0.6703 |
| 0.10 | 128 | 0.003530 | 24.92 | 0.7241 |
| 0.10 | 256 | 0.003619 | 24.82 | 0.7302 |
| 0.20 | 64 | 0.004459 | 23.90 | 0.6462 |
| 0.20 | 128 | 0.005172 | 23.23 | 0.6488 |
| 0.20 | 256 | 0.004107 | 24.22 | 0.6611 |

#### Salt-and-Pepper Noise Results

| Noise Level | Bottleneck | MSE | PSNR (dB) | SSIM |
|---|---|---|---|---|
| 0.02 | 64 | 0.004015 | 24.43 | 0.7175 |
| 0.02 | 128 | 0.003152 | 25.49 | 0.7371 |
| 0.02 | **256** | 0.003211 | 25.40 | **0.7569** |
| 0.05 | 64 | 0.003711 | 24.75 | 0.7049 |
| 0.05 | 128 | 0.003360 | 25.22 | 0.7190 |
| 0.05 | 256 | 0.003543 | 24.94 | 0.7250 |
| 0.10 | 64 | 0.004471 | 23.86 | 0.6747 |
| 0.10 | 128 | 0.003952 | 24.46 | 0.6845 |
| 0.10 | 256 | 0.004018 | 24.32 | 0.6966 |

### 1.8 Key Observations

1. **Noise level impact**: Higher noise consistently degrades PSNR/SSIM. Best results at lowest noise (Gaussian sigma=0.05 achieves PSNR ~25.7 dB, SSIM ~0.77).
2. **Bottleneck size impact**: The biggest improvement is from 64 to 128 channels. Going from 128 to 256 provides marginal, sometimes inconsistent gains.
3. **Training stability**: All 18 experiments show train/val curves that converge smoothly with no overfitting — train and val loss track each other closely.
4. **Denoised outputs**: The DAE successfully removes both Gaussian and salt-and-pepper noise while preserving global colour and structure. Fine details (text, plaid patterns) are blurred — a known MSE-based reconstruction limitation.

### 1.9 Sample Visualisations

Each triplet grid shows (top) **Clean**, (middle) **Noisy**, (bottom) **Denoised** for 5 test images. Key comparisons:

- **Baseline** (Gaussian sigma=0.1, bneck=256): Noise is visibly removed; colours preserved; edges softened.
- **Gaussian sigma=0.2, bneck=64**: Heaviest noise + smallest bottleneck = most blur.
- **Salt-and-pepper 0.02, bneck=256**: Mildest impulse noise = closest to clean.

---

## Question 2 — Generative Modeling using Variational Autoencoder (VAE)

### 2.1 Problem Statement

Implement a Variational Autoencoder (VAE) to learn latent representations of **Fashion-MNIST** images and generate new synthetic samples. Analyze the impact of different latent dimensions.

### 2.2 Dataset

| Property | Value |
|---|---|
| Source | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (via torchvision) |
| Training set | 50,000 (from original 60k) |
| Validation set | 10,000 (held out from training) |
| Test set | 10,000 |
| Image size | 28 x 28 grayscale |
| Normalisation | Pixel values in [0, 1] |
| Classes | T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |

### 2.3 VAE Architecture

```
Encoder:
  Conv2d(1 → 32, k=3, s=2, p=1) → BN → ReLU       # 28 → 14
  Conv2d(32 → 64, k=3, s=2, p=1) → BN → ReLU      # 14 → 7
  Flatten → FC → mu (latent_dim)
           → FC → log_var (latent_dim)

Reparameterisation:  z = mu + sigma * epsilon,  epsilon ~ N(0, I)

Decoder:
  FC(latent_dim → 64*7*7) → ReLU → Reshape(64, 7, 7)
  ConvTranspose2d(64 → 32, k=3, s=2, p=1, op=1) → BN → ReLU   # 7 → 14
  ConvTranspose2d(32 → 1, k=3, s=2, p=1, op=1) → Sigmoid        # 14 → 28
```

### 2.4 Loss Function

```
Total Loss = Reconstruction Loss (BCE) + KL Divergence
BCE  = sum of pixel-wise binary cross-entropy / batch_size
KL   = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) / batch_size
```

Both components are monitored separately during training.

### 2.5 Training Details

| Hyperparameter | Value |
|---|---|
| Loss | BCE reconstruction + KL divergence |
| Optimiser | Adam (lr = 1e-3) |
| Scheduler | ReduceLROnPlateau (patience=4, factor=0.5) |
| Epochs | 30 |
| Batch size | 128 |

### 2.6 Results — Experimental Study

Latent dimension sweep: **{2, 8, 16, 32, 64}**.

| Latent Dim | Test MSE | Test BCE | Train Loss | Val Loss | Train KL | Val KL |
|---|---|---|---|---|---|---|
| 2 | 0.029231 | 0.329082 | 261.34 | 263.30 | 6.37 | 6.38 |
| 8 | 0.015954 | 0.290195 | 239.85 | 240.98 | 14.61 | 14.62 |
| **16** | **0.014140** | **0.285125** | **239.22** | **240.26** | 17.31 | 17.85 |
| 32 | 0.014365 | 0.285788 | 239.43 | 240.48 | 17.67 | 17.64 |
| 64 | 0.014225 | 0.285427 | 239.71 | 240.61 | 17.73 | 18.00 |

### 2.7 Key Observations

1. **Latent dim 2** is clearly insufficient: MSE is 2x worse than dim 16, and KL is low (~6) because only 2 dimensions contribute. Generated samples are blurry with significant class mixing.
2. **Latent dim 8 to 16** shows the biggest improvement in reconstruction quality (MSE drops from 0.016 to 0.014; BCE from 0.290 to 0.285).
3. **Latent dim 16 is the sweet spot**: achieves the best test MSE/BCE. Dimensions 32 and 64 show diminishing returns with nearly identical metrics.
4. **KL divergence saturates** around 17-18 for dimensions >= 16, indicating the model uses a similar amount of information regardless of extra capacity.
5. **Training stability**: All models converge smoothly with train/val losses tracking each other — no overfitting observed.

### 2.8 Reconstruction Quality

- **ldim=2**: Global shape preserved but heavy blur; fine details lost (e.g. "Lee" text on pullover becomes a blank patch).
- **ldim=8**: Noticeably sharper than 2; shoe treads and trouser creases partially visible.
- **ldim=16/32/64**: Visually very similar — class identity and silhouette are preserved well. All still exhibit typical VAE blur.

### 2.9 Generated Samples

- **ldim=2**: Blurry, low diversity, class mixing (some cells are ambiguous blobs).
- **ldim=8**: More recognisable categories; still some fuzzy cells.
- **ldim=16/32/64**: Broad coverage of all 10 Fashion-MNIST classes; individual samples are identifiable as specific garment types. Typical VAE softness remains.

### 2.10 Latent Space Visualisation (2-D)

The `latent2d_ldim2.png` scatter plot (coloured by class) shows:
- **Trousers** (bottom-left) and **Ankle boots** (top-left) form well-separated clusters — structurally distinct shapes.
- **Bags** cluster separately (left-centre).
- **Shirts, Coats, Pullovers, T-shirts** heavily overlap in the centre — visually similar upper-body garments share latent space.
- **Sneakers vs Sandals** partially overlap — fine footwear distinctions are compressed.

This demonstrates that even 2 latent dimensions learn meaningful structure, but cannot disentangle similar classes.

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib Pillow scikit-image pandas
```

### On Kaggle

1. Upload the BSDS300 dataset (containing `BSDS300-images.tgz`) as a Kaggle dataset.
2. Open the respective notebook (`.ipynb`) and run all cells.
3. Outputs are saved to `/kaggle/working/outputs/` or `/kaggle/working/outputs_vae/`.

### Locally

```bash
cd denoising-autoencoder/code
python denoising_autoencoder.py

cd ../../vae-autoencoder/code
python vae_fashion_mnist.py
```

---

## Deliverables

- [x] Python source code for both questions
- [x] Kaggle notebooks (.ipynb)
- [x] All output plots and visualisations
- [x] Experimental study with tables and figures
- [x] `prompts_used_per_question.txt` — all prompts used
- [x] This README
