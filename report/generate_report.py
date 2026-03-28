"""
Generate Assignment 02 PDF Report
==================================
Creates a professional PDF with cover page (FAST logo, student info)
and full content for both questions including tables, figures, and analysis.

Requirements:  pip install fpdf2 Pillow
Usage:         python generate_report.py
Output:        A02_Report_Hammad_Zahid_22I-2433.pdf
"""

from pathlib import Path
from fpdf import FPDF

BASE = Path(__file__).resolve().parent
DAE_OUT = BASE / "denoising-autoencoder" / "outputs"
VAE_OUT = BASE / "vae-autoencoder" / "outputs_vae"
LOGO = BASE / "image.png"
OUTPUT_PDF = BASE / "A02_Report_Hammad_Zahid_22I-2433.pdf"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Assignment 02 - Generative AI | Hammad Zahid (22I-2433)", align="C")
            self.ln(10)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 70, 130)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 70, 130)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 70, 130)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        for j, row in enumerate(rows):
            fill = j % 2 == 0
            if fill:
                self.set_fill_color(235, 240, 250)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, align="C", fill=fill)
            self.ln()
        self.ln(3)

    def add_image_safe(self, path, w=None, caption=None):
        p = Path(path)
        if not p.exists():
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(180, 0, 0)
            self.cell(0, 6, f"[Image not found: {p.name}]", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
            return
        if w is None:
            w = self.w - self.l_margin - self.r_margin
        space_needed = 60
        if self.get_y() + space_needed > self.h - 25:
            self.add_page()
        self.image(str(p), w=w)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(3)


def build():
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===================== COVER PAGE =====================
    pdf.add_page()
    pdf.ln(15)
    if LOGO.exists():
        logo_w = 50
        x = (pdf.w - logo_w) / 2
        pdf.image(str(LOGO), x=x, w=logo_w)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 70, 130)
    pdf.cell(0, 10, "National University of Computer & Emerging Sciences", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 14, "Assignment 02", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Generative AI", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)
    info_lines = [
        ("Name", "Hammad Zahid"),
        ("Roll Number", "22I-2433"),
        ("Section", "SE-D"),
        ("Submitted to", "Dr. Shahela Saif"),
        ("Course", "Generative AI"),
        ("Date", "March 28, 2026"),
    ]
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(30, 30, 30)
    for label, value in info_lines:
        pdf.cell(70, 9, f"{label}:", align="R")
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, f"  {value}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 13)

    # ===================== QUESTION 1 =====================
    pdf.add_page()
    pdf.section_title("Question 1: Image Denoising using Denoising Autoencoder")

    pdf.sub_title("1.1  Problem Statement")
    pdf.body_text(
        "Design and implement a Denoising Autoencoder (DAE) capable of reconstructing "
        "clean images from noisy inputs. The dataset used is the Berkeley Segmentation "
        "Dataset and Benchmark (BSDS300), containing 300 natural images."
    )

    pdf.sub_title("1.2  Dataset Preparation")
    pdf.body_text(
        "The BSDS300 dataset (300 images: 200 train, 100 test) is loaded from the official "
        "Berkeley tarball. Images are resized to 128x128 and normalised to [0, 1]. "
        "The 200 training images are split into 160 train and 40 validation."
    )

    pdf.sub_title("1.3  Noise Injection")
    pdf.body_text(
        "Two noise types are applied on-the-fly during training:\n"
        "  - Gaussian noise: zero-mean additive noise with sigma in {0.05, 0.10, 0.20}\n"
        "  - Salt-and-pepper noise: random pixel flipping with amount in {0.02, 0.05, 0.10}\n"
        "Noise is re-sampled each epoch, providing implicit data augmentation."
    )

    pdf.sub_title("1.4  Model Architecture")
    pdf.body_text(
        "A fully-convolutional DAE with 3 encoder layers (Conv2d + BatchNorm + ReLU, stride=2) "
        "and 3 decoder layers (ConvTranspose2d + BatchNorm + ReLU, final layer uses Sigmoid). "
        "The bottleneck channel count B is swept over {64, 128, 256}. "
        "Activation: ReLU throughout, Sigmoid on output to keep pixels in [0,1]."
    )

    pdf.sub_title("1.5  Training Configuration")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Loss Function", "MSE (pixel-level L2)"],
            ["Optimiser", "Adam (lr = 1e-3)"],
            ["Scheduler", "ReduceLROnPlateau (patience=3)"],
            ["Epochs", "30"],
            ["Batch Size", "16"],
        ],
        col_widths=[60, 120],
    )

    pdf.sub_title("1.6  Baseline Results")
    pdf.body_text(
        "Baseline model (Gaussian sigma=0.1, bottleneck=256):\n"
        "  Test MSE = 0.003947 | PSNR = 24.47 dB | SSIM = 0.7313"
    )
    pdf.add_image_safe(DAE_OUT / "loss_baseline_gaussian_0.1_bneck256.png",
                       w=140, caption="Fig 1: Baseline loss curve (Gaussian sigma=0.1, bneck=256)")
    pdf.add_image_safe(DAE_OUT / "vis_Baseline_Gaussian_sigma0.1_bneck256.png",
                       w=170, caption="Fig 2: Baseline denoising - Clean / Noisy / Denoised")

    # Results table - Gaussian
    pdf.add_page()
    pdf.sub_title("1.7  Experimental Study - Gaussian Noise")
    cw = [28, 28, 28, 28, 28]
    pdf.add_table(
        ["Level", "Bneck", "MSE", "PSNR", "SSIM"],
        [
            ["0.05", "64", "0.003488", "25.10", "0.7266"],
            ["0.05", "128", "0.003013", "25.71", "0.7546"],
            ["0.05", "256", "0.002974", "25.72", "0.7659"],
            ["0.10", "64", "0.006003", "22.58", "0.6703"],
            ["0.10", "128", "0.003530", "24.92", "0.7241"],
            ["0.10", "256", "0.003619", "24.82", "0.7302"],
            ["0.20", "64", "0.004459", "23.90", "0.6462"],
            ["0.20", "128", "0.005172", "23.23", "0.6488"],
            ["0.20", "256", "0.004107", "24.22", "0.6611"],
        ],
        col_widths=cw,
    )
    pdf.add_image_safe(DAE_OUT / "bar_gaussian_PSNR.png", w=140,
                       caption="Fig 3: PSNR vs Noise Level - Gaussian")
    pdf.add_image_safe(DAE_OUT / "bar_gaussian_SSIM.png", w=140,
                       caption="Fig 4: SSIM vs Noise Level - Gaussian")

    # Results table - Salt & Pepper
    pdf.add_page()
    pdf.sub_title("1.8  Experimental Study - Salt-and-Pepper Noise")
    pdf.add_table(
        ["Level", "Bneck", "MSE", "PSNR", "SSIM"],
        [
            ["0.02", "64", "0.004015", "24.43", "0.7175"],
            ["0.02", "128", "0.003152", "25.49", "0.7371"],
            ["0.02", "256", "0.003211", "25.40", "0.7569"],
            ["0.05", "64", "0.003711", "24.75", "0.7049"],
            ["0.05", "128", "0.003360", "25.22", "0.7190"],
            ["0.05", "256", "0.003543", "24.94", "0.7250"],
            ["0.10", "64", "0.004471", "23.86", "0.6747"],
            ["0.10", "128", "0.003952", "24.46", "0.6845"],
            ["0.10", "256", "0.004018", "24.32", "0.6966"],
        ],
        col_widths=cw,
    )
    pdf.add_image_safe(DAE_OUT / "bar_salt_pepper_PSNR.png", w=140,
                       caption="Fig 5: PSNR vs Noise Level - Salt & Pepper")
    pdf.add_image_safe(DAE_OUT / "bar_salt_pepper_SSIM.png", w=140,
                       caption="Fig 6: SSIM vs Noise Level - Salt & Pepper")

    # Sample visualisations
    pdf.add_page()
    pdf.sub_title("1.9  Sample Denoising Visualisations")

    vis_samples = [
        ("vis_gaussian__\u03c30.05__bneck256.png", "Gaussian sigma=0.05, bneck=256"),
        ("vis_gaussian__\u03c30.1__bneck256.png", "Gaussian sigma=0.1, bneck=256"),
        ("vis_gaussian__\u03c30.2__bneck256.png", "Gaussian sigma=0.2, bneck=256"),
        ("vis_salt_pepper__\u03c30.02__bneck256.png", "Salt & Pepper 0.02, bneck=256"),
        ("vis_salt_pepper__\u03c30.1__bneck256.png", "Salt & Pepper 0.10, bneck=256"),
    ]
    for fname, cap in vis_samples:
        pdf.add_image_safe(DAE_OUT / fname, w=170,
                           caption=f"Fig: {cap} - Clean / Noisy / Denoised")

    # Observations
    pdf.add_page()
    pdf.sub_title("1.10  Discussion & Observations")
    pdf.body_text(
        "1. Noise level impact: Higher noise consistently degrades PSNR and SSIM. "
        "Best results are achieved at the lowest noise (Gaussian sigma=0.05: PSNR ~25.7 dB, SSIM ~0.77).\n\n"
        "2. Bottleneck size impact: The largest improvement occurs from 64 to 128 channels. "
        "Going from 128 to 256 yields marginal, sometimes inconsistent gains - indicating "
        "the bottleneck is no longer the primary limiter past 128 channels.\n\n"
        "3. Training stability: All 18 experiments show smooth convergence with train and "
        "validation losses tracking each other closely - no overfitting.\n\n"
        "4. Qualitative results: The DAE successfully removes both noise types while "
        "preserving global colour and structure. Fine details (text, plaid) are blurred "
        "- a known limitation of MSE-based autoencoders.\n\n"
        "5. Noise type comparison: The model handles salt-and-pepper noise slightly better "
        "at equivalent visual severity, since impulse noise is more localised than Gaussian."
    )

    # ===================== QUESTION 2 =====================
    pdf.add_page()
    pdf.section_title("Question 2: Generative Modeling using VAE")

    pdf.sub_title("2.1  Problem Statement")
    pdf.body_text(
        "Implement a Variational Autoencoder (VAE) to learn latent representations of "
        "Fashion-MNIST images and generate new synthetic samples. Analyze the impact of "
        "different latent dimensions on reconstruction quality and generation capability."
    )

    pdf.sub_title("2.2  Dataset Preparation")
    pdf.body_text(
        "Fashion-MNIST: 60,000 training + 10,000 test grayscale images (28x28). "
        "The training set is split into 50,000 train and 10,000 validation. "
        "Pixel values are normalised to [0, 1]. 10 classes of clothing items."
    )

    pdf.sub_title("2.3  VAE Architecture")
    pdf.body_text(
        "Convolutional VAE:\n"
        "  Encoder: Conv(1->32, s2) -> BN -> ReLU -> Conv(32->64, s2) -> BN -> ReLU -> "
        "Flatten -> FC -> mu, FC -> log_var\n"
        "  Reparameterisation: z = mu + sigma * epsilon, epsilon ~ N(0,I)\n"
        "  Decoder: FC -> Reshape(64,7,7) -> ConvT(64->32, s2) -> BN -> ReLU -> "
        "ConvT(32->1, s2) -> Sigmoid"
    )

    pdf.sub_title("2.4  Loss Function")
    pdf.body_text(
        "Combined loss = Reconstruction BCE + KL Divergence.\n"
        "BCE: pixel-wise binary cross-entropy (summed over pixels, averaged over batch).\n"
        "KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var)), averaged over batch.\n"
        "Both components are tracked separately every epoch."
    )

    pdf.sub_title("2.5  Training Configuration")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Loss Function", "BCE + KL Divergence"],
            ["Optimiser", "Adam (lr = 1e-3)"],
            ["Scheduler", "ReduceLROnPlateau (patience=4)"],
            ["Epochs", "30"],
            ["Batch Size", "128"],
        ],
        col_widths=[60, 120],
    )

    pdf.sub_title("2.6  Baseline Results (latent_dim=16)")
    pdf.body_text(
        "Test Pixel MSE = 0.014274 | Test Pixel BCE = 0.285491\n"
        "Final Train Loss = 239.34 | Final Val Loss = 240.19"
    )
    pdf.add_image_safe(VAE_OUT / "curves_baseline_ldim16.png", w=160,
                       caption="Fig 7: Baseline training curves (latent_dim=16)")
    pdf.add_image_safe(VAE_OUT / "recon_baseline_ldim16.png", w=170,
                       caption="Fig 8: Baseline reconstruction - Original (top) vs Reconstructed (bottom)")
    pdf.add_image_safe(VAE_OUT / "gen_baseline_ldim16.png", w=130,
                       caption="Fig 9: Baseline generated samples (z ~ N(0,I), latent_dim=16)")

    # Experimental study
    pdf.add_page()
    pdf.sub_title("2.7  Experimental Study - Latent Dimension Sweep")
    pdf.add_table(
        ["Dim", "MSE", "BCE", "Train Loss", "Val Loss", "Train KL", "Val KL"],
        [
            ["2", "0.0292", "0.329", "261.3", "263.3", "6.37", "6.38"],
            ["8", "0.0160", "0.290", "239.9", "241.0", "14.6", "14.6"],
            ["16", "0.0141", "0.285", "239.2", "240.3", "17.3", "17.9"],
            ["32", "0.0144", "0.286", "239.4", "240.5", "17.7", "17.6"],
            ["64", "0.0142", "0.285", "239.7", "240.6", "17.7", "18.0"],
        ],
        col_widths=[18, 24, 24, 30, 28, 28, 28],
    )

    pdf.add_image_safe(VAE_OUT / "bar_test_MSE.png", w=130,
                       caption="Fig 10: Test Pixel MSE vs Latent Dimension")
    pdf.add_image_safe(VAE_OUT / "bar_test_BCE.png", w=130,
                       caption="Fig 11: Test Pixel BCE vs Latent Dimension")
    pdf.add_image_safe(VAE_OUT / "bar_kl.png", w=130,
                       caption="Fig 12: KL Divergence vs Latent Dimension")

    # Reconstructions comparison
    pdf.add_page()
    pdf.sub_title("2.8  Reconstruction Comparison")
    for dim in [2, 8, 16, 32, 64]:
        pdf.add_image_safe(VAE_OUT / f"recon_ldim{dim}.png", w=170,
                           caption=f"Reconstruction - latent_dim={dim}")

    # Generated samples
    pdf.add_page()
    pdf.sub_title("2.9  Generated Samples Comparison")
    for dim in [2, 16, 64]:
        pdf.add_image_safe(VAE_OUT / f"gen_ldim{dim}.png", w=130,
                           caption=f"Generated samples - latent_dim={dim}")

    # Latent space
    pdf.add_page()
    pdf.sub_title("2.10  2-D Latent Space Visualisation")
    pdf.add_image_safe(VAE_OUT / "latent2d_ldim2.png", w=130,
                       caption="Fig: 2-D latent space scatter (coloured by class)")
    pdf.body_text(
        "The scatter plot shows partial class separation: Trousers and Ankle boots form "
        "distinct clusters (structurally different shapes), while Shirts, Coats, and "
        "Pullovers overlap heavily in the centre (visually similar upper-body garments). "
        "Bags form a separate cluster. This demonstrates that even 2 latent dimensions "
        "learn meaningful structure, but cannot disentangle similar classes."
    )

    # Discussion
    pdf.sub_title("2.11  Discussion & Observations")
    pdf.body_text(
        "1. Reconstruction quality improves sharply from latent dimension 2 to 16 "
        "(MSE halves from 0.029 to 0.014). Beyond 16, gains are negligible.\n\n"
        "2. Latent dimension 16 is the sweet spot for this architecture and training budget - "
        "best test MSE/BCE with reasonable KL divergence.\n\n"
        "3. KL divergence saturates around 17-18 for dimensions >= 16. The encoder uses a "
        "similar total information budget regardless of extra latent capacity.\n\n"
        "4. Generated samples become more recognisable and diverse from dim 2 to 16. "
        "Beyond 16, visual quality plateaus - typical VAE blur persists.\n\n"
        "5. The 2D latent space shows structured separation for dissimilar classes but "
        "heavy overlap for similar garment types, consistent with the information bottleneck.\n\n"
        "6. No overfitting is observed in any configuration - train and validation "
        "losses remain close throughout training."
    )

    # Save
    pdf.output(str(OUTPUT_PDF))
    print(f"[DONE] PDF saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build()
