# 🩻 X-Ray Denoiser (U-Net)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning pipeline built to restore and denoise chest X-ray images using a **U-Net** architecture. This project specifically simulates quantum mottle (Poisson noise caused by low X-ray photon counts) and trains a convolutional autoencoder to recover the original high-quality radiograph. 

By utilizing Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) metrics, this pipeline provides a quantitative measure of image restoration quality.

---

## 🌟 Key Features

- **Automated Data Acquisition**: Integrated scripts to easily fetch the `alifrahman/chestxraydataset` from Kaggle.
- **Physics-Informed Noise Simulation**: accurately simulates low-dose X-ray characteristics using Poisson distribution. Also includes modules for Gaussian, Salt & Pepper, and Speckle noise.
- **Deep U-Net Architecture**: Uses a robust encoder-decoder network with skip connections to preserve high-frequency structural details of the anatomy.
- **Comprehensive Evaluation**: Automated PSNR and SSIM calculation combined with side-by-side visual comparisons.

---

## 📂 Repository Structure

```text
xray_denoiser/
├── src/
│   ├── data_download.py               # Fetches chest X-ray dataset from Kaggle
│   ├── generate_artifacts.py          # Applies various noise models to images
│   ├── generate_poisson_dataset.py    # Builds the noisy/clean dataset pairs 
│   ├── train_denoiser.py              # U-Net architecture & training loop
│   └── test_denoiser.py               # Inference, evaluation metrics & plotting
├── notebooks/
│   └── download_data.ipynb            # Jupyter notebook for interactive data exploration
├── data/                              # (Ignored by Git) Datasets and noisy variations
├── models/                            # (Ignored by Git) Saved PyTorch .pth weights
├── results/                           # (Ignored by Git) Inference output plots
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/Anirudh2465/xray_denoiser.git
cd xray_denoiser
pip install torch torchvision opencv-python matplotlib numpy scikit-image kagglehub tqdm
```

### 2. Download the Dataset

Fetch the base dataset containing the clean Normal and Pneumonia X-rays.

```bash
python src/data_download.py
```
*(Move the downloaded images into the `data/train` folder after fetching)*

### 3. Generate the Noisy Dataset

Simulate low-photon count X-rays by adding varying degrees of Poisson noise. This script dynamically pulls images from the `data/train` folder and saves the noisy/clean pairs to `data/poisson_dataset_2`.

```bash
python src/generate_poisson_dataset.py
```

### 4. Train the Model

Train the U-Net model. This will automatically split the dataset into training and validation sets, log the MSE loss per epoch, and save the final weights to `models/unet_denoiser_model.pth`.

```bash
python src/train_denoiser.py
```

### 5. Evaluate the Model

Run inference on random test images. The script calculates **PSNR** and **SSIM** scores and outputs comparison plots (Noisy vs. Denoised vs. Original) into the `results_2/` directory.

```bash
python src/test_denoiser.py --num_images 5
```

---

## 🧠 Model Architecture

The denoiser employs a **U-Net** topology optimized for medical image restoration:

- **Encoder**: 3 contracting blocks using `3x3 Conv2D -> ReLU -> MaxPool2D` to extract hierarchical features.
- **Bottleneck**: Latent space representation capturing dense, high-level structural semantics.
- **Decoder**: 3 expanding blocks using `ConvTranspose2D` (up-sampling).
- **Skip Connections**: Concatenates early encoder feature maps with deep decoder feature maps, ensuring spatial resolution and fine edges (e.g., rib cages, lung opacities) are not lost during the down-sampling phase.
- **Output**: A final `1x1 Conv2D` paired with a Sigmoid activation to output normalized grayscale intensities.

---

## 📊 Evaluation Metrics

To rigorously evaluate the denoising performance, we utilize:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible pixel intensity and the power of corrupting noise. Higher is better.
- **SSIM (Structural Similarity Index Measure)**: Perceives changes in structural information, luminance, and contrast. A value closer to `1.0` indicates perfect structural integrity.

Our model consistently achieves high structural similarity (> 0.90 SSIM), making it highly effective at restoring clinical interpretability to low-dose radiographs.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
