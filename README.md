# 🩻 X-Ray Denoiser & Pneumonia Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning pipeline built for two primary clinical tasks:
1. **Restoring and denoising** chest X-ray images suffering from quantum mottle (Poisson noise) using a U-Net architecture.
2. **Classifying** chest X-ray images into Normal vs. Pneumonia using multiple model architectures (U-Net, ResNet18, DenseNet121).

---

## 🌟 Key Features

- **Automated Data Acquisition**: Integrated scripts to fetch the `alifrahman/chestxraydataset` from Kaggle.
- **Physics-Informed Noise Simulation**: Accurately simulates low-dose X-ray characteristics using Poisson distribution.
- **Image Restoration**: Deep U-Net architecture with skip connections to denoise images while preserving high-frequency structural details (PSNR & SSIM evaluated).
- **Disease Classification**: A comparative classification pipeline utilizing U-Net, ResNet18, and DenseNet121 to detect Pneumonia, complete with automated dataset splitting, training, and comprehensive evaluation (Accuracy, Precision, Recall, F1, ROC/AUC, Confusion Matrices).

---

## 📂 Repository Structure

```text
xray_processing_denoising_classification_using_DL/
├── src/
│   ├── data_download.py               # Fetches chest X-ray dataset from Kaggle
│   ├── generate_artifacts.py          # Applies various noise models to images
│   ├── generate_poisson_dataset.py    # Builds the noisy/clean dataset pairs 
│   ├── train_denoiser.py              # Denoising: U-Net architecture & training loop
│   ├── test_denoiser.py               # Denoising: Inference, evaluation & plotting
│   ├── classifier_dataset.py          # Classification: Dataset loader and train/val/test splitter
│   ├── classifier_models.py           # Classification: UNet, ResNet, and DenseNet architectures
│   ├── train_classifier.py            # Classification: Training loop for all models
│   └── evaluate_classifiers.py        # Classification: Generates metrics, ROC curves, and confusion matrices
├── run_all.ps1                        # PowerShell script to run the entire classification pipeline sequentially
├── notebooks/                         # Jupyter notebook for interactive data exploration
├── data/                              # (Ignored by Git) Datasets and noisy variations
├── classification_results/            # (Ignored by Git) Output plots and metrics for classification
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/Anirudh2465/xray_processing_denoising_classification_using_DL.git
cd xray_processing_denoising_classification_using_DL
pip install torch torchvision opencv-python matplotlib numpy scikit-image kagglehub tqdm seaborn scikit-learn
```

### 2. Download the Dataset

Fetch the base dataset containing the clean Normal and Pneumonia X-rays.

```bash
python src/data_download.py
```
*(Move the downloaded images into the `data/train` folder after fetching)*

---

## 🛠️ Pipeline A: Image Denoising

### 1. Generate the Noisy Dataset
Simulate low-photon count X-rays by adding varying degrees of Poisson noise. 
```bash
python src/generate_poisson_dataset.py
```

### 2. Train the Denoiser
Train the U-Net autoencoder model to reconstruct clean images.
```bash
python src/train_denoiser.py
```

### 3. Evaluate the Denoiser
Run inference on random test images to calculate **PSNR** and **SSIM** scores and generate visual comparisons.
```bash
python src/test_denoiser.py --num_images 5
```

---

## 🔬 Pipeline B: Pneumonia Classification

This pipeline trains three different architectures to classify images as either Normal or Pneumonia.

### 1. Run the Full Pipeline
You can run the full training and evaluation process for all three models sequentially using the provided PowerShell script:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

### 2. Run Individually
Alternatively, train a specific model:
```bash
python src/train_classifier.py --model resnet --epochs 10
```
*(Options: `unet`, `resnet`, `densenet`)*

Evaluate all trained models and generate comparison plots:
```bash
python src/evaluate_classifiers.py
```

### 3. Classification Results
Outputs are saved to the `classification_results/` directory, including:
- `metrics_summary.txt`: Accuracy, Precision, Recall, and F1-Scores.
- `{model_name}_learning_curves.png`: Training/Validation Loss and Accuracy plots.
- `confusion_matrices.png`: Visual heatmap of True/False Positives & Negatives.
- `roc_curves.png`: ROC Curve and AUC scores.

---

## 🧠 Model Architectures

### Denoiser
- **U-Net**: An encoder-decoder architecture optimized for image-to-image tasks. Skip connections preserve spatial resolution and fine edges (e.g., rib cages, lung opacities) during the down-sampling phase.

### Classifiers
- **ResNet18**: A standard convolutional network utilizing residual blocks to solve the vanishing gradient problem. Fine-tuned from ImageNet weights.
- **DenseNet121**: A deeply connected network where each layer receives feature maps from all preceding layers. Highly efficient for medical imaging tasks.
- **UNetClassifier**: The U-Net encoder adapted for classification via a Global Average Pooling head.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
