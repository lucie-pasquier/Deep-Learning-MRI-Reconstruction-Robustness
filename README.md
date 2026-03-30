# Robustness of Deep Learning MRI Reconstruction Models to Distribution Shifts

## Overview

This project investigates the robustness of deep learning-based MRI reconstruction models when subjected to distribution shifts. Specifically, it examines how models trained to reconstruct undersampled k-space data (accelerated MRI acquisition) perform when the test distribution differs from the training distribution — for example, through changes in undersampling pattern, acceleration factor, or noise level.

The work is submitted as the final assignment for **ELEC70121: Trustworthy Artificial Intelligence in Medical Imaging** at Imperial College London.

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch` — model training and inference
- `numpy`, `scipy` — numerical operations and Fourier transforms
- `scikit-image` — image quality metrics (PSNR, SSIM)
- `matplotlib` — visualisation
- `h5py` — loading FastMRI `.h5` data files
- `tqdm` — progress bars
- `lpips` — perceptual similarity metric

## How to Run

1. Place the FastMRI tiny dataset under `data/` following the structure described in [data/README.md](data/README.md).

2. Train the baseline U-Net reconstruction model:

```bash
python src/train.py
```

3. Run experiments interactively via the notebooks:

   - [notebooks/01_baseline_training.ipynb](notebooks/01_baseline_training.ipynb) — trains and evaluates the baseline model
   - [notebooks/02_robustness_experiments.ipynb](notebooks/02_robustness_experiments.ipynb) — applies distribution shifts and measures performance degradation

## Project Structure

```
MI Final Assignment/
├── data/               # FastMRI dataset (not tracked)
│   └── README.md       # Instructions for obtaining the data
├── masks/              # k-space undersampling masks (.npy)
├── weights/            # Saved model checkpoints
├── results/            # Output figures and metric logs
├── src/
│   ├── __init__.py
│   ├── model.py        # U-Net and U-Net residual model definitions
│   ├── dataset.py      # FastMRI dataset class and k-space undersampling
│   ├── metrics.py      # PSNR, SSIM, LPIPS metric functions
│   ├── utils.py        # Shared utilities (normalisation, visualisation, I/O)
│   └── train.py        # Training and evaluation entry point
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   └── 02_robustness_experiments.ipynb
├── requirements.txt
└── README.md
```
