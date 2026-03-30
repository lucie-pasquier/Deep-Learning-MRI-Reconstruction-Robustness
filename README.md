# Robustness of Deep Learning MRI Reconstruction to Distribution Shift

## Overview

This project investigates the robustness of a U-Net MRI reconstruction 
model to distribution shifts, as part of the final assignment for 
**ELEC70121: Trustworthy Artificial Intelligence in Medical Imaging** 
at Imperial College London.

The model is trained on the FastMRI single-coil knee dataset using a 
regular AF4 undersampling mask, then evaluated under three forms of 
distribution shift: changes in acceleration factor, changes in 
undersampling mask pattern, and additive k-space noise. A mask 
switching mitigation strategy is also evaluated.

## Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `numpy`, `scipy`, `scikit-image`, 
`matplotlib`, `h5py`, `tqdm`

## Data

The FastMRI single-coil knee dataset is not included in this 
repository. Place the data under `data/fastmri_tiny/` following 
the structure described in [data/README.md](data/README.md). 
The dataset was provided as part of the ELEC70121 course lab materials.

## Model Weights

Trained model weights are not included due to file size. To reproduce 
them, run `notebooks/01_baseline_training.ipynb` with the dataset in 
place. Training takes approximately 25 minutes on Apple Silicon (M4).

## How to Run

Run the notebooks in order:

1. [notebooks/01_baseline_training.ipynb](notebooks/01_baseline_training.ipynb) 
   — trains the baseline U-Net and saves weights to `weights/`
2. [notebooks/02_robustness_experiments.ipynb](notebooks/02_robustness_experiments.ipynb) 
   — runs all robustness experiments and saves results to `results/`

All notebooks import from `src/` — no separate script needs to be run.

## Project Structure
```
├── data/                    # Dataset (not tracked — see data/README.md)
├── masks/                   # k-space undersampling masks (.npy)
├── weights/                 # Saved model checkpoints (not tracked)
├── results/                 # Output figures (not tracked)
├── src/
│   ├── model.py             # U-Net and residual U-Net definitions
│   ├── dataset.py           # FastMRI dataset class, noisy variant
│   ├── metrics.py           # PSNR and SSIM metric functions
│   ├── utils.py             # Loss functions and utilities
│   └── train.py             # Training and validation functions
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   └── 02_robustness_experiments.ipynb
├── requirements.txt
└── README.md
```

## Reference

This work builds on the following papers:
- Zbontar et al., FastMRI (2019)
- Antun et al., On Instabilities of Deep Learning (2020)
- Johnson et al., Robustness of Learned MR Reconstruction (2021)
- Cheng et al., Addressing the False Negative Problem (2020)