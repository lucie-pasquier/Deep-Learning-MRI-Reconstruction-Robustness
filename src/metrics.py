"""
metrics.py
----------
Image quality metrics for evaluating MRI reconstruction.

Contains:
  - calculate_psnr_single: Peak Signal-to-Noise Ratio (PSNR) for a single image pair
  - calculate_ssim_single: Structural Similarity Index (SSIM) for a single image pair
  - calculate_lpips_single: stub for Learned Perceptual Image Patch Similarity (LPIPS)
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr_single(img1, img2, data_range=None):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim_single(img1, img2, data_range=None):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return structural_similarity(img1, img2, data_range=data_range)


def calculate_lpips_single(img1, img2, data_range=None):
    # install lpips package and implement if needed
    pass
