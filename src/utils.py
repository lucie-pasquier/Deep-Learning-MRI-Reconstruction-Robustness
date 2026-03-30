"""
utils.py
--------
Shared utility functions for MRI reconstruction training.

Contains:
  - mkdir: create a directory if it does not already exist
  - fft_map: compute the real and imaginary parts of the FFT of a tensor
  - define_loss: return a pixel-space loss function (L1 or L2) by name
  - total_loss: combined image-domain + frequency-domain loss used for training
"""

import os
from functools import partial

import torch
import torch.nn as nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag
    return fft_x_real, fft_x_imag


def define_loss(loss_type):
    if loss_type == 'l1':
        lossfn = nn.L1Loss()
    elif loss_type == 'l2':
        lossfn = nn.MSELoss()
    else:
        raise NotImplementedError
    return lossfn


def total_loss(predict,
               target,
               loss_image_weight=15,
               loss_image_type='l1',
               loss_freq_weight=0.1,
               loss_freq_type='l1',
               device='cpu'):

    lossfn_image = define_loss(loss_image_type).to(device)
    loss_image = lossfn_image(predict, target)

    lossfn_freq = define_loss(loss_freq_type).to(device)
    target_k_real, target_k_imag = fft_map(target)
    predict_k_real, predict_k_imag = fft_map(predict)
    loss_freq = (lossfn_freq(predict_k_real, target_k_real) +
                 lossfn_freq(predict_k_imag, target_k_imag)) / 2

    return loss_image_weight * loss_image + loss_freq_weight * loss_freq
