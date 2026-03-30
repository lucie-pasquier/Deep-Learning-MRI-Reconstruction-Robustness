"""
dataset.py
----------
Data loading and preprocessing for FastMRI reconstruction.

Contains:
  - read_processed_h5: load a single .h5 file into a dict with the complex image
  - preprocess_normalisation: normalise a complex image so its magnitude is in [0, 1]
  - define_mask: load a 1-D k-space undersampling mask and broadcast to 2-D
  - undersample_kspace: apply a mask to fully-sampled k-space via FFT / IFFT
  - DatasetFastMRI: PyTorch Dataset that returns (undersampled, ground-truth) pairs
    as 2-channel (real/imag) float tensors
"""

import os
import h5py
import numpy as np
from glob import glob
from scipy.fftpack import fftshift, ifftshift, fftn, ifftn

import torch
from torch.utils.data import Dataset

# Default mask root: <project_root>/masks/  (resolved relative to this file)
_DEFAULT_MASK_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'masks'))


def read_processed_h5(data_path):
    with h5py.File(data_path, 'r') as file:
        data = {
            'image_complex': file['image_complex'][()],
            'data_name': file['image_complex'].attrs['data_name'],
            'slice_idx': file['image_complex'].attrs['slice_idx'],
        }
    return data


def preprocess_normalisation(img, type):
    if type == 'complex_mag':
        mag = np.abs(img)
        img = img / mag.max()
    else:
        raise NotImplementedError
    return img


def define_mask(mask_name, masks_root=None):
    """Load a named undersampling mask and return a 2-D (H x W) array.

    Args:
        mask_name:  string key identifying the mask (e.g. 'fMRI_Reg_AF4_CF0.08_PE320')
        masks_root: optional path to the fastmri masks directory.
                    If None, defaults to <project_root>/masks/fastmri/.
    """
    root = masks_root if masks_root is not None else os.path.join(_DEFAULT_MASK_ROOT, 'fastmri')

    # GRAPPA-like (with ACS) Regular Acceleration Factor x Central Fraction x PE (from fastMRI)
    if mask_name == 'fMRI_Reg_AF2_CF0.16_PE320':
        mask_1d = np.load(os.path.join(root, 'regular', 'regular_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE320':
        mask_1d = np.load(os.path.join(root, 'regular', 'regular_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE320':
        mask_1d = np.load(os.path.join(root, 'regular', 'regular_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE320':
        mask_1d = np.load(os.path.join(root, 'regular', 'regular_af16_cf0.02_pe320.npy'))

    # GRAPPA-like (with ACS) Random (Gaussian) Acceleration Factor x Central Fraction x PE (from fastMRI)
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE320':
        mask_1d = np.load(os.path.join(root, 'random', 'random_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE320':
        mask_1d = np.load(os.path.join(root, 'random', 'random_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE320':
        mask_1d = np.load(os.path.join(root, 'random', 'random_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE320':
        mask_1d = np.load(os.path.join(root, 'random', 'random_af16_cf0.02_pe320.npy'))

    else:
        raise NotImplementedError

    mask_1d = mask_1d[:, np.newaxis]
    mask = np.repeat(mask_1d, 320, axis=1).transpose((1, 0))

    return mask


def undersample_kspace(x, mask):
    fft = fftshift(fftn(ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    fft = fft * mask
    x = fftshift(ifftn(ifftshift(fft, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    return x


class DatasetFastMRI(Dataset):

    def __init__(self, data_path_src, mask_name, masks_root=None, is_debug=False):
        super(DatasetFastMRI, self).__init__()

        # get undersampling mask
        self.mask = define_mask(mask_name, masks_root=masks_root)

        # get data paths
        self.data_paths = glob(os.path.join(data_path_src, '*.h5'))

        # for debug: use a small subset
        self.data_paths = self.data_paths[:12] if is_debug else self.data_paths

    def __getitem__(self, index):

        mask = self.mask  # H, W

        H_path = self.data_paths[index]

        img_dict = read_processed_h5(H_path)

        # fully-sampled complex image
        img_H_SC = img_dict['image_complex']

        # normalise magnitude to [0, 1]
        img_H_SC = preprocess_normalisation(img_H_SC, type='complex_mag')

        # simulate undersampled acquisition
        img_L_SC = undersample_kspace(img_H_SC, mask)

        # expand channel dim
        img_H_SC = img_H_SC[:, :, np.newaxis]  # H, W, 1
        img_L_SC = img_L_SC[:, :, np.newaxis]  # H, W, 1

        # complex -> 2-channel (real, imag)
        img_H_SC = np.concatenate((np.real(img_H_SC), np.imag(img_H_SC)), axis=-1)  # H, W, 2
        img_L_SC = np.concatenate((np.real(img_L_SC), np.imag(img_L_SC)), axis=-1)  # H, W, 2

        # HWC -> CHW, numpy -> tensor
        img_L_SC = torch.from_numpy(np.ascontiguousarray(img_L_SC)).permute(2, 0, 1).to(torch.float32)
        img_H_SC = torch.from_numpy(np.ascontiguousarray(img_H_SC)).permute(2, 0, 1).to(torch.float32)

        return img_L_SC, img_H_SC

    def __len__(self):
        return len(self.data_paths)


class DatasetFastMRI_Noisy(DatasetFastMRI):
    """DatasetFastMRI with additive complex Gaussian noise injected into k-space.

    Args:
        noise_sigma: noise level as a fraction of the signal standard deviation
                     of the undersampled k-space magnitude. e.g. 0.5 = half a std dev.
        All other arguments are forwarded to DatasetFastMRI.
    """

    def __init__(self, data_path_src, mask_name, noise_sigma, masks_root=None, is_debug=False):
        super().__init__(data_path_src, mask_name, masks_root=masks_root, is_debug=is_debug)
        self.noise_sigma = noise_sigma

    def __getitem__(self, index):
        mask = self.mask  # H, W

        H_path = self.data_paths[index]
        img_dict = read_processed_h5(H_path)

        img_H_SC = img_dict['image_complex']
        img_H_SC = preprocess_normalisation(img_H_SC, type='complex_mag')

        # compute undersampled k-space
        fft = fftshift(fftn(ifftshift(img_H_SC, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        kspace = fft * mask

        # add complex Gaussian noise scaled to the signal std dev
        sigma = self.noise_sigma * np.mean(np.abs(kspace[kspace != 0]))
        noise = sigma * (np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape)) / np.sqrt(2)
        kspace_noisy = kspace + noise

        # reconstruct via IFFT
        img_L_SC = fftshift(ifftn(ifftshift(kspace_noisy, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

        # expand channel dim and convert to 2-channel real/imag
        img_H_SC = img_H_SC[:, :, np.newaxis]
        img_L_SC = img_L_SC[:, :, np.newaxis]

        img_H_SC = np.concatenate((np.real(img_H_SC), np.imag(img_H_SC)), axis=-1)
        img_L_SC = np.concatenate((np.real(img_L_SC), np.imag(img_L_SC)), axis=-1)

        img_L_SC = torch.from_numpy(np.ascontiguousarray(img_L_SC)).permute(2, 0, 1).to(torch.float32)
        img_H_SC = torch.from_numpy(np.ascontiguousarray(img_H_SC)).permute(2, 0, 1).to(torch.float32)

        return img_L_SC, img_H_SC
