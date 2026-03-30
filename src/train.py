"""
train.py
--------
Training and validation loops for MRI reconstruction models.

Contains:
  - train_one_epoch: single-epoch supervised training loop (U-Net baseline)
  - train_gan_one_epoch: single-epoch GAN training loop (generator + discriminator)
  - validate: run inference over a dataloader and return PSNR / SSIM metrics
  - validate_one_step: run inference on a single batch and plot input / prediction / ground-truth
"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .metrics import calculate_psnr_single, calculate_ssim_single

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Supervised training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_batch_losses = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}", leave=False):
        input, gt = batch
        input, gt = input.to(device), gt.to(device)

        optimizer.zero_grad()
        pred = model(input)
        loss = loss_fn(pred, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_batch_losses.append(loss.item())

    return total_loss / len(dataloader), all_batch_losses


# ---------------------------------------------------------------------------
# GAN training
# ---------------------------------------------------------------------------

def train_gan_one_epoch(generator, discriminator, dataloader,
                        optimizer_G, optimizer_D,
                        loss_fn_recon, loss_fn_adv,
                        epoch, total_epochs,
                        loss_g_weight=1, loss_d_weight=0.1):
    generator.train()
    discriminator.train()

    total_loss_G = 0
    total_loss_D_real = 0
    total_loss_D_fake = 0
    all_batch_losses_G = []
    all_batch_losses_D_real = []
    all_batch_losses_D_fake = []

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{total_epochs}", leave=False):
        input, gt = batch
        input, gt = input.to(device), gt.to(device)

        # ------------------------------------------------------------------
        # Train the generator
        # ------------------------------------------------------------------

        # D: Freeze
        for p in discriminator.parameters():
            p.requires_grad = False
        # G: Zero grad
        optimizer_G.zero_grad()
        # G: Forward pass
        pred = generator(input)
        # D: Forward pass on fake (for generator loss)
        d_fake_for_g = discriminator(pred)
        # G: Loss
        loss_adversarial = loss_fn_adv(d_fake_for_g, torch.ones_like(d_fake_for_g))
        loss_recon = loss_fn_recon(pred, gt)
        loss_g_total = loss_g_weight * loss_recon + loss_d_weight * loss_adversarial
        # G: Backward pass
        loss_g_total.backward()
        optimizer_G.step()

        # ------------------------------------------------------------------
        # Train the discriminator
        # ------------------------------------------------------------------

        # D: Unfreeze
        for p in discriminator.parameters():
            p.requires_grad = True
        # D: Zero grad
        optimizer_D.zero_grad()
        # D: Forward pass
        d_real = discriminator(gt)                   # real data
        d_fake = discriminator(pred.detach())        # fake data, detach to avoid BP to generator
        # D: Loss
        loss_d_real = loss_fn_adv(d_real, torch.ones_like(d_real))    # vs True
        loss_d_fake = loss_fn_adv(d_fake, torch.zeros_like(d_fake))   # vs False
        # D: Backward pass
        loss_d_real.backward()
        loss_d_fake.backward()
        optimizer_D.step()

        total_loss_G += loss_g_total.item()
        total_loss_D_real += loss_d_real.item()
        total_loss_D_fake += loss_d_fake.item()
        all_batch_losses_G.append(loss_g_total.item())
        all_batch_losses_D_real.append(loss_d_real.item())
        all_batch_losses_D_fake.append(loss_d_fake.item())

    return (total_loss_G / len(dataloader),
            total_loss_D_real / len(dataloader),
            total_loss_D_fake / len(dataloader),
            all_batch_losses_G,
            all_batch_losses_D_real,
            all_batch_losses_D_fake)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, dataloader):
    model.eval()

    metrics_dict = {
        'psnr': [],
        'ssim': [],
        'psnr_zf': [],
        'ssim_zf': [],
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input, gt = batch
            input, gt = input.to(device), gt.to(device)
            pred = model(input)

            input = torch.abs(torch.complex(input[:, :1, ...], input[:, 1:, ...]))
            pred  = torch.abs(torch.complex(pred[:, :1, ...],  pred[:, 1:, ...]))
            gt    = torch.abs(torch.complex(gt[:, :1, ...],    gt[:, 1:, ...]))

            input = input.squeeze().float().cpu().numpy()
            pred  = pred.data.squeeze().float().cpu().numpy()
            gt    = gt.data.squeeze().float().cpu().numpy()

            metrics_dict['psnr'].append(calculate_psnr_single(pred,  gt, data_range=gt.max()))
            metrics_dict['ssim'].append(calculate_ssim_single(pred,  gt, data_range=gt.max()))
            metrics_dict['psnr_zf'].append(calculate_psnr_single(input, gt, data_range=gt.max()))
            metrics_dict['ssim_zf'].append(calculate_ssim_single(input, gt, data_range=gt.max()))

    return metrics_dict


def validate_one_step(model, dataloader):
    model.eval()

    with torch.no_grad():
        batch = next(iter(dataloader))
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)

        input_image     = torch.abs(torch.complex(x[0, 0, :, :],    x[0, 1, :, :])).cpu().numpy()
        ground_truth    = torch.abs(torch.complex(y[0, 0, :, :],    y[0, 1, :, :])).cpu().numpy()
        predicted_image = torch.abs(torch.complex(pred[0, 0, :, :], pred[0, 1, :, :])).cpu().numpy()

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(input_image, cmap='gray')
        plt.title("Input")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(predicted_image, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.show()
