import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

def calculate_mse(pred, target):
    """Calculate Mean Squared Error"""
    return F.mse_loss(pred, target).item()

def calculate_metrics(pred, target, data_range=2.0):
    """Calculate evaluation metrics for colorization"""
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # Calculate MSE
    mse = np.mean((pred - target) ** 2)

    # Calculate PSNR
    try:
        psnr_val = peak_signal_noise_ratio(target, pred, data_range=data_range)
    except:
        psnr_val = 0.0

    # Calculate SSIM
    try:
        ssim_val = structural_similarity(target, pred, data_range=data_range, channel_axis=0)
    except:
        ssim_val = 0.0

    return {
        'mse': float(mse),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val)
    }

if __name__ == "__main__":
    print(f"Metrics for random images: {calculate_metrics(torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224))}")