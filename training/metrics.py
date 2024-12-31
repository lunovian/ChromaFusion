import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def calculate_metrics(original_image, colorized_image):
    # Ensure images are in the expected data range (0 to 1)
    original_image = np.clip(original_image, 0, 1)
    colorized_image = np.clip(colorized_image, 0, 1)
    
    mse_value = mse(original_image, colorized_image)
    
    # Add a small epsilon to avoid log(0) in PSNR calculation
    psnr_value = psnr(original_image, colorized_image, data_range=1.0)
    
    # Ensure the win_size is appropriate for the image dimensions and is odd
    min_dim = min(original_image.shape[-2], original_image.shape[-1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:  # Ensure win_size is odd
        win_size -= 1
    
    # Avoid division by zero by ensuring win_size is at least 3
    if win_size < 3:
        win_size = 3

    # Ensure win_size does not exceed the dimensions of the image
    win_size = min(win_size, min_dim)

    # Calculate SSIM
    ssim_value = ssim(original_image, colorized_image, win_size=win_size, data_range=1.0, channel_axis=0)
    
    return mse_value, psnr_value, ssim_value

def calculate_metrics_parallel(pred_batch, true_batch, max_workers=4):
    pred_np = pred_batch.cpu().numpy()
    true_np = true_batch.cpu().numpy()
    
    def calc_metrics(idx):
        pred = pred_np[idx]
        true = true_np[idx]
        psnr = peak_signal_noise_ratio(true, pred, data_range=2.0)
        ssim = structural_similarity(true, pred, data_range=2.0, channel_axis=0)
        return psnr, ssim
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(calc_metrics, range(pred_np.shape[0])))
    
    psnr_values, ssim_values = zip(*results)
    return np.mean(psnr_values), np.mean(ssim_values)

if __name__ == "__main__":
    print(f"Metrics for random images: {calculate_metrics(np.random.rand(3, 224, 224), np.random.rand(3, 224, 224))}")