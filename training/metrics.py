import numpy as np
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim

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
