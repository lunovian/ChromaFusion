import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.dataset import CFDataLoader  # Change this import
from skimage.color import rgb2lab, lab2rgb

def combine_lab_and_convert_to_rgb(L, AB):
    """
    Combine L and AB channels to form LAB image and convert to RGB.

    Args:
        L (numpy.ndarray or torch.Tensor): The L channel tensor.
        AB (numpy.ndarray or torch.Tensor): The AB channels tensor.

    Returns:
        numpy.ndarray: The converted RGB image.
    """
    if isinstance(L, torch.Tensor):
        L = L.squeeze().cpu().numpy()
    if isinstance(AB, torch.Tensor):
        AB = AB.squeeze().cpu().numpy()

    # Check the shapes of L and AB
    if L.ndim != 2:
        raise ValueError(f"L channel should be a 2D array, but got shape {L.shape}")
    if AB.ndim != 3 or AB.shape[0] != 2:
        raise ValueError(f"AB channels should be a 3D array with shape (2, H, W), but got shape {AB.shape}")

    # Scale L channel from [-1, 1] to [0, 100]
    L = (L + 1.0) * 50.0
    # Scale AB channels from [-1, 1] to [-128, 127]
    AB = AB * 128.0

    # Ensure LAB values are within the correct ranges
    L = np.clip(L, 0, 100)
    AB = np.clip(AB, -128, 127)

    # Combine L and AB channels to form LAB image and ensure correct shape (H, W, 3)
    LAB = np.stack((L, AB[0], AB[1]), axis=-1)
    
    # Convert LAB to RGB
    RGB = lab2rgb(LAB)
    
    # Scale RGB values to [0, 255] and convert to uint8
    RGB = (RGB * 255).astype(np.uint8)
    
    return RGB

def load_and_visualize_sample(dataset_path, model, device):
    # Ensure the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset
    val_dataset_path = os.path.join(dataset_path, 'val')
    dataset = CFDataLoader(root_dir=val_dataset_path, dataset_type='val', transform=val_transforms)

    # Select a random index
    random_index = random.randint(0, len(dataset) - 1)
    L, AB = dataset[random_index]
    
    # Move the data to the appropriate device and ensure no batch dimension
    L, AB = L.to(device).unsqueeze(0), AB.to(device).unsqueeze(0)  # Add batch dimension
    L, AB = L.squeeze(0), AB.squeeze(0)  # Remove batch dimension

    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        output = model(L.unsqueeze(0))  # Add batch dimension for the model
        output = F.interpolate(output, size=AB.shape[1:], mode='bilinear', align_corners=False).squeeze(0)  # Remove batch dimension
    
    # Convert LAB to RGB for visualization using the provided function
    output_rgb = combine_lab_and_convert_to_rgb(L, output)
    AB_rgb = combine_lab_and_convert_to_rgb(L, AB)
    grayscale_img = (L.cpu().numpy() + 1.0) * 50.0  # Denormalize L to [0, 100]
    grayscale_img = (grayscale_img / 100.0 * 255).astype(np.uint8)  # Convert L to grayscale [0, 255]

    # Ensure grayscale_img has correct shape for imshow
    if grayscale_img.ndim == 3 and grayscale_img.shape[0] == 1:
        grayscale_img = grayscale_img.squeeze(0)

    # Plot the images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(grayscale_img, cmap='gray')
    ax[0].set_title('Grayscale')
    ax[0].axis('off')
    
    ax[1].imshow(output_rgb)
    ax[1].set_title('Colorized Output')
    ax[1].axis('off')
    
    ax[2].imshow(AB_rgb)
    ax[2].set_title('Ground Truth')
    ax[2].axis('off')
    
    plt.show()