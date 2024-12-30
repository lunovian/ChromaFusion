import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb

class DataLoader(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        # Construct paths based on the dataset type (train/val)
        grayscale_path = os.path.join(self.root_dir, 'grayscale')
        color_path = os.path.join(self.root_dir, 'original')

        if not os.path.exists(grayscale_path) or not os.path.exists(color_path):
            raise FileNotFoundError(f"Could not find directories: {grayscale_path} or {color_path}")

        data = []
        for filename in os.listdir(grayscale_path):
            grayscale_img_path = os.path.join(grayscale_path, filename)
            color_img_path = os.path.join(color_path, filename)
            if os.path.exists(grayscale_img_path) and os.path.exists(color_img_path):
                data.append((grayscale_img_path, color_img_path))
            else:
                print(f"Missing pair for {filename}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        grayscale_img_path, color_img_path = self.data[idx]
        grayscale_img = self._load_image(grayscale_img_path)
        color_img = self._load_image(color_img_path)

        if self.transform:
            # Apply transforms to both grayscale and color images
            grayscale_img = self.transform(grayscale_img)
            color_img = self.transform(color_img)

        # Convert the transformed color image to numpy array and ensure the shape is (height, width, 3)
        color_img_np = np.array(color_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        # Convert RGB color image to LAB
        lab = rgb2lab(color_img_np)
        L = lab[:, :, 0] / 50.0 - 1.0  # Normalize L channel to [-1, 1]
        AB = lab[:, :, 1:] / 128.0  # Normalize AB channels to [-1, 1]

        L = torch.tensor(L).unsqueeze(0).float()  # Add channel dimension and convert to tensor
        AB = torch.tensor(AB).permute(2, 0, 1).float()  # Convert to tensor and permute dimensions

        return L, AB

    def _load_image(self, img_path):
        return Image.open(img_path).convert('RGB')  # Load image as RGB