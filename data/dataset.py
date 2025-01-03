import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import rgb2lab
import torch.amp as amp  # Add this import

class CFDataLoader(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None, dataset_type="unknown"):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        self.dataset_type = dataset_type
        self.image_files = []

        # Create a unique cache filename that includes the size limit
        cache_name = f'dataset_cache_{dataset_type}_{max_samples}_files.txt'
        self.cache_file = os.path.join(data_dir, cache_name)

        if os.path.exists(self.cache_file):
            # Load from cache if it exists
            print(f"Loading {dataset_type} dataset from cache...")
            with open(self.cache_file, 'r') as f:
                self.image_files = [line.strip() for line in f.readlines()]
        else:
            # Collect files and create cache
            print(f"Scanning {dataset_type} directory: {data_dir}")
            self._collect_files()
            self._create_cache()

        # Verify and apply size limit
        total_files = len(self.image_files)
        if max_samples and max_samples > 0:
            if max_samples < total_files:
                print(f"\nLimiting {dataset_type} dataset:")
                print(f"Total available: {total_files:,} images")
                print(f"Using random subset of {max_samples:,} images")
                # Shuffle and limit size
                random.seed(42)  # For reproducibility
                random.shuffle(self.image_files)
                self.image_files = self.image_files[:max_samples]
            else:
                print(f"\nRequested {max_samples:,} images but only {total_files:,} available")
                print(f"Using all available {total_files:,} images")

        print(f"Final {dataset_type} dataset size: {len(self.image_files):,} images")
        self.setup_device()

    def _collect_files(self):
        """Collect all valid image files recursively"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        for root, dirs, files in os.walk(self.data_dir):
            valid_images = [os.path.join(root, f) for f in files if f.lower().endswith(valid_extensions)]
            if valid_images:
                self.image_files.extend(valid_images)
                if root != self.data_dir:
                    print(f"  Found {len(valid_images):,} images in {os.path.basename(root)}")

    def _create_cache(self):
        """Create cache file with collected image paths"""
        if self.image_files:
            with open(self.cache_file, 'w') as f:
                f.write('\n'.join(self.image_files))
            print(f"Created cache with {len(self.image_files):,} images")
        else:
            raise ValueError(f"No valid images found in {self.data_dir}")

    def setup_device(self):
        """Setup device and transforms"""
        self.to_tensor = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Dataset using device: {self.device}")

    def __len__(self):
        return len(self.image_files)

    @amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = self.to_tensor(img)
                
            # Convert to LAB on CPU
            img_np = img.cpu().numpy() if img.is_cuda else img.numpy()
            img_np = img_np.transpose(1, 2, 0)
            lab_img = rgb2lab(img_np)
            
            L = torch.from_numpy(lab_img[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
            ab = torch.from_numpy(lab_img[:, :, 1:]).permute(2, 0, 1) / 128.0
            
            # Let PyTorch Lightning handle device movement
            return {'L': L.float(), 'ab': ab.float()}
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__(random.randint(0, len(self) - 1))