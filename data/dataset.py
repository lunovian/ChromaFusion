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
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")
            
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        
        # Get all image files recursively from all subdirectories
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        category_count = 0
        
        print(f"Scanning directory: {data_dir}")
        
        self.cache_file = os.path.join(data_dir, 'dataset_cache.txt')
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.image_files = f.read().splitlines()
        else:
            for root, dirs, files in os.walk(data_dir):
                if root == data_dir:
                    print(f"Found {len(dirs)} categories")
                    category_count = len(dirs)
                    
                valid_images = [os.path.join(root, f) for f in files if f.lower().endswith(valid_extensions)]
                if valid_images:
                    self.image_files.extend(valid_images)
                    if root != data_dir:
                        print(f"  Found {len(valid_images)} images in {os.path.basename(root)}")
            
            if len(self.image_files) == 0:
                raise ValueError(f"No valid images found in directory: {data_dir}")
                
            print(f"Total: Found {len(self.image_files)} images across {category_count} categories")
            
            print(f"\nProcessing {dataset_type} dataset:")
            total_files = len(self.image_files)
            print(f"Found {total_files} total images")
            
            # Apply max_samples limit before caching
            if max_samples and max_samples > 0:
                if max_samples > total_files:
                    print(f"Warning: Requested {max_samples} images but only {total_files} available")
                    max_samples = total_files
                print(f"Limiting dataset to {max_samples} samples")
                # Shuffle files before limiting to get random subset
                random.shuffle(self.image_files)
                self.image_files = self.image_files[:max_samples]
                print(f"Final {dataset_type} dataset size: {len(self.image_files)} images")
            
            # Cache the limited dataset
            self.cache_file = os.path.join(data_dir, f'dataset_cache_{dataset_type}_{max_samples if max_samples else "full"}.txt')
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.image_files = f.read().splitlines()
            else:
                with open(self.cache_file, 'w') as f:
                    f.write('\n'.join(self.image_files))
        
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
                
            # Optimize LAB conversion
            img_np = img.permute(1, 2, 0).numpy()
            lab_img = rgb2lab(img_np)
            
            L = torch.from_numpy(lab_img[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
            ab = torch.from_numpy(lab_img[:, :, 1:]).permute(2, 0, 1) / 128.0
            
            return {'L': L.float(), 'ab': ab.float()}
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random valid index instead
            return self.__getitem__(random.randint(0, len(self) - 1))