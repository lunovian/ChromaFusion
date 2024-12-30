import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from models.chroma_fusion import ChromaFusion
from .metrics import calculate_metrics

def augmentations():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return train_transforms, val_transforms

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def find_train_val_folders(base_path):
    """
    Recursively searches for 'train' and 'val' folders starting from 'base_path'.

    Parameters:
    base_path (str): The path to start searching from.

    Returns:
    tuple: A tuple containing the paths to the 'train' and 'val' folders if found, otherwise raises FileNotFoundError.
    """
    for root, dirs, files in os.walk(base_path):
        if 'train' in dirs and 'val' in dirs:
            train_path = os.path.join(root, 'train')
            val_path = os.path.join(root, 'val')
            return train_path, val_path
    raise FileNotFoundError(f"Could not find 'train' and 'val' folders in '{base_path}' or its subdirectories.")

def check_required_folders(base_dir):
    """
    Check if the required folders exist in the base directory.

    Parameters:
    base_dir (str): The base directory to check.

    Raises:
    FileNotFoundError: If any of the required folders are not found.
    """
    required_folders = ['grayscale', 'original']
    for folder in required_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Required folder '{folder}' not found in {base_dir}")

class ColorizationModel(pl.LightningModule):
    def __init__(self, config):
        super(ColorizationModel, self).__init__()
        self.config = config
        self.model = ChromaFusion(config)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        L, AB = batch
        output = self.model(L)
        output = F.interpolate(output, size=AB.shape[2:], mode='bilinear', align_corners=False)
        loss = self.criterion(output, AB)
        self.log('train_loss', loss)
        self.log('train_batch_idx', batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        L, AB = batch
        output = self.model(L)
        output = F.interpolate(output, size=AB.shape[2:], mode='bilinear', align_corners=False)
        val_loss = self.criterion(output, AB)
        self.log('val_loss', val_loss)
        self.log('val_batch_idx', batch_idx)

        output_np = output.cpu().numpy()
        AB_np = AB.cpu().numpy()

        val_mse, val_psnr, val_ssim = 0, 0, 0
        for i in range(output_np.shape[0]):
            mse_value, psnr_value, ssim_value = calculate_metrics(AB_np[i], output_np[i])
            val_mse += mse_value
            val_psnr += psnr_value
            val_ssim += ssim_value

        self.log('val_mse', val_mse / len(batch))
        self.log('val_psnr', val_psnr / len(batch))
        self.log('val_ssim', val_ssim / len(batch))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }