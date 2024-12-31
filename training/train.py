import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from models.chroma_fusion import ChromaFusion
from .metrics import calculate_metrics
import torch.amp as amp  # Add this import

class SaveImageCallback(pl.Callback):
    """Callback for saving images during training"""
    def __init__(self, output_dir, save_frequency):
        super().__init__()
        self.output_dir = output_dir
        self.save_frequency = save_frequency

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.save_frequency == 0:
            L = batch['L']
            ab_true = batch['ab']
            ab_pred = pl_module(L)
            
            prefix = f"epoch_{trainer.current_epoch}_batch_{batch_idx}"
            save_image(L, f'{self.output_dir}/{prefix}_gray.png')
            save_image(ab_pred, f'{self.output_dir}/{prefix}_pred.png')
            save_image(ab_true, f'{self.output_dir}/{prefix}_true.png')

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
        super().__init__()
        self.save_hyperparameters()
        self.model = ChromaFusion(config)
        self.criterion = nn.SmoothL1Loss()
        
        # Fix GradScaler initialization
        self.scaler = torch.amp.GradScaler(
            enabled=(torch.cuda.is_available() and config.get('precision', 32) == 16)
        )  # Remove device_type parameter
        
        # Training parameters
        self.warmup_epochs = 5
        self.base_lr = config['learning_rate']
        self.min_lr = self.base_lr * 0.001
        self.clip_grad_val = config.get('gradient_clip_val', 1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        L = batch['L']
        ab = batch['ab']
        
        # Ensure inputs require gradients
        L = L.float().requires_grad_()
        ab = ab.float().requires_grad_()
        
        # Ensure input is float tensor and in valid range
        L = torch.clamp(L.float(), min=-1.0, max=1.0)
        ab = torch.clamp(ab.float(), min=-1.0, max=1.0)
        
        # Check for NaN inputs
        if torch.isnan(L).any() or torch.isnan(ab).any():
            raise ValueError("NaN detected in input tensors")
        
        with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            pred_ab = self.model(L)
            loss = self.criterion(pred_ab, ab)
        
        # Check for NaN loss
        if torch.isnan(loss):
            raise ValueError("NaN loss detected")
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        L = batch['L']
        ab = batch['ab']
        
        # Ensure inputs require gradients even in validation
        L = L.float().requires_grad_()
        ab = ab.float().requires_grad_()
        
        # Ensure input is float tensor
        L = L.float()
        ab = ab.float()
        
        pred_ab = self.model(L)
        loss = self.criterion(pred_ab, ab)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        scheduler = {
            'scheduler': lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.base_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # Warmup period
                div_factor=25.0,  # Initial learning rate = max_lr/25
                final_div_factor=1000.0,  # Min learning rate = max_lr/1000
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        super().optimizer_step(*args, **kwargs)

def main(args):
    # ...existing setup code...

    # Update callbacks list with proper SaveImageCallback
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='colorization-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        DetailedProgressBar()
    ]

    # Add image saving callback if requested
    if args.save_images:
        callbacks.append(SaveImageCallback(args.output_dir, args.save_frequency))

    # Updated Trainer configuration
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        # ...rest of existing trainer config...
    )

    # ...rest of existing code...