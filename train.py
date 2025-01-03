import argparse
import os
import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
from data.dataset import CFDataLoader  # Import CFDataLoader directly
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ProgressBar, TQDMProgressBar
import torch
import torch.cuda
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.amp as amp
from models.chroma_fusion import ChromaFusion
import multiprocessing
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from training import (
    calculate_metrics,
    ColorizeVisualizationCallback
)

# Add Tensor Cores optimization
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')
    print("Enabled Tensor Cores optimization with high precision")

def get_optimal_worker_count():
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count > 0:
        # For GPU training, use CPU count - 2 (leave cores for system and GPU communication)
        optimal_workers = max(1, min(cpu_count - 2, 8))
    else:
        # For CPU training, use CPU count - 1 (leave one core for system)
        optimal_workers = max(1, cpu_count - 1)
    
    return optimal_workers

def save_results(model, batch, output_dir, batch_idx, is_train=True):
    L = batch['L']
    ab_true = batch['ab']
    ab_pred = model(L)
    
    # Convert to images and save
    prefix = 'train' if is_train else 'val'
    save_image(L, f'{output_dir}/{prefix}_gray_{batch_idx}.png')
    save_image(ab_pred, f'{output_dir}/{prefix}_pred_{batch_idx}.png')
    save_image(ab_true, f'{output_dir}/{prefix}_true_{batch_idx}.png')

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset with proper device handling."""
    model.eval()
    model = model.to(device)  # Ensure model is on correct device
    metrics_sum = {'mse': 0, 'psnr': 0, 'ssim': 0}
    batch_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            L = batch['L'].to(device)
            ab_true = batch['ab'].to(device)
            
            # Forward pass
            with torch.amp.autocast(enabled=device.type == 'cuda'):
                ab_pred = model(L)
            
            # Calculate metrics using imported calculate_metrics
            batch_metrics = calculate_metrics(ab_pred, ab_true)
            
            # Accumulate metrics
            for k in metrics_sum.keys():
                metrics_sum[k] += batch_metrics[k]
            batch_count += 1
    
    # Calculate averages
    return {k: v / batch_count for k, v in metrics_sum.items()}

class ImageSavingCallback(pl.Callback):
    def __init__(self, output_dir, save_frequency):
        super().__init__()
        self.output_dir = output_dir
        self.save_frequency = save_frequency
        os.makedirs(output_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.save_frequency == 0:
            with torch.no_grad():
                L = batch['L']
                ab_true = batch['ab']
                ab_pred = pl_module(L)
                
                # Save images
                for i in range(min(3, L.size(0))):  # Save first 3 images from batch
                    prefix = f"epoch_{trainer.current_epoch}_batch_{batch_idx}_sample_{i}"
                    save_image(L[i:i+1], f'{self.output_dir}/{prefix}_gray.png')
                    save_image(ab_pred[i:i+1], f'{self.output_dir}/{prefix}_pred.png')
                    save_image(ab_true[i:i+1], f'{self.output_dir}/{prefix}_true.png')

# Move ColorizationModel class here from training/train.py
class ColorizationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        # Remove the device assignment since LightningModule handles this
        self.model = ChromaFusion(config)
        self.criterion = nn.SmoothL1Loss()
        self.scaler = torch.amp.GradScaler(
            enabled=(torch.cuda.is_available() and config.get('precision', 32) == 16)
        )
        
        # Training parameters
        self.warmup_epochs = 5
        self.base_lr = config['learning_rate']
        self.min_lr = self.base_lr * 0.001
        self.clip_grad_val = config.get('gradient_clip_val', 1.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        L = batch['L'].to(self.device)
        ab = batch['ab'].to(self.device)
        
        L = torch.clamp(L.float(), min=-1.0, max=1.0)
        ab = torch.clamp(ab.float(), min=-1.0, max=1.0)
        
        with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            pred_ab = self.model(L)
            loss = self.criterion(pred_ab, ab)
        
        self.log('train_loss', loss, prog_bar=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        L = batch['L'].to(self.device)
        ab = batch['ab'].to(self.device)
        
        with torch.no_grad():
            pred_ab = self.model(L)
            metrics = calculate_metrics(pred_ab, ab)
        
        self.log('val_loss', metrics['mse'], prog_bar=True)
        self.log('val_psnr', metrics['psnr'], prog_bar=True)
        self.log('val_ssim', metrics['ssim'], prog_bar=True)
        
        return metrics['mse']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Calculate total steps with a safety margin
        total_steps = self.trainer.estimated_stepping_batches
        if hasattr(self.trainer, 'max_epochs') and hasattr(self.trainer, 'num_training_batches'):
            # Add 10% safety margin to prevent stepping beyond total steps
            total_steps = int(self.trainer.num_training_batches * self.trainer.max_epochs * 1.1)
        
        scheduler = {
            'scheduler': lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.base_lr,
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1000.0,
                three_phase=True,  # Add three-phase for smoother transition
            ),
            'interval': 'step',
            'frequency': 1,
            'strict': False,  # Add this to prevent strict step checking
        }
        
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # Reset step count if needed
        if hasattr(self, 'scheduler_step_count'):
            current_epoch = self.trainer.current_epoch
            expected_steps = self.trainer.num_training_batches * (current_epoch + 1)
            if self.scheduler_step_count > expected_steps:
                print(f"Warning: Resetting scheduler step count from {self.scheduler_step_count} to {expected_steps}")
                self.scheduler_step_count = expected_steps

    def optimizer_step(self, *args, **kwargs):
        # Track scheduler steps
        if not hasattr(self, 'scheduler_step_count'):
            self.scheduler_step_count = 0
        self.scheduler_step_count += 1

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        super().optimizer_step(*args, **kwargs)

class DetailedProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if trainer.training:
            if 'loss' in trainer.callback_metrics:
                items["loss"] = f"{trainer.callback_metrics['loss']:.3f}"
            if 'lr' in trainer.callback_metrics:
                items["lr"] = f"{trainer.callback_metrics['lr']:.2e}"
        elif trainer.validating:
            if 'val_loss' in trainer.callback_metrics:
                items["val_loss"] = f"{trainer.callback_metrics['val_loss']:.3f}"
        return items

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        super().on_train_epoch_start(trainer, *args, **kwargs)
        print(f"\nEpoch {trainer.current_epoch}")

def get_args():
    parser = argparse.ArgumentParser(description='Train the colorization model on a specified dataset.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--display_step', type=int, default=1, help='Frequency of displaying images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Size of input images')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers (0 for auto)')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--num_train_images', type=int, default=0, help='Number of training images (0 for all)')
    parser.add_argument('--num_val_images', type=int, default=0, help='Number of validation images (0 for all)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with limited dataset')
    parser.add_argument('--limit_batches', type=float, default=0.1, help='Limit batches in debug mode')
    parser.add_argument('--save_images', action='store_true', help='Save generated images during training')
    parser.add_argument('--save_frequency', type=int, default=100, help='How often to save images')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    parser.add_argument('--visualization_frequency', type=int, default=1,
                       help='Frequency of saving visualization images (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g. output/checkpoints/final_model.ckpt)')
    parser.add_argument('--start_epoch', type=int, default=None,
                       help='Epoch to start from when resuming (overrides checkpoint)')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer state when resuming training')
    
    return parser.parse_args()

def main(args):
    # Add CUDA detection and device setup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")

    # Validate directories
    if not os.path.exists(args.train_dir):
        sys.exit(f"Error: Training directory not found: {args.train_dir}")
    if not os.path.exists(args.val_dir):
        sys.exit(f"Error: Validation directory not found: {args.val_dir}")

    # Create versioned output directory
    version = 0
    while os.path.exists(os.path.join(args.output_dir, f'version_{version}')):
        version += 1
    version_dir = os.path.join(args.output_dir, f'version_{version}')
    checkpoint_dir = os.path.join(version_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Output directory: {version_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    print(f"Training directory: {args.train_dir}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Output directory: {args.output_dir}")

    config = {
        'model_name': 'vit_base_patch16_224',
        'efficientnet_model_name': 'efficientnet_b3',
        'input_channels': 1,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 4,
        'bottleneck_in': 768,
        'bottleneck_out': 256,
        'decoder_out': 2,
        'pretrained': True,
        'learning_rate': args.learning_rate,
        'display_step': args.display_step,
        'batch_size': args.batch_size,
        'strict_load': False,  # Add this to handle unexpected keys
    }

    # Update precision handling
    if device.type == 'cuda':
        precision = '16-mixed' if args.precision == 16 else '32'
        accelerator = 'gpu'
    else:
        precision = 'bf16-mixed' if args.precision == 16 else '32'
        accelerator = 'cpu'
    
    print(f"Using {accelerator} with precision={precision}")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Apply dataset size limits first
    if args.num_train_images > 0:
        print(f"\nLimiting training dataset to {args.num_train_images} images")
    if args.num_val_images > 0:
        print(f"Limiting validation dataset to {args.num_val_images} images")

    # Initialize datasets with size validation
    train_dataset = CFDataLoader(
        data_dir=args.train_dir,
        transform=train_transform,
        max_samples=args.num_train_images,
        dataset_type="train"
    )

    val_dataset = CFDataLoader(
        data_dir=args.val_dir,
        transform=val_transform,
        max_samples=args.num_val_images,
        dataset_type="validation"
    )

    # Verify final dataset sizes
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    print("\nDataset Summary:")
    print(f"Training images: {train_size:,}")
    print(f"Validation images: {val_size:,}")
    
    # Validate dataset sizes
    min_train_images = 100  # Minimum recommended training images
    min_val_images = 10     # Minimum recommended validation images
    
    if train_size < min_train_images:
        print(f"\nWarning: Training dataset size ({train_size}) is smaller than recommended minimum ({min_train_images})")
    if val_size < min_val_images:
        print("Warning: Validation dataset size ({val_size}) is smaller than recommended minimum ({min_val_images})")
    
    # Calculate and display ratio
    val_ratio = val_size / train_size if train_size > 0 else 0
    print(f"Validation/Training ratio: {val_ratio:.2%}")
    
    if val_ratio > 0.3:
        print("Warning: Validation set might be too large compared to training set")
    elif val_ratio < 0.1:
        print("Warning: Validation set might be too small compared to training set")

    # Verify dataset sizes
    actual_train_size = len(train_dataset)
    actual_val_size = len(val_dataset)
    
    if args.num_train_images > 0 and actual_train_size > args.num_train_images:
        print(f"Warning: Training dataset size {actual_train_size} exceeds requested size {args.num_train_images}")
    if args.num_val_images > 0 and actual_val_size > args.num_val_images:
        print(f"Warning: Validation dataset size {actual_val_size} exceeds requested size {args.num_val_images}")

    print(f"\nFinal dataset sizes:")
    print(f"Training samples: {actual_train_size}")
    print(f"Validation samples: {actual_val_size}")

    # Calculate accurate steps per epoch
    steps_per_epoch = actual_train_size // args.batch_size
    print(f"Steps per epoch: {steps_per_epoch}")

    # Remove the debug mode dataset limiting since we handle it explicitly
    if args.debug:
        args.limit_batches = 0.1
        print(f"Debug mode: Using {args.limit_batches * 100}% of batches per epoch")

    # Override num_workers with optimal count if not explicitly set
    if args.num_workers <= 0:
        args.num_workers = get_optimal_worker_count()
        print(f"Using {args.num_workers} workers for data loading")

    # Calculate optimal prefetch factor based on batch size and workers
    optimal_prefetch = max(2, min(4, args.batch_size // args.num_workers))
    
    # Optimize DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=optimal_prefetch,  # Updated prefetch factor
        drop_last=True  # Drop incomplete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=optimal_prefetch  # Updated prefetch factor
    )

    # Initialize model
    model = ColorizationModel(config)
    
    # Load checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['state_dict'])
        
        # Optionally reset optimizer state
        if not args.reset_optimizer and 'optimizer_states' in checkpoint:
            print("Restoring optimizer state from checkpoint")
            model.optimizer_states = checkpoint['optimizer_states']
        
        # Set starting epoch
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        else:
            start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Enhanced checkpoint callback without unsupported metadata
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}-{val_loss:.4f}-{val_psnr:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )

    # Add metadata saving callback
    class MetadataCallback(pl.Callback):
        def __init__(self, checkpoint_dir, config):
            super().__init__()
            self.checkpoint_dir = checkpoint_dir
            self.config = config

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            metadata = {
                'architecture': 'ChromaFusion',
                'dataset_size': len(train_dataset),
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'precision': args.precision,
                'created_at': datetime.datetime.now().isoformat(),
                'current_epoch': trainer.current_epoch,
                'current_step': trainer.global_step,
            }
            checkpoint['metadata'] = metadata

    # Add a callback to save final model with metadata
    class FinalModelCallback(pl.Callback):
        def on_train_end(self, trainer, pl_module):
            final_path = os.path.join(checkpoint_dir, 'final_model.ckpt')
            metadata = {
                'architecture': 'ChromaFusion',
                'total_epochs': trainer.current_epoch + 1,
                'final_val_loss': float(trainer.callback_metrics.get('val_loss', 0)),
                'final_val_psnr': float(trainer.callback_metrics.get('val_psnr', 0)),
                'final_val_ssim': float(trainer.callback_metrics.get('val_ssim', 0)),
                'training_completed': datetime.datetime.now().isoformat()
            }
            trainer.save_checkpoint(final_path, weights_only=False)
            
            # Save metadata separately for easy access
            with open(os.path.join(checkpoint_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nFinal model saved to: {final_path}")
            print("Final metrics:")
            for k, v in metadata.items():
                if k.startswith('final_'):
                    print(f"{k[6:]}: {v}")

    # Update trainer initialization with both callbacks
    trainer = pl.Trainer(
        max_epochs=args.epochs + start_epoch,
        callbacks=[
            checkpoint_callback,
            MetadataCallback(checkpoint_dir, config),
            FinalModelCallback(),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=args.early_stopping_patience,
                mode='min',
                min_delta=1e-4,  # Minimum improvement required
                verbose=True
            ),
            # ...other existing callbacks...
            LearningRateMonitor(logging_interval='epoch'),
            DetailedProgressBar(),  # Replace default progress bar with detailed one
            ColorizeVisualizationCallback(
                output_dir=args.output_dir,
                num_samples=3,
                save_freq=args.visualization_frequency
            )
        ],
        logger=TensorBoardLogger(
            save_dir=version_dir,
            name='logs',
            version='latest',
            default_hp_metric=False,  # Disable default hp logging
            log_graph=True  # Log model graph
        ),
        accelerator=accelerator,
        devices=1 if device.type == 'cuda' else None,
        precision=precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,  # Reduce logging frequency
        deterministic=True,  # Add for reproducibility
        enable_progress_bar=True,
        detect_anomaly=args.debug,  # Enable anomaly detection in debug mode
        limit_train_batches=args.limit_batches if args.debug else 1.0,  # Limit batches in debug mode
        limit_val_batches=args.limit_batches if args.debug else 1.0
    )

    # Train the model with ckpt_path parameter
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=args.resume if not args.reset_optimizer else None
    )

    # Evaluation phase if requested
    if args.evaluate:
        print("\nRunning evaluation...")
        # Ensure model is in eval mode and on correct device
        model.eval()
        model = model.to(device)
        metrics = evaluate_model(model, val_loader, device)
        print(f"Evaluation Results:")
        print(f"PSNR: {metrics['psnr']:.2f}")
        print(f"SSIM: {metrics['ssim']:.4f}")

    # Save the final model
    trainer.save_checkpoint(os.path.join(args.output_dir, 'checkpoints', 'final_model.ckpt'))

if __name__ == "__main__":
    # Add necessary imports at the top of the file
    import datetime
    import json
    args = get_args()
    main(args)