import argparse
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from training.train import ColorizationModel
from torch.utils.data import DataLoader
from data.dataset import CFDataLoader  # Import CFDataLoader directly
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ProgressBar
import torch
import torch.cuda
from training.callbacks import DetailedProgressBar
import multiprocessing
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

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
    model.eval()
    metrics = {'psnr': [], 'ssim': []}
    
    with torch.no_grad():
        for batch in test_loader:
            L = batch['L'].to(device)
            ab_true = batch['ab'].to(device)
            ab_pred = model(L)
            
            # Calculate metrics
            ab_true_np = ab_true.cpu().numpy()
            ab_pred_np = ab_pred.cpu().numpy()
            
            for i in range(len(ab_true_np)):
                metrics['psnr'].append(psnr(ab_true_np[i], ab_pred_np[i], data_range=2.0))
                metrics['ssim'].append(ssim(ab_true_np[i], ab_pred_np[i], data_range=2.0, channel_axis=0))
    
    return {k: np.mean(v) for k, v in metrics.items()}

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

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

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

    # Initialize datasets with enforced size limits
    train_dataset = CFDataLoader(
        data_dir=args.train_dir,
        transform=train_transform,
        max_samples=args.num_train_images if args.num_train_images > 0 else None
    )

    val_dataset = CFDataLoader(
        data_dir=args.val_dir,
        transform=val_transform,
        max_samples=args.num_val_images if args.num_val_images > 0 else None
    )

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

    # Move model to device explicitly
    model = ColorizationModel(config).to(device)

    # Callbacks
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
        DetailedProgressBar()  # Replace default progress bar with detailed one
    ]

    # Add image saving callback if requested
    if args.save_images:
        callbacks.append(ImageSavingCallback(
            output_dir=os.path.join(args.output_dir, 'samples'),
            save_frequency=args.save_frequency
        ))

    # Logger with fallback
    try:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name='logs',
            version='latest'
        )
        print("Using TensorBoard logger")
    except (ImportError, ModuleNotFoundError):
        logger = CSVLogger(
            save_dir=args.output_dir,
            name='logs'
        )
        print("TensorBoard not available, using CSV logger instead")

    # Updated Trainer configuration
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
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

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluation phase if requested
    if args.evaluate:
        print("\nRunning evaluation...")
        metrics = evaluate_model(model, val_loader, device)
        print(f"Evaluation Results:")
        print(f"PSNR: {metrics['psnr']:.2f}")
        print(f"SSIM: {metrics['ssim']:.4f}")

    # Save the final model
    trainer.save_checkpoint(os.path.join(args.output_dir, 'checkpoints', 'final_model.ckpt'))

if __name__ == "__main__":
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

    args = parser.parse_args()
    
    main(args)