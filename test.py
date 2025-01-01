import argparse
import torch
import os
from torch.utils.data import DataLoader
from data.dataset import CFDataLoader
import torchvision.transforms as transforms
from train import ColorizationModel
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.amp as amp
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from utils.color_conversion import combine_lab_and_convert_to_rgb

def lab_to_rgb(L, ab):
    """Convert L and ab channels to RGB image."""
    Lab = np.concatenate([L, ab], axis=0)  # Combine channels
    Lab = Lab.transpose(1, 2, 0)  # Change to HWC format
    # Convert to BGR and then RGB
    rgb = cv2.cvtColor(Lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return rgb

def save_visualization(L, ab_pred, ab_true, save_dir, idx):
    """Save grayscale, predicted, and ground truth images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    save_image(L, save_dir / f'sample_{idx}_gray.png')
    save_image(ab_pred, save_dir / f'sample_{idx}_predicted.png')
    save_image(ab_true, save_dir / f'sample_{idx}_original.png')
    
    # Convert tensors to numpy arrays and ensure correct shapes
    L_np = L[0].cpu().numpy()  # Shape: (1, H, W)
    if L_np.shape[0] == 1:  # If we have a channel dimension
        L_np = L_np[0]  # Remove it to get (H, W)
    
    ab_pred_np = ab_pred[0].cpu().numpy()  # Shape: (2, H, W)
    ab_true_np = ab_true[0].cpu().numpy()  # Shape: (2, H, W)
    
    # Convert to RGB for visualization using utility function
    rgb_pred = combine_lab_and_convert_to_rgb(L_np, ab_pred_np)
    rgb_true = combine_lab_and_convert_to_rgb(L_np, ab_true_np)
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(L_np, cmap='gray')
    axes[0].set_title('Grayscale Input')
    axes[1].imshow(rgb_pred)
    axes[1].set_title('Generated Color')
    axes[2].imshow(rgb_true)
    axes[2].set_title('Original Color')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{idx}_comparison.png')
    plt.close()

def evaluate_model(model, test_loader, device, args):
    model.eval()
    metrics = {'psnr': [], 'ssim': []}
    save_dir = os.path.join(args.output_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad(), amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            L = batch['L'].to(device)
            ab_true = batch['ab'].to(device)
            
            # Generate predictions
            ab_pred = model(L)
            
            # Save visualizations for the first N samples
            if idx < args.num_visualizations:
                save_visualization(
                    L=L, 
                    ab_pred=ab_pred, 
                    ab_true=ab_true, 
                    save_dir=save_dir,
                    idx=idx
                )
            
            # Calculate metrics
            ab_true_np = ab_true.cpu().numpy()
            ab_pred_np = ab_pred.cpu().numpy()
            
            # Batch processing of metrics
            for i in range(ab_true_np.shape[0]):
                metrics['psnr'].append(psnr(ab_true_np[i], ab_pred_np[i], data_range=2.0))
                metrics['ssim'].append(ssim(ab_true_np[i], ab_pred_np[i], data_range=2.0, channel_axis=0))

    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # Save metrics to file
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for k, v in avg_metrics.items():
            f.write(f'{k}: {v:.4f}\n')
    
    return avg_metrics

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    model = ColorizationModel.load_from_checkpoint(args.checkpoint_path)
    model = model.to(device)
    model.eval()

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load test dataset
    test_dataset = CFDataLoader(
        data_dir=args.test_dir,
        transform=transform,
        max_samples=args.num_test_images if args.num_test_images > 0 else None
    )

    # Fast DataLoader configuration
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Add output directory for test results
    args.output_dir = os.path.join(os.path.dirname(args.checkpoint_path), 'test_results')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    metrics = evaluate_model(model, test_loader, device, args)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the colorization model')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_test_images', type=int, default=0, help='Number of test images (0 for all)')
    parser.add_argument('--num_visualizations', type=int, default=5, 
                       help='Number of test images to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save test results (default: checkpoint_dir/test_results)')
    
    args = parser.parse_args()
    main(args)
