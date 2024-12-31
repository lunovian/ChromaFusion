import argparse
import torch
import os
from torch.utils.data import DataLoader
from data.dataset import CFDataLoader
import torchvision.transforms as transforms
from training.train import ColorizationModel
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.amp as amp

def evaluate_model(model, test_loader, device, args):
    model.eval()
    metrics = {'psnr': [], 'ssim': []}
    
    with torch.no_grad(), amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for batch in tqdm(test_loader, desc="Evaluating"):
            L = batch['L'].to(device)
            ab_true = batch['ab'].to(device)
            
            # Generate predictions
            ab_pred = model(L)
            
            # Calculate metrics
            ab_true_np = ab_true.cpu().numpy()
            ab_pred_np = ab_pred.cpu().numpy()
            
            # Batch processing of metrics
            for i in range(ab_true_np.shape[0]):
                metrics['psnr'].append(psnr(ab_true_np[i], ab_pred_np[i], data_range=2.0))
                metrics['ssim'].append(ssim(ab_true_np[i], ab_pred_np[i], data_range=2.0, channel_axis=0))

    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
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

    # Run evaluation
    metrics = evaluate_model(model, test_loader, device, args)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the colorization model')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_test_images', type=int, default=0, help='Number of test images (0 for all)')
    
    args = parser.parse_args()
    main(args)
