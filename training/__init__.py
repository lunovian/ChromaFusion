import torch
import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
import os

def calculate_metrics(pred, target):
    """Calculate MSE, PSNR, and SSIM metrics."""
    mse = mse_loss(pred, target).item()
    
    # Convert tensors to numpy arrays for skimage metrics
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Calculate PSNR and SSIM for each image in batch
    psnr_vals = []
    ssim_vals = []
    
    for i in range(pred_np.shape[0]):
        psnr_vals.append(psnr(target_np[i], pred_np[i], data_range=2.0))
        ssim_vals.append(ssim(target_np[i], pred_np[i], data_range=2.0, channel_axis=0))
    
    return {
        'mse': mse,
        'psnr': sum(psnr_vals) / len(psnr_vals),
        'ssim': sum(ssim_vals) / len(ssim_vals)
    }

class ColorizeVisualizationCallback(pl.Callback):
    def __init__(self, output_dir, num_samples=3, save_freq=1):
        super().__init__()
        self.output_dir = os.path.join(output_dir, 'visualizations')
        self.num_samples = num_samples
        self.save_freq = save_freq
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.save_freq != 0:
            return
            
        # Get validation dataloader correctly
        val_dataloader = trainer.val_dataloaders
        if isinstance(val_dataloader, (list, tuple)):
            val_dataloader = val_dataloader[0]
        
        try:
            # Get a batch from validation dataloader
            batch = next(iter(val_dataloader))
            L = batch['L'][:self.num_samples].to(pl_module.device)
            ab_true = batch['ab'][:self.num_samples].to(pl_module.device)
            
            with torch.no_grad():
                ab_pred = pl_module(L)
                
                # Save images
                for i in range(self.num_samples):
                    prefix = f"epoch_{trainer.current_epoch}_sample_{i}"
                    save_image(L[i:i+1], f'{self.output_dir}/{prefix}_gray.png')
                    save_image(ab_pred[i:i+1], f'{self.output_dir}/{prefix}_pred.png')
                    save_image(ab_true[i:i+1], f'{self.output_dir}/{prefix}_true.png')
        except StopIteration:
            print("Warning: Could not get validation batch for visualization")
        except Exception as e:
            print(f"Warning: Error during visualization: {str(e)}")
