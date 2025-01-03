import torch
import matplotlib.pyplot as plt
import os
from skimage.color import lab2rgb
import numpy as np
import pytorch_lightning as pl

class ColorizeVisualizationCallback(pl.Callback):
    def __init__(self, output_dir, num_samples=3, save_freq=1):
        super().__init__()
        self.output_dir = os.path.join(output_dir, 'visualizations')
        self.num_samples = num_samples
        self.save_freq = save_freq
        os.makedirs(self.output_dir, exist_ok=True)
        self.validation_batch = None
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, *args, **kwargs):
        if batch_idx == 0 and self.validation_batch is None:
            # Store first validation batch for consistent visualization
            self.validation_batch = batch

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_freq == 0 and self.validation_batch is not None:
            self.visualize_batch(trainer, pl_module)

    def visualize_batch(self, trainer, pl_module):
        # Move model to eval mode
        pl_module.eval()
        with torch.no_grad():
            L = self.validation_batch['L'].to(pl_module.device)
            ab_true = self.validation_batch['ab'].to(pl_module.device)
            ab_pred = pl_module(L)

            # Convert to numpy for visualization
            L_np = L.cpu().numpy()
            ab_true_np = ab_true.cpu().numpy()
            ab_pred_np = ab_pred.cpu().numpy()

            # Create figure
            fig, axes = plt.subplots(self.num_samples, 3, figsize=(12, 4*self.num_samples))
            plt.subplots_adjust(hspace=0.3)

            for i in range(min(self.num_samples, L.size(0))):
                # Prepare images
                grayscale = L_np[i][0]  # Original grayscale
                true_color = self.lab_to_rgb(L_np[i], ab_true_np[i])
                pred_color = self.lab_to_rgb(L_np[i], ab_pred_np[i])

                # Plot images
                axes[i, 0].imshow(grayscale, cmap='gray')
                axes[i, 0].set_title('Grayscale Input')
                axes[i, 1].imshow(pred_color)
                axes[i, 1].set_title('Generated Color')
                axes[i, 2].imshow(true_color)
                axes[i, 2].set_title('Original Color')

                # Remove axes
                for ax in axes[i]:
                    ax.axis('off')

            # Add epoch information
            plt.suptitle(f'Epoch {trainer.current_epoch + 1}', y=0.95, fontsize=16)
            
            # Save figure
            fig_path = os.path.join(self.output_dir, f'epoch_{trainer.current_epoch + 1:03d}.png')
            plt.savefig(fig_path, bbox_inches='tight', dpi=150)
            plt.close()

        # Return to training mode
        pl_module.train()

    @staticmethod
    def lab_to_rgb(L, ab):
        """Convert L and ab channels to RGB image"""
        L = (L + 1.) * 50.  # [-1, 1] -> [0, 100]
        ab = ab * 128.      # [-1, 1] -> [-128, 128]
        Lab = np.concatenate([L[np.newaxis, ...], ab], axis=0)
        Lab = np.transpose(Lab, (1, 2, 0))
        return lab2rgb(Lab)
