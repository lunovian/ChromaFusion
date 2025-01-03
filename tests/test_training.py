import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import ColorizationModel

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {
            'learning_rate': 1e-4,
            'model_name': 'vit_base_patch16_224',
            'input_channels': 1,
            'hidden_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'efficientnet_model_name': 'efficientnet_b3',
            'bottleneck_in': 768,
            'bottleneck_out': 256,
            'decoder_out': 2,
            'pretrained': False,
            'image_size': 224  # Add explicit image size
        }
        self.model = ColorizationModel(self.config)
        self.model = self.model.to(self.device)  # Move model to device
        
        # Mock trainer and logging
        self.model.trainer = MagicMock()
        self.model.trainer.estimated_stepping_batches = 1000
        # Disable logging for tests
        self.model.log = MagicMock()
        
    @patch('pytorch_lightning.LightningModule.log')
    def test_training_step(self, mock_log):
        batch = {
            'L': torch.randn(4, 1, 224, 224).to(self.device),  # Move to device
            'ab': torch.randn(4, 2, 224, 224).to(self.device)  # Move to device
        }
        loss = self.model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Loss should be a scalar

    @patch('pytorch_lightning.LightningModule.log')
    def test_validation_step(self, mock_log):
        batch = {
            'L': torch.randn(4, 1, 224, 224).to(self.device),
            'ab': torch.randn(4, 2, 224, 224).to(self.device)
        }
        with torch.no_grad():
            val_loss = self.model.validation_step(batch, 0)
        # Check that val_loss is a float (MSE loss)
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0.0)  # MSE should be non-negative

    def test_configure_optimizers(self):
        optimizers = self.model.configure_optimizers()
        self.assertIsInstance(optimizers[0][0], torch.optim.Optimizer)
        self.assertIsInstance(optimizers[1][0], dict)  # Scheduler config

if __name__ == '__main__':
    unittest.main()
