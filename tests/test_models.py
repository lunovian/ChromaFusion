import unittest
import torch
from models.chroma_fusion import ChromaFusion

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = {
            'efficientnet_model_name': 'efficientnet_b3',
            'model_name': 'vit_base_patch16_224',
            'input_channels': 1,
            'hidden_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'bottleneck_in': 768,
            'bottleneck_out': 256,
            'decoder_out': 2,
            'pretrained': False,
            'image_size': 224
        }
    
    def test_model_creation(self):
        model = ChromaFusion(self.config)
        self.assertIsInstance(model, ChromaFusion)
    
    def test_forward_pass(self):
        model = ChromaFusion(self.config)
        input_tensor = torch.randn(1, 1, 224, 224)  # Match config image size
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 2, 224, 224))  # Match config image size
        
    def test_model_parameters(self):
        model = ChromaFusion(self.config)
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 0)

if __name__ == '__main__':
    unittest.main()
