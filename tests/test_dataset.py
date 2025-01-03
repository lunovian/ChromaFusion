import unittest
import os
import sys
import shutil
from PIL import Image
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CFDataLoader
from torchvision import transforms

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create proper directory structure for train/val
        self.data_dir = os.path.join('data', 'sample', 'train')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create multiple sample images
        for i in range(5):  # Create 5 test images
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.data_dir, f'test_image_{i}.jpg'))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def tearDown(self):
        # Clean up the entire sample directory
        if os.path.exists('data/sample'):
            shutil.rmtree('data/sample')
    
    def test_dataset_loading(self):
        dataset = CFDataLoader(
            data_dir='data/sample',  # Point to parent directory
            transform=self.transform,
            dataset_type="train"
        )
        self.assertEqual(len(dataset), 5)  # Should match number of created images
    
    def test_dataset_item(self):
        dataset = CFDataLoader(
            data_dir='data/sample',  # Point to parent directory
            transform=self.transform,
            dataset_type="train"
        )
        item = dataset[0]
        self.assertIn('L', item)
        self.assertIn('ab', item)
        self.assertEqual(item['L'].shape, (1, 224, 224))
        self.assertEqual(item['ab'].shape, (2, 224, 224))

if __name__ == '__main__':
    unittest.main()
