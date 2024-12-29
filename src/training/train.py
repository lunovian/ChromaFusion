import os
import torch
from torch.utils.data import DataLoader
from src.models.chroma_fusion import HybridColorize
from data.dataset import HybridConvViTDataset
from data.augmentations import get_augmentations
from data.utils import show_image

def list_available_datasets(base_dir):
    """
    List available datasets in the base data directory.
    """
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def check_required_folders(base_dir):
    """
    Check if the required folders are present in the dataset directory.
    """
    required_main_folders = ['train', 'val']
    required_sub_folders = ['original', 'grayscale']

    for main_folder in required_main_folders:
        main_folder_path = os.path.join(base_dir, main_folder)
        if not os.path.exists(main_folder_path):
            raise FileNotFoundError(f"Required folder '{main_folder}' not found in {base_dir}")
        for sub_folder in required_sub_folders:
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
            if not os.path.exists(sub_folder_path):
                raise FileNotFoundError(f"Required folder '{sub_folder}' not found in {main_folder_path}")

def train(dataset_name):
    # Define the base directory for the dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Check if the dataset exists
    while dataset_name not in list_available_datasets(base_dir):
        print(f"Dataset '{dataset_name}' not found.")
        print("Available datasets:", ", ".join(list_available_datasets(base_dir)))
        dataset_name = input("Enter the correct dataset name to train on: ").strip().lower()
    
    dataset_dir = os.path.join(base_dir, dataset_name)
    
    # Check if required folders exist
    check_required_folders(dataset_dir)

    # Load augmentations
    train_transforms, val_transforms = get_augmentations()
    
    # Load datasets
    train_dataset = HybridConvViTDataset(root_dir=dataset_dir, dataset_type='train', transform=train_transforms)
    val_dataset = HybridConvViTDataset(root_dir=dataset_dir, dataset_type='val', transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = HybridColorize(
        image_size=256,
        in_channels=1,
        hidden_dim=512,
        num_heads=8,
        num_layers=12
    )
    
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            grayscale_img, color_img = batch
            optimizer.zero_grad()
            output = model(grayscale_img)
            loss = criterion(output['pred_ab'], color_img)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                grayscale_img, color_img = batch
                output = model(grayscale_img)
                val_loss = criterion(output['pred_ab'], color_img)
        
        print(f"Validation Loss: {val_loss.item()}")
        
        # Show example images
        show_image(grayscale_img[0], title="Grayscale Image")
        show_image(color_img[0], title="Color Image")
        show_image(output['pred_ab'][0], title="Predicted Color Image")

if __name__ == "__main__":
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    available_datasets = list_available_datasets(base_data_dir)
    
    if not available_datasets:
        print("No datasets available in the data directory.")
    else:
        dataset_name = input("Enter the dataset name to train on: ").strip().lower()
        train(dataset_name)