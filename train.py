import argparse
import pytorch_lightning as pl
from training.train import ColorizationModel
from torchvision.datasets import Places365
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def main(train_dir, val_dir, epochs, learning_rate, display_step, batch_size, image_size, num_train_images, num_val_images):
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
        'learning_rate': learning_rate,
        'display_step': display_step,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_train_images': num_train_images,
        'num_val_images': num_val_images,
    }

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = Places365(root=train_dir, split='train-standard', download=True, transform=transform)
    val_dataset = Places365(root=val_dir, split='val', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ColorizationModel(config)

    # Define early stopping and checkpoint callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min', save_top_k=1)

    trainer = pl.Trainer(max_epochs=epochs, callbacks=[early_stopping, model_checkpoint])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the colorization model on a specified dataset.')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--display_step', type=int, default=1, help='Frequency of displaying images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Size of input images')

    args = parser.parse_args()

    num_train_images = int(input("Enter the number of training images to use (or press Enter to use all): ").strip() or 0)
    num_val_images = int(input("Enter the number of validating images to use (or press Enter to use all): ").strip() or 0)

    main(args.train_dir, args.val_dir, args.epochs, args.learning_rate, args.display_step, args.batch_size, args.image_size, 
         num_train_images if num_train_images > 0 else None,
         num_val_images if num_val_images > 0 else None)