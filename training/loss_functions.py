import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=None):
        super(PerceptualLoss, self).__init__()
        # Load the pre-trained VGG16 model
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features

        # Extract features from specified layers
        self.feature_layers = feature_layers if feature_layers else [3, 8, 15, 22]
        self.vgg = nn.Sequential(*list(vgg.children())[:max(self.feature_layers) + 1]).eval()

        # Freeze the VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Normalize the input images (assuming they are in the range [0, 1])
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        # Extract features
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        # Compute the MSE loss between the features
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += F.mse_loss(xf, yf)

        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class TotalVariationLoss(nn.Module):
    def forward(self, x):
        batch_size, c, h, w = x.size()
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / (batch_size * c * h * w)

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_loss_weight=0.4, mse_loss_weight=0.5, tv_loss_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TotalVariationLoss()
        self.perceptual_loss_weight = perceptual_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.tv_loss_weight = tv_loss_weight

    def forward(self, pred, target):
        perceptual_loss_value = self.perceptual_loss(pred, target)
        mse_loss_value = self.mse_loss(pred, target)
        tv_loss_value = self.tv_loss(pred)
        combined_loss_value = (self.perceptual_loss_weight * perceptual_loss_value +
                               self.mse_loss_weight * mse_loss_value +
                               self.tv_loss_weight * tv_loss_value)
        return combined_loss_value

# Example usage
if __name__ == "__main__":
    # Create random tensors to simulate a batch of images
    pred = torch.randn((4, 3, 224, 224), requires_grad=True)  # Predicted images
    target = torch.randn((4, 3, 224, 224))  # Target images

    # Initialize the combined loss
    combined_loss = CombinedLoss(perceptual_loss_weight=0.4, mse_loss_weight=0.5, tv_loss_weight=0.1)

    # Example optimizer
    optimizer = torch.optim.Adam([pred], lr=0.001)

    # Training step
    optimizer.zero_grad()
    loss = combined_loss(pred, target)
    loss.backward()
    optimizer.step()

    print("Combined Loss:", loss.item())