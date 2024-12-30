import torch.nn as nn
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=3, padding=1)  # Convert 1-channel input to 3 channels
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.out_channels = self.model.feature_info[-1]['num_chs']  # Get the number of output channels from the last feature layer

    def forward(self, x):
        x = self.input_adapter(x)  # Convert grayscale input to 3 channels
        features = self.model(x)
        return features[-1]  # Return the features from the last layer