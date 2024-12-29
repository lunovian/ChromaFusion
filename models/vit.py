import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Identity()  # Remove the classification head

    def forward(self, x):
        # Ensure that the input to the ViT model is 4D [batch_size, channels, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add a channel dimension
        return self.model.forward_features(x)