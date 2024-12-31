import torch
import torch.nn as nn
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        # Filter out unused parameters during model creation
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
            in_chans=3,
            num_classes=0,  # Disable classifier
            global_pool=''  # Disable global pooling
        )
        
        # Remove unused layers
        if hasattr(self.model, 'classifier'):
            delattr(self.model, 'classifier')
        if hasattr(self.model, 'conv_head'):
            delattr(self.model, 'conv_head')
        
        self.out_channels = self.model.feature_info.channels()[-1]
        print(f"EfficientNet encoder initialized with {self.out_channels} output channels")

    def forward(self, x):
        try:
            if x.shape[1] != 3:
                raise ValueError(f"EfficientNet expects 3 input channels, got {x.shape[1]}")
            
            features = self.model(x)
            return features[-1]
        except Exception as e:
            print(f"Error in EfficientNet forward pass: {str(e)}")
            raise