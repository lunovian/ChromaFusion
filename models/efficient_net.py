import torch
import torch.nn as nn
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        # Get all intermediate features and ignore/remove unused layers
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1,2,3,4),
            in_chans=3,
            num_classes=0,
            global_pool='',
            drop_rate=0.,
            drop_path_rate=0.
        )
        
        # Clear unused attributes
        unused_attributes = ['bn2', 'classifier', 'conv_head']
        for attr in unused_attributes:
            if hasattr(self.model, attr):
                delattr(self.model, attr)
        
        self.out_channels = self.model.feature_info.channels()[-1]
        self.skip_channels = self.model.feature_info.channels()
        print(f"EfficientNet encoder initialized with skip connection channels: {self.skip_channels}")

    def forward(self, x):
        try:
            if x.shape[1] != 3:
                raise ValueError(f"EfficientNet expects 3 input channels, got {x.shape[1]}")
            
            features = self.model(x)
            return features  # Returns list of features at different scales
        except Exception as e:
            print(f"Error in EfficientNet forward pass: {str(e)}")
            raise