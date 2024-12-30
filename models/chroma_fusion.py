import torch
import torch.nn as nn
from .efficient_net import EfficientNetEncoder
from .bottleneck import Bottleneck
from .decoder import Decoder
from .vit import ViT

class ChromaFusion(nn.Module):
    def __init__(self, config):
        super(ChromaFusion, self).__init__()
        self.cnn_encoder = EfficientNetEncoder(
            model_name=config['efficientnet_model_name'],
            pretrained=config['pretrained']
        )

        # Adjust the channel adapter to match the expected input channels for the ViT
        self.channel_adapter = nn.Conv2d(self.cnn_encoder.out_channels, 3, kernel_size=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.vit = ViT(model_name=config['model_name'])

        self.bottleneck = Bottleneck(
            in_channels=config['bottleneck_in'],
            out_channels=config['bottleneck_out']
        )
        self.decoder = Decoder(
            in_channels=config['bottleneck_out'],
            out_channels=config['decoder_out']
        )

    def forward(self, x):
        cnn_features = self.cnn_encoder(x)
        adapted_features = self.channel_adapter(cnn_features)
        upsampled_features = self.upsample(adapted_features)
        vit_features = self.vit(upsampled_features)
        batch_size, seq_len, embed_dim = vit_features.size()

        if seq_len == 197:
            vit_features = vit_features[:, 1:, :]  # Remove the first token
        
        assert vit_features.size(1) == 196, f"Unexpected seq_len: {vit_features.size(1)}"
        vit_features = vit_features.view(batch_size, 14, 14, embed_dim).permute(0, 3, 1, 2)
        bottleneck_features = self.bottleneck(vit_features)

        return self.decoder(bottleneck_features)