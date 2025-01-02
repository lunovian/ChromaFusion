import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Decoder, self).__init__()
        
        # Store skip connection channels
        self.skip_channels = skip_channels[::-1]  # Reverse order to match decoder
        
        # Define main channel sizes
        self.channel_sizes = [
            in_channels,      # Initial input channels
            in_channels // 2, # Half the input channels
            in_channels // 4, # Quarter the input channels
            in_channels // 8, # Eighth the input channels
            in_channels // 16 # Sixteenth the input channels
        ]
        
        # Create decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            # First upsampling block
            self._make_decoder_block(
                self.channel_sizes[0] + self.skip_channels[0], 
                self.channel_sizes[1]
            ),
            # Second upsampling block
            self._make_decoder_block(
                self.channel_sizes[1] + self.skip_channels[1], 
                self.channel_sizes[2]
            ),
            # Third upsampling block
            self._make_decoder_block(
                self.channel_sizes[2] + self.skip_channels[2], 
                self.channel_sizes[3]
            ),
            # Fourth upsampling block
            self._make_decoder_block(
                self.channel_sizes[3] + self.skip_channels[3], 
                self.channel_sizes[4]
            )
        ])
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channel_sizes[4], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels)  # Add CBAM after each decoder block
        )

    def forward(self, x, skip_features):
        # Reverse skip features to match decoder order
        skip_features = skip_features[::-1]
        
        # Debug print shapes (only once)
        if not hasattr(self, '_printed_shapes'):
            print("Decoder input shapes:")
            print(f"Main features: {x.shape}")
            for i, skip in enumerate(skip_features):
                print(f"Skip connection {i}: {skip.shape}")
            self._printed_shapes = True
        
        # Process through decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Resize skip connection to match current feature map size
            target_size = x.shape[2:]  # Get current spatial dimensions
            skip = F.interpolate(skip_features[i], 
                               size=target_size,
                               mode='bilinear', 
                               align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        return self.final_conv(x)