import torch
import torch.nn as nn
import timm
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        # Define channel sizes for each layer
        self.channel_sizes = [
            in_channels,      # Initial input channels
            in_channels // 2, # Half the input channels
            in_channels // 4, # Quarter the input channels
            in_channels // 8, # Eighth the input channels
            in_channels // 16 # Sixteenth the input channels
        ]
        
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(self.channel_sizes[0], self.channel_sizes[1], 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[2], 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(inplace=True),
            
            # Third upsampling block
            nn.ConvTranspose2d(self.channel_sizes[2], self.channel_sizes[3], 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_sizes[3]),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling block
            nn.ConvTranspose2d(self.channel_sizes[3], self.channel_sizes[4], 
                              kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_sizes[4]),
            nn.ReLU(inplace=True),
            
            # Final convolution
            nn.Conv2d(self.channel_sizes[4], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)