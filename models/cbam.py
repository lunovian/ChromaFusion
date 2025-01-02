import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        )
        
    def forward(self, x):
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch using torch.amax for deterministic behavior
        max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))
        
        out = torch.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # Channel pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        
        # Concatenate and process
        x = torch.cat([avg_out, max_out], dim=1)
        x = torch.sigmoid(self.conv(x))
        return x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.use_channel = channels >= reduction_ratio * 2  # Only use channel attention if enough channels
        if self.use_channel:
            self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        try:
            # Apply channel attention only if we have enough channels
            if (self.use_channel):
                x = x * self.channel_attention(x)
            
            # Apply spatial attention
            x = x * self.spatial_attention(x)
            return x
            
        except Exception as e:
            print(f"Error in CBAM forward pass: {str(e)}")
            return x  # Return input unchanged if error occurs
