import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.amp as amp  # Add this import
from .efficient_net import EfficientNetEncoder
from .bottleneck import Bottleneck
from .decoder import Decoder
from .vit import ViT

class ChromaFusion(nn.Module):
    def __init__(self, config):
        super(ChromaFusion, self).__init__()
        
        # Create components
        self.input_norm = nn.BatchNorm2d(1)
        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)  # Additional normalization
        )
        
        self.efficient_net_features = EfficientNetEncoder(
            model_name=config['efficientnet_model_name'],
            pretrained=config['pretrained']
        )

        # Adjust the channel adapter to match the expected input channels for the ViT
        self.channel_adapter = nn.Conv2d(self.efficient_net_features.out_channels, 3, kernel_size=1)
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

        # Add LayerNorm for stability
        self.layer_norm = nn.LayerNorm(self.efficient_net_features.out_channels)

        # Initialize weights properly
        self._init_weights()
        
        # Move everything to device at once
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    @amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):
        # Move input to same device as model
        x = x.to(self.device)
        
        # Input checks and normalization
        if x.isnan().any():
            raise ValueError("Input contains NaN values")
            
        # Normalize input
        x = self.input_norm(x)
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be torch.Tensor, got {type(x)}")
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
        
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 input channel, got {x.shape[1]}")

        # Debug print (only once)
        if not hasattr(self, '_printed_shapes'):
            print(f"Input shape: {x.shape}")
            x_adapted = self.input_adapter(x)
            print(f"After input adapter: {x_adapted.shape}")
            print(f"Device: {x.device}")
            self._printed_shapes = True
            
        # Process through network with gradient checks
        try:
            # Ensure input requires gradient
            if not x.requires_grad:
                x = x.requires_grad_()
            
            # Input adapter with residual connection
            x_input = x
            x = self.input_adapter(x)
            x = x + F.interpolate(x_input, size=x.shape[2:], mode='bilinear')
            
            if torch.isnan(x).any():
                raise ValueError("NaN detected after input adapter")
                
            efficient_net_features = self.efficient_net_features(x)
            efficient_net_features = self.layer_norm(efficient_net_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
            if torch.isnan(efficient_net_features).any():
                raise ValueError("NaN detected after EfficientNet")
                
            adapted_features = self.channel_adapter(efficient_net_features)
            upsampled_features = self.upsample(adapted_features)
            
            # Update ViT processing with gradient checkpointing
            with amp.autocast(device_type=self.device.type):
                # Ensure upsampled features require gradient
                if not upsampled_features.requires_grad:
                    upsampled_features = upsampled_features.requires_grad_()
                    
                def vit_forward(*inputs):
                    return self.vit(inputs[0])
                
                vit_features = torch.utils.checkpoint.checkpoint(
                    vit_forward,
                    upsampled_features,
                    use_reentrant=False,
                    preserve_rng_state=False
                )
            
            # Shape transformations
            batch_size, seq_len, embed_dim = vit_features.size()
            if seq_len == 197:
                vit_features = vit_features[:, 1:, :]
            
            if vit_features.size(1) != 196:
                raise ValueError(f"Expected sequence length 196, got {vit_features.size(1)}")
            
            # Final processing
            vit_features = vit_features.view(batch_size, 14, 14, embed_dim)
            vit_features = vit_features.permute(0, 3, 1, 2)
            bottleneck_features = self.bottleneck(vit_features)
            output = torch.clamp(self.decoder(bottleneck_features), min=-1.0, max=1.0)

            # Add final normalization
            output = torch.tanh(output)  # Ensure output is in [-1, 1]

            return output
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {x.shape}, Device: {x.device}")
            raise