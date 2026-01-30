import torch
import torch.nn as nn

class Masknet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5, 7), use_residual=True):
        super().__init__()
        kernel_size1, kernel_size2, kernel_size3 = kernel_size
        self.use_residual = use_residual

        # Define three convolution layers with different kernel sizes to capture features at multiple scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size1, padding=kernel_size1 // 2),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size2, padding=kernel_size2 // 2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size3, padding=kernel_size3 // 2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )

        # Project concatenated channels back to the original number of channels
        self.proj_channels = nn.Conv2d((in_channels//2 + in_channels//4 + in_channels//4), in_channels, kernel_size=1)

        # Final projection layer with stride 2 for potential down-sampling
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)

        if self.use_residual:
            # Residual connection setup for stabilizing training
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            )

    def forward(self, x):
        identity = x

        # Apply each convolution branch independently on the input
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Concatenate outputs from all branches
        x = torch.cat((x1, x2, x3), dim=1)
        
        # Project concatenated features back to the original channel count
        x = self.proj_channels(x)
        
        # Final feature processing and possibly down-sampling
        x = self.proj(x)

        if self.use_residual:
            # Adjust the input dimensions to match those of the processed output
            identity = self.downsample(identity)
            x += identity

        return x

   
class ScaleLayer(nn.Module):
    def __init__(self, scale=100):
        super(ScaleLayer, self).__init__()
        self.scale = scale  
    
    def forward(self, x):
        return x * self.scale
    
class AdaptiveScaleLayer(nn.Module):
    def __init__(self, init_scale=1.0, dim=256):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_scale] * dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return x * self.scale + self.shift



def init_linear_weight(layer, bias_value=0.1):
    """
    Initialize the weights of a linear layer with a normal distribution.
    
    Parameters:
        layer (nn.Linear): The linear layer to initialize.
        bias_value (float): The value to initialize the bias to, default is 0.1.
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_value)