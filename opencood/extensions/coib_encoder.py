import torch
from torch import Tensor
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_ELU: bool = True
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if use_ELU is True:
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_ELU: bool = True
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        if use_ELU is True:
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class CoIBEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 256,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        layers: List[int] = [2,2,2],
        zero_init_residual: bool = False,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CoIBEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = input_dim
        self.dilation = 1
        self.pool_size = 32
        self.base_width = 64

        self.groups = groups
        self.layer1 = self._make_layer(block, max(8, input_dim // 2), layers[0], stride=1)
        self.layer2 = self._make_layer(block, max(4, input_dim // 4), layers[1], stride=2)
        self.layer3 = self._make_layer(block, max(2, input_dim // 8), layers[2], stride=2)
        self.maxpool = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        # self.meanpool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.fc = nn.Linear(self.pool_size * self.pool_size * max(2, input_dim // 8), output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_max = self.maxpool(x)
        x_max = torch.flatten(x_max, 1)
        # x_mean = self.meanpool(x)
        # x_mean = torch.flatten(x_mean, 1)
        # x = x_mean + x_max
        x = self.fc(x_max)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
class CoIBDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 64,
        initial_channels: int = 256,
        initial_size: tuple = (25, 88),
        use_mask: bool = True,
        flnal_relu: bool = True,
        activation = nn.ELU
    ) -> None:
        super(CoIBDecoder, self).__init__()
        
        self.use_mask = use_mask
        self.final_relu = flnal_relu
        self.initial_size = initial_size  # H x W
        self.fc = nn.Linear(input_dim, initial_channels * initial_size[0] * initial_size[1])
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(initial_channels, initial_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(initial_channels // 2),
            activation(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(initial_channels // 2, initial_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(initial_channels // 2),
            activation(inplace=True)
        )
        
        self.conv3 = nn.Sequential( 
            nn.ConvTranspose2d(initial_channels // 2, output_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            activation(inplace=True)
            )

        if flnal_relu:
            self.relu = nn.ReLU(inplace=True)

        if self.use_mask:
            self.mask_conv1 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
                activation(inplace=True),
                nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
            )
            self.mask_conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
            self.mask_conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        
        if self.use_mask is False:

            # Linear layer to expand input
            x = self.fc(x)  # [B, C*H*W], torch.Size([16, 563200])

            # Reshape to 4D tensor
            x = x.view(x.size(0), -1, self.initial_size[0], self.initial_size[1])  # [B, C, H, W]

            # First deconv + activation
            x = self.conv1(x) #torch.Size([16, 256, 25, 88]) -> torch.Size([16, 128, 50, 176])

            # Second deconv + activation
            x = self.conv2(x) # torch.Size([16, 128, 50, 176]) -> torch.Size([16, 128, 100, 352])

            # Final deconv
            x = self.conv3(x) # torch.Size([16, 128, 50, 176]) -> torch.Size([16, 64, 200, 704])

            return x
        else:
            # Linear layer to expand input
            x = self.fc(x)  # [B, C*H*W], torch.Size([16, 563200])

            # Reshape to 4D tensor
            x = x.view(x.size(0), -1, self.initial_size[0], self.initial_size[1])  # [B, C, H, W]

            # First deconv
            x = self.conv1(x) * self.mask_conv1(mask) #torch.Size([16, 256, 25, 88]) -> torch.Size([16, 128, 50, 176])

            # Second deconv
            x = self.conv2(x) * self.mask_conv2(mask)# torch.Size([16, 128, 50, 176]) -> torch.Size([16, 128, 100, 352])

            # Final deconv 
            x = self.conv3(x) * self.mask_conv3(mask)# torch.Size([16, 128, 50, 176]) -> torch.Size([16, 64, 200, 704])
            
            if self.final_relu:
                x = self.relu(x)

            return x


if __name__ == "__main__":
    encoder_model = CoIBEncoder()
    output = encoder_model(torch.randn(1, 64, 200, 704))
    print(encoder_model)
    print(output.shape)

    decoder_model = CoIBDecoder()
    output = decoder_model(output, torch.randn(1, 1, 200, 704))
    print(decoder_model)
    print(output.shape)
