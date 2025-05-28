import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    '''Residual Block for EDSR model.'''
    def __init__(self, num_filters, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        residual = self.block(x)
        return x + residual * self.res_scale  # Apply residual scaling    
    

class UpsampleBlock(nn.Module):
    '''Upsampling Block for EDSR model.'''
    def __init__(self, num_filters, scale_factor):
        super().__init__()
        layers = []
        if (scale_factor & (scale_factor - 1)) == 0: 
            for _ in range(int(math.log2(scale_factor))):
                layers += [
                    nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1, bias=True),
                    nn.PixelShuffle(2),
                ]
        elif scale_factor == 3:
            layers += [
                nn.Conv2d(num_filters, num_filters * 9, 3, 1, 1, bias=True),
                nn.PixelShuffle(3),
            ]
        else:
            raise NotImplementedError(f"Unsupported scale {scale_factor}")
        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample(x)


class EDSR(nn.Module):
    '''Enhanced Deep Residual Networks for Single Image Super-Resolution.'''
    def __init__(self, scale_factor=2, num_filters=64, num_res_blocks=16, res_scale=0.1):
        super(EDSR, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, res_scale) for _ in range(num_res_blocks)]
        )
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.upsample = UpsampleBlock(num_filters, scale_factor)
        self.conv3 = nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = self.res_blocks(x)
        x = self.conv2(residual) + x
        x = self.upsample(x)
        x = self.conv3(x)
        return x