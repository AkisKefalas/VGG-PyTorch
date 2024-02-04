import torch
import torch.nn as nn
from typing import Tuple


def conv_block(in_channels: int, out_channels: int, kernel_size: int, stride: Tuple[int, int] = (1, 1),
               padding: int = 1, dilation: Tuple[int, int] = (1, 1), bias: bool = True,
               include_relu: bool = True) -> nn.Module:

    conv_layer = nn.Conv2d(in_channels = in_channels,
                           out_channels = out_channels,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           dilation = dilation,
                           bias = bias)

    bn_layer = nn.BatchNorm2d(num_features = out_channels, eps=1e-05, momentum=0.1, affine=True,
                              track_running_stats=True)

    if include_relu:
        return nn.Sequential(conv_layer, bn_layer, nn.ReLU())
    
    else:
        return nn.Sequential(conv_layer, bn_layer)