import torch
import torch.nn as nn
from typing import Tuple

from .utils.conv_block_fn import conv_block
from .utils.sum_layer import SumTwoInputs


class DownsamplingBlock(nn.Module):
    
    def __init__(self, dim1: int, dim2: int, dim3: int, dim4: int,
                 kernel1_size: Tuple[int, int] = (1, 1),
                 kernel2_size: Tuple[int, int] = (3, 3),
                 kernel3_size: Tuple[int, int] = (1, 1)) -> None:
        super(DownsamplingBlock, self).__init__()
        
        self.conv_down = conv_block(dim1, dim4, kernel_size = (1, 1),
                                    stride = (2, 2),
                                    padding = (0, 0),
                                    bias = False,
                                    include_relu = False)
        
        self.conv1 = conv_block(dim1, dim2, kernel1_size, stride = (2, 2),
                                padding = (0, 0), bias = False)
        self.conv2 = conv_block(dim2, dim3, kernel2_size, stride = (1, 1),
                                padding = (1, 1), bias = False)
        self.conv3 = conv_block(dim3, dim4, kernel3_size, stride = (1, 1),
                                padding = (0, 0), bias = False, include_relu = False)
        
        self.sum_layer = SumTwoInputs()
        self.relu = nn.ReLU()       
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_downsampled = self.conv_down(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.sum_layer(x_downsampled, x)
        
        return self.relu(x)