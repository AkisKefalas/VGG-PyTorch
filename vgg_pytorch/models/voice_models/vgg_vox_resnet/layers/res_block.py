import torch
import torch.nn as nn

from .utils.conv_block_fn import conv_block
from .utils.sum_layer import SumTwoInputs


class ResBlock(nn.Module):
        
    def __init__(self, dim1: int, dim2: int, dim3: int, dim4: int) -> None:
        super(ResBlock, self).__init__()
    
        self.conv1 = conv_block(dim1, dim2, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), bias = False)
        self.conv2 = conv_block(dim2, dim3, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        self.conv3 = conv_block(dim3, dim4, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0),
                                bias = False, include_relu = False)
        
        self.sum_layer = SumTwoInputs()
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        
        x = self.sum_layer(x, x1)
        
        return self.relu(x)