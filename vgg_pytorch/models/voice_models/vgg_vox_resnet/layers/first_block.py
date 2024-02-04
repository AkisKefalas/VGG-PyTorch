import torch
import torch.nn as nn

from .utils.zero_pad_bottom_and_right import ZeroPadBottomAndRight
from .utils.conv_block_fn import conv_block
from .utils.sum_layer import SumTwoInputs

    
class FirstBlock(nn.Module):
    
    def __init__(self) -> None:
        super(FirstBlock, self).__init__()
        
        self.conv0 = conv_block(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = True)
        self.zero_pad = ZeroPadBottomAndRight()
        self.maxpool0 = nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2), padding = (0, 0))
        self.conv_pool0 = conv_block(64, 256, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0),
                                     bias = False, include_relu = False)
        
        self.conv1 = conv_block(64, 64, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), bias = False)
        self.conv2 = conv_block(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        self.conv3 = conv_block(64, 256, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0),
                                bias = False, include_relu = False)
        
        self.sum_layer = SumTwoInputs()
        self.relu = nn.ReLU()     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv0(x)
        x = self.zero_pad(x)
        x = self.maxpool0(x)
        
        x1 = self.conv_pool0(x)
        
        x2 = self.conv1(x)
        x2 = self.conv2(x2)        
        x2 = self.conv3(x2)
        
        x = self.sum_layer(x1, x2)
        
        return self.relu(x)