import torch
import torch.nn as nn
from typing import Tuple


class AvgPool2d_variable_input(nn.Module):
    """2D average pooling with the input size computed at runtime
    """
    
    def __init__(self, kernel_size: int = 1, stride: Tuple[int, int] = (1,1),
                 padding: int = 0) -> None:
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        layer = nn.AvgPool2d(kernel_size = (self.kernel_size, x.shape[-1]), stride = self.stride,
                             padding = self.padding)
        
        return layer(x)