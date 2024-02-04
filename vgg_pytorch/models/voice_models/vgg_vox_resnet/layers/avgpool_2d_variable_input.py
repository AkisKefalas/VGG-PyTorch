import torch
import torch.nn as nn
from typing import Tuple


class AvgPool2dVariableInput(nn.Module):    
    """2D average pooling with the input size computed at runtime
    """
    
    def __init__(self, dim_to_pool: int, kernel_size: int = 1, stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0)) -> None:
        super().__init__()

        self.dim_to_pool = dim_to_pool
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.dim_to_pool == 2:            
            x = x.permute(0, 1, 3, 2)            
            layer = nn.AvgPool2d(kernel_size = (self.kernel_size, x.shape[-1]),
                                 stride = self.stride,
                                 padding = self.padding)
            x = layer(x)            
            x = x.permute(0, 1, 3, 2)
        
            return x
            
        else:            
            layer = nn.AvgPool2d(kernel_size = (1, x.shape[-1]),
                                 stride = self.stride,
                                 padding = self.padding)
            
            return layer(x)