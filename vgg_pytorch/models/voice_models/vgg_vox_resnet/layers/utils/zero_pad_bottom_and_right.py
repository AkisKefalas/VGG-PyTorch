import torch
import torch.nn as nn


class ZeroPadBottomAndRight(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        [batch_size, num_channels, height, width] = list(x.shape)

        new_tensor = torch.zeros([batch_size, num_channels, height + 1, width + 1],
                                 device = x.device)
        new_tensor[:, :, 0:height, 0:width] = x
        
        return new_tensor