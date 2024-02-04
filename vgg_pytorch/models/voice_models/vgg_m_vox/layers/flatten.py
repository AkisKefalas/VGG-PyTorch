import torch
import torch.nn as nn


class Flatten(nn.Module):
    """ Flattening operation as a layer    
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)