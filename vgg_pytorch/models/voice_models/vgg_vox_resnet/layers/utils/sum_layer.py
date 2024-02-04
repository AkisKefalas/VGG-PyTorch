import torch
import torch.nn as nn


class SumTwoInputs(nn.Module):
    def __init__(self) -> None:
        super(SumTwoInputs, self).__init__()
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        return x + y