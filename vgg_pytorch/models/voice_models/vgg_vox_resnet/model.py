import torch
import torch.nn as nn

from .layers.first_block import FirstBlock
from .layers.res_block import ResBlock
from .layers.downsampling_block import DownsamplingBlock
from .layers.avgpool_2d_variable_input import AvgPool2dVariableInput
from .layers.flatten import Flatten


class VGGVoxResNet(nn.Module):
    """ ResNet architecture for speaker verification
    
    References
    ----------
    .. [1]  J. S. Chung, A. Nagrani, A. Zisserman, "VoxCeleb2: Deep Speaker Recognition",
            INTERSPEECH, 2018, pp. 1086-1090
            
    .. [2]  A. Nagrani, https://github.com/a-nagrani/VGGVox            
    """
    
    def __init__(self) -> None:
        super(VGGVoxResNet, self).__init__()

        self.list_of_layers = nn.ModuleList()
        self.list_of_layers.append(FirstBlock())
        
        self.list_of_layers.append(ResBlock(256, 64, 64, 256))
        self.list_of_layers.append(ResBlock(256, 64, 64, 256))
        
        self.list_of_layers.append(DownsamplingBlock(256, 128, 128, 512))
        
        self.list_of_layers.append(ResBlock(512, 128, 128, 512))
        self.list_of_layers.append(ResBlock(512, 128, 128, 512))
        self.list_of_layers.append(ResBlock(512, 128, 128, 512))
        
        self.list_of_layers.append(DownsamplingBlock(512, 256, 256, 1024))
        
        self.list_of_layers.append(ResBlock(1024, 256, 256, 1024))
        self.list_of_layers.append(ResBlock(1024, 256, 256, 1024))
        self.list_of_layers.append(ResBlock(1024, 256, 256, 1024))
        self.list_of_layers.append(ResBlock(1024, 256, 256, 1024))
        self.list_of_layers.append(ResBlock(1024, 256, 256, 1024))
        
        self.list_of_layers.append(DownsamplingBlock(1024, 512, 512, 2048))
        
        self.list_of_layers.append(ResBlock(2048, 512, 512, 2048))
        self.list_of_layers.append(ResBlock(2048, 512, 512, 2048))
        
        self.list_of_layers.append(nn.AvgPool2d(kernel_size = (8, 3), stride = (1, 1), padding = (0, 0)))
        
        self.list_of_layers.append(nn.Conv2d(2048, 2048, kernel_size = (9, 1), stride = (1, 1), padding = (0, 0)))
        self.list_of_layers.append(nn.ReLU())
        
        self.list_of_layers.append(AvgPool2dVariableInput(dim_to_pool = 3))
        
        self.list_of_layers.append(Flatten())    
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """         
        Parameters
        ----------
        x: torch.Tensor
            pre-processed audio (spectrogram)
            must have a shape of the form [batch_size, 1, freq_bins, num_audio_frames]
            
        Returns
        -------
        x: torch.Tensor
            output feature
        """
        
        for layer in self.list_of_layers:
            x = layer(x)
            
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """         
        Parameters
        ----------
        x: torch.Tensor
            pre-processed audio (spectrogram)
            must have a shape of the form [batch_size, 1, freq_bins, num_audio_frames]
            
        Returns
        -------
        x: torch.Tensor
            output feature
        """
        
        return self.extract_features(x)