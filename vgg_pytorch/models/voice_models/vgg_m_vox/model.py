import torch
import torch.nn as nn

from .layers.flatten import Flatten
from .layers.avgpool_2d_variable_input import AvgPool2d_variable_input
    

class VGGMVox(nn.Module):
    """ VGG-M CNN model for speaker identification/verification
    
    Parameters
    ----------    
    include_final_fc_layer: bool
        If true, includes the last fully-connected layer outputting the logits
        as given by the pre-trained model on VoxCeleb
    
    include_relu_after_penultimate_fc_layer: bool
        If true, includes a ReLU activation after the penultimate fully-connected
        layer
    
    References
    ----------
    .. [1]  A. Nagrani, J. S. Chung, A. Zisserman, "VoxCeleb: a large-scale speaker identification dataset",
            INTERSPEECH, 2017, pp. 2616-2620
            
    .. [2]  A. Nagrani, https://github.com/a-nagrani/VGGVox            
    """    
        
    def __init__(self, include_final_fc_layer = True,
                 include_relu_after_penultimate_fc_layer = True) -> None:
        super(VGGMVox, self).__init__()

        # Meta information
        self.meta = {'features_dim': 1024}
        self.include_final_fc_layer = include_final_fc_layer
        self.include_relu_after_penultimate_fc_layer = include_relu_after_penultimate_fc_layer
        
        # Convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Sequential(nn.Conv2d(1, 96, (7, 7),stride = (2, 2),padding = 1, dilation = 1),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2))))
                                         
        self.conv_layers.append(nn.Sequential(nn.Conv2d(96, 256, (5, 5), stride = (2, 2), padding = 1, dilation = 1),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size = (3, 3), stride = (2, 2))))
                           
        self.conv_layers.append(nn.Sequential(nn.Conv2d(256, 384, (3, 3), stride = (1, 1), padding = 1, dilation = 1),
                                              nn.ReLU()))
                           
        self.conv_layers.append(nn.Sequential(nn.Conv2d(384, 256, (3, 3), stride = (1, 1), padding = 1, dilation = 1),
                                              nn.ReLU()))
                           
        self.conv_layers.append(nn.Sequential(nn.Conv2d(256, 256, (3, 3), stride = (1, 1), padding = 1, dilation = 1),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size = (5, 3), stride = (3, 2))))
                           
        self.conv_layers.append(nn.Sequential(nn.Conv2d(256, 4096, (9, 1), stride = (1, 1), padding = 0),
                                              nn.ReLU()))
                
            
        # Average pooling
        self.avg_pool_layer = AvgPool2d_variable_input()
                
        # Flattening
        self.flattening_layer = Flatten()        
        
        # Fully-connected layers        
        if self.include_relu_after_penultimate_fc_layer:
            self.fc_layer = nn.Sequential(nn.Linear(4096, 1024),
                                          nn.ReLU())
        else:
            self.fc_layer = nn.Sequential(nn.Linear(4096, 1024))
        
        if self.include_final_fc_layer:
            self.fc_layer2 = nn.Linear(1024, 1300) # 1024, 1251 on paper
                              
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """ Extract features from penultimate fully-connected layer        
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
                              
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.avg_pool_layer(x)
        x = self.flattening_layer(x)        
        x = self.fc_layer(x)
        
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
        
        x = self.extract_features(x)
        
        if self.include_final_fc_layer:
            x = self.fc_layer2(x)
        
        return x