import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """ Fusion layer of SVHF
    
    Parameters
    ----------
    include_final_softmax: bool
        If true, includes a softmax layer as the last layer
        
    References
    ----------
    .. [1]  A. Nagrani, S. Albanie, A. Zisserman, 
            "Seeing Voices and Hearing Faces: Cross-modal biometric matching", IEEE CVPR, 2018        
    """
    
    def __init__(self, include_final_softmax: bool = False) -> None:
        super(FusionLayer, self).__init__()
        
        self.include_final_softmax = include_final_softmax
        
        self.fc0 = nn.Sequential(nn.Linear(3072, 1024), nn.ReLU())        
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())        
        self.fc2 = nn.Sequential(nn.Linear(512, 2))
        
        if self.include_final_softmax:
            self.softmax = nn.Softmax(1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            concatenation query and two gallery embeddings
           
        Returns
        -------
        x: torch.Tensor
            Logits if not include_final_softmax, otherwise softmax probabilities
        """
        
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        if self.include_final_softmax:
            x = self.softmax(x)
        
        return x