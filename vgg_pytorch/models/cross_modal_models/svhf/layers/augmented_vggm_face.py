import torch
import torch.nn as nn

from vgg_pytorch.models import VGG_M_face_bn_dag

    
class AugmentedVGGMFace(nn.Module):
    """ Extracts face features for fusion layer of SVHF
        
    References
    ----------
    .. [1]  A. Nagrani, S. Albanie, A. Zisserman, 
            "Seeing Voices and Hearing Faces: Cross-modal biometric matching", IEEE CVPR, 2018
            
    .. [2]  O.M. Parkhi, A. Vedaldi, A. Zisserman, "Deep face recognition", British Machine Vision Conference,
            2015
    """    
    
    def __init__(self, model_type:str = "static") -> None:
        super(AugmentedVGGMFace, self).__init__()
        
        assert model_type in ["static", "dynamic"]
        
        self.vgg_m = VGG_M_face_bn_dag(include_final_fc_layer = False)
        self.fc_layer = nn.Linear(4096, 1024)
        
        self.model_type = model_type        
        self.meta = self.vgg_m.meta
        self.meta['features_dim'] = 1024        
    
    def get_vggm_dynamic_img_features_with_temporal_pooling(self, x: torch.Tensor,
                                                           ) -> torch.Tensor:
        """ Extracts VGG-M features from dynamic images using temporal pooling        
        Follows method described in [1]

        Parameters
        ----------
        x: torch.Tensor
            Tensor of dynamic images
            Must have shape [batch size, timesteps, channels, height, width]
            
        Returns
        -------
        x: torch.Tensor
        """

        b, t = x.shape[:2]

        # Extract convolution features
        x = torch.reshape(x, [-1] + list(x.shape[2:]))
        x = self.vgg_m.extract_convolution_features(x)
        x = torch.reshape(x, [b, t] + list(x.shape[1:]))

        # Temporal pooling
        x, _ = torch.max(x, 1, keepdim = False)

        # FC layers
        x = self.vgg_m.extract_fc_features_from_convolution_features(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            If model is static, x is a Tensor of face images of shape:
            [batch size, channels, height, width]
            If model is dynamic, x is a Tensor of dynamic images of shape:
            [batch size, timesteps, channels, height, width]
            
        Returns
        -------
        x: torch.Tensor
        """        
        
        if self.model_type == "static":
            x = self.vgg_m(x)
        else:
            x = self.get_vggm_dynamic_img_features_with_temporal_pooling(x)
            
        x = self.fc_layer(x)
        
        return x