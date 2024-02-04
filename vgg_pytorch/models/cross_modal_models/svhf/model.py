import torch
import torch.nn as nn
from typing import Tuple

from vgg_pytorch.models import VGGMVox

from .layers.augmented_vggm_face import AugmentedVGGMFace
from .layers.fusion_layer import FusionLayer


class SVHF(nn.Module):
    """ SVHF model for binary cross-modal matching
    
    i.e., given a query sample from one modality and two gallery samples from the other
    modality, which of the two gallery samples does the query correspond to?
    
    Parameters
    ----------
    experiment_type: str
        "V_F" for voice-to-face matching (voice query, two gallery face images)
        "F_V" for face-to-voice matching (face query, two gallery voice samples)

    include_dynamic_images: bool
        If true, the model also processes dynamic images

    include_final_softmax: bool
        If true, includes a softmax layer as the last layer
        
    References
    ----------
    .. [1]  A. Nagrani, S. Albanie, A. Zisserman, 
            "Seeing Voices and Hearing Faces: Cross-modal biometric matching", IEEE CVPR, 2018
    """    
    
    def __init__(self, experiment_type = "V_F", include_dynamic_images: bool = False,
                 include_final_softmax: bool = False) -> None:
        super(SVHF, self).__init__()
        
        assert experiment_type in ["V_F", "F_V"]
        
        self.experiment_type = experiment_type
        self.include_dynamic_images = include_dynamic_images
        
        # Face (static image) network
        self.face_net = AugmentedVGGMFace()
        
        # Dynamic image network        
        if self.include_dynamic_images:
            self.dimg_net = AugmentedVGGMFace(model_type = "dynamic")
            
        # Voice network            
        self.voice_net = VGGMVox(include_final_fc_layer = False,
                                 include_relu_after_penultimate_fc_layer = False)
        
        # Fusion network        
        self.fusion_net = FusionLayer(include_final_softmax)
        
    def extract_features(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """ Extracts features for the two gallery samples and query sample
        
        Parameters
        ----------
        x: Tuple[torch.Tensor]
            Inputs following the sequence (gallery1_sample, gallery2_sample, query_sample)
            If we include dynamic images, then each visual modality input contains
            two samples (one for the static face image, and one for the dynamic images)
            
        Returns
        -------
        Tuple[torch.Tensor]
            Features corresponding to gallery1_sample, gallery2_sample, query_sample
        """

        if self.experiment_type == "V_F":
            if self.include_dynamic_images:
                img1, dimg1, img2, dimg2, audio = x
                
                visual1_feat = 0.5 * (self.face_net(img1) + self.dimg_net(dimg1))
                visual2_feat = 0.5 * (self.face_net(img2) + self.dimg_net(dimg2))
                audio_feat = self.voice_net(audio)
                
                return visual1_feat, visual2_feat, audio_feat
                
            else:
                img1, img2, audio = x
                
                img1_feat = self.face_net(img1)
                img2_feat = self.face_net(img2)
                audio_feat = self.voice_net(audio)
                
                return img1_feat, img2_feat, audio_feat                
                
        else:
            if self.include_dynamic_images:
                audio1, audio2, img, dimg = x
                
                audio1_feat = self.voice_net(audio1)
                audio2_feat = self.voice_net(audio2)
                visual_feat = 0.5 * (self.face_net(img) + self.dimg_net(dimg))
                
                return audio1_feat, audio2_feat, visual_feat

            else:
                audio1, audio2, img = x
                
                audio1_feat = self.voice_net(audio1)
                audio2_feat = self.voice_net(audio2)
                img_feat = self.face_net(img)
                
                return audio1_feat, audio2_feat, img_feat
        
    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        """ Compute output of SVHF
        
        Parameters
        ----------
        x: Tuple[torch.Tensor]
            Inputs following the sequence (gallery1_sample, gallery2_sample, query_sample)
            If we include dynamic images, then each visual modality input contains
            two samples (one for the static face image, and one for the dynamic images)
            
        Returns
        -------
        z: torch.Tensor
            Softmax probabilities if self.include_final_softmax, otherwise logits
        """        
            
        gallery1_feat, gallery2_feat, query_feat = self.extract_features(x)        
        z = torch.cat([gallery1_feat, gallery2_feat, query_feat], dim=1)        
        z = self.fusion_net(z)
        
        return z