import torch
import torch.nn as nn


class VGG_M_face_bn_dag(nn.Module):
    """ VGG-M face architecture
    
    Code adapted from http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.py
    
    Parameters
    ----------
    include_final_fc_layer: bool
        If false, excludes the last fully-connected layer (fc8)

    References
    ----------
    .. [1]  O.M. Parkhi, A. Vedaldi, A. Zisserman, "Deep face recognition", British Machine Vision Conference,
            2015

    .. [2]  S. Albanie, http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.py
    """

    def __init__(self, include_final_fc_layer: bool = True) -> None:
        super(VGG_M_face_bn_dag, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3],
                     'features_dim': 4096}

        self.include_final_fc_layer = include_final_fc_layer
        
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        
        if self.include_final_fc_layer:
            self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def extract_convolution_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features of last convolution block
        
        Parameters
        ----------
        x: torch.Tensor
            pre-processed image
            
        Returns
        -------
        x18: torch.Tensor
            features
        """        
        
        x1 = self.conv1(x)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        
        return x18
    
    def extract_fc_features_from_convolution_features(self, x18: torch.Tensor) -> torch.Tensor:
        """Passes convolution features through the fully-connected layers (excluding the last)
        
        Parameters
        ----------
        x18: torch.Tensor
            extracted convolution features (using the method "extract_convolution_features")
            
        Returns
        -------
        x24: torch.Tensor
            features
        """
        
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)
        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        
        return x24

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features of penultimate layer
        
        Parameters
        ----------
        x: torch.Tensor
            pre-processed image
            
        Returns
        -------
        x24: torch.Tensor
            features
        """        

        x18 = self.extract_convolution_features(x)
        x24 = self.extract_fc_features_from_convolution_features(x18)

        return x24

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """         
        Parameters
        ----------
        x: torch.Tensor
            pre-processed image
            
        Returns
        -------
        x: torch.Tensor
            logits
        """

        x = self.extract_features(x)

        if self.include_final_fc_layer:
            x = self.fc8(x)
        
        return x