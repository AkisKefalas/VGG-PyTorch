=======================================
VGG-PyTorch
=======================================

PyTorch implementation of selected VGG models. Useful for feature extraction from face images and speech waveforms.

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Model
     - Modality
     - Checkpoint
     - Checkpoint notes
     - Other comments
   * - VGG-M [1]_
     - Face
     - `Download <http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.pth>`_
     - Face recognition model trained on VGG-Face [1]_.
     - The model's code is adapted from `the official implementation <http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.py>`_. I have added some functions to facilitate pre-processing and feature extraction.
   * - VGG-M [2]_
     - Voice
     - `Download <https://drive.google.com/file/d/1HDMALbtu93bfeS2Wz3Wnh80r5PGMpxL-/view?usp=sharing>`_
     - Speaker recognition model trained on VoxCeleb1 [2]_
     - The original code is in `Matlab <https://github.com/a-nagrani/VGGVox/tree/master>`_ and I re-implemented it here in PyTorch.
   * - VGGVoxResNet [3]_
     - Voice
     - `Download <https://drive.google.com/file/d/1KRwnf0p6WgAmczZ3IPFnf-Fg-OPbo49p/view?usp=sharing>`_
     - Speaker verification model trained on VoxCeleb2 [3]_
     - The original code is in `Matlab <https://github.com/a-nagrani/VGGVox/tree/master>`_ and I re-implemented it here in PyTorch.
   * - SVHF [4]_
     - Face + Voice
     - `Download <https://drive.google.com/file/d/1Zzrq1s3ooxeVfphI0mBDESW0-VvU_5Zm/view?usp=sharing>`_
     - Static binary voice-to-face matching model trained on VoxCeleb2 [3]_ [4]_
     - The original code, for static binary voice-to-face matching, is in `Matlab <https://github.com/a-nagrani>`_. I re-implemented it here in PyTorch and added functionality for face-to-voice matching, as well as dynamic images.


Installation
=======================================

Prerequisites:

* NumPy https://numpy.org/
* PyTorch https://pytorch.org/
* OpenCV https://pypi.org/project/opencv-python/
* SciPy https://scipy.org/
* Librosa https://librosa.org/doc/latest/index.html

To install from source, run the following:

.. code-block::

  pip install -e . 

Example usage
=======================================
**Face feature extraction**:

.. code-block:: python

    import torch
    
    from vgg_pytorch.preprocessing import preprocess_image
    from vgg_pytorch.models import VGG_M_face_bn_dag

    device = "cuda:0"
    path_to_checkpoint = "<insert path to pre-trained model checkpoint>"
    path_to_image = "<insert path to image>"

    # Load model
    model = VGG_M_face_bn_dag()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(path_to_checkpoint, map_location = device))

    # Pre-process input image
    x = preprocess_image(path_to_image)

    # Extract features
    with torch.no_grad():
        z = model.extract_features(x.unsqueeze(0).to(device))    

**Voice feature extraction**:

.. code-block:: python

    import torch
    
    from vgg_pytorch.preprocessing import preprocess_audio
    from vgg_pytorch.models import VGGMVox, VGGVoxResNet

    device = "cuda:0"
    path_to_checkpoint = "<insert path to pre-trained model checkpoint>"
    path_to_audio = "<insert path to an audio file>"

    # Load model
    model = VGGVoxResNet()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(path_to_checkpoint, map_location = device))

    # Pre-process input audio
    x = preprocess_audio(path_to_audio)

    # Extract features
    with torch.no_grad():
        z = model.extract_features(x.unsqueeze(0).to(device))

**Cross-modal face and voice feature extraction**:

.. code-block:: python

    import torch

    from vgg_pytorch.preprocessing import preprocess_image, preprocess_audio
    from vgg_pytorch.models import SVHF

    device = "cuda:0"
    path_to_checkpoint = "<insert path to pre-trained model checkpoint>"
    path_to_face = "<insert path to image>"
    path_to_audio = "<insert path to an audio file>"

    # Load model
    model = SVHF()
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(path_to_ckpt, map_location = device))    

    # Pre-process inputs
    x_f = preprocess_image(path_to_face)    
    x_a = preprocess_audio(path_to_audio)

    # Extract features
    with torch.no_grad():
        z_f = model.face_net(x_f.unsqueeze(0).to(device))
        z_a = model.voice_net(x_a.unsqueeze(0).to(device))


Citing
=======
If you use this code, please cite the original publications of the authors:

.. code-block:: bibtex

  @InProceedings{Parkhi15,
  author       = "Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman",
  title        = "Deep Face Recognition",
  booktitle    = "British Machine Vision Conference",
  year         = "2015",
  }

  @InProceedings{Nagrani17,
  author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
  title        = "VoxCeleb: a large-scale speaker identification dataset",
  booktitle    = "INTERSPEECH",
  year         = "2017",
  }

  @InProceedings{Nagrani17,
  author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
  title        = "VoxCeleb2: Deep Speaker Recognition",
  booktitle    = "INTERSPEECH",
  year         = "2018",
  }

  @InProceedings{Nagrani18a,
  author       = "Nagrani, A. and Albanie, S. and Zisserman, A.",
  title        = "Seeing Voices and Hearing Faces: Cross-modal biometric matching",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2018",
  }

Please also refer to the following official implementations:

* VGG-M face model: `http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_m_face_bn_dag.py`
* VGGVox models in Matlab: `https://github.com/a-nagrani/VGGVox/tree/master`
* SVHF model in Matlab: `https://github.com/a-nagrani/SVHF-Net`
* Converting VGG Matlab models to PyTorch: `https://github.com/albanie/pytorch-mcn`


References
==========

.. [1]  Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, "Deep face recognition", British Machine Vision Conference, 2015
.. [2]  Arsha Nagrani, Joon S. Chung, Andrew Zisserman, "VoxCeleb: a large-scale speaker identification dataset", Interspeech, 2017
.. [3]  Joon S. Chung, Arsha Nagrani, Andrew Zisserman, "VoxCeleb2: Deep Speaker Recognition", Interspeech, 2018
.. [4]  Arsha Nagrani, Samuel Albanie, Andrew Zisserman, "Seeing Voices and Hearing Faces: Cross-modal biometric matching", IEEE CVPR, 2018
