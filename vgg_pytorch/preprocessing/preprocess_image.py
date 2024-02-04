import numpy as np
import imageio
import cv2
import torch
from typing import List, Union

MEAN_ARRAY_OF_VGG_FACE = [131.45376586914062, 103.98748016357422, 91.46234893798828]


def preprocess_image(img: Union[str, np.ndarray], mean_array: List[float] = MEAN_ARRAY_OF_VGG_FACE,
                    ) -> torch.Tensor:
    """ Pre-processes image for use by VGG-M model

    Parameters
    ----------
    img: str or np.ndarray
        path to image or image
    
    mean_array: int
        array of mean pixel value for each channel
        
    Returns
    -------
    torch.Tensor
        pre-processed image

    References
    ----------
    .. [1]  O.M. Parkhi, A. Vedaldi, A. Zisserman, "Deep face recognition", British Machine Vision Conference,
            2015       
    """
    
    # Load image
    if isinstance(img, str):
        img = imageio.imread(img)
    
    # Resize and extract random patch
    img = resize_image(img, 256)
    img = extract_random_patch(img, 224, 224)
    
    # Subtract mean from channels
    broadcasted_mean_array = get_broadcasted_mean_array(mean_array, 224, 224)    
    img = img - broadcasted_mean_array

    return torch.from_numpy(img.astype(np.float32).transpose([2, 0, 1]))

def resize_image(img: np.ndarray, largest_dim: int = 256) -> np.ndarray:
    """ Resizes image respecting the height/width proportions
    """
    
    h, w, _ = img.shape
    
    if h>w:
        new_h = largest_dim
        new_w = int(w/h * new_h)

    else:
        new_w = largest_dim
        new_h = int(h/w * new_w)
    
    img = cv2.resize(img, (new_w, new_h))
    
    return img

def extract_random_patch(img: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """ Extracts random patch of specifided height and width from the image
    """
    
    h, w, _ = img.shape

    # Get indices of random patch to extract
    h_start = np.random.choice(np.arange(0, max(h - target_h, 0) + 1))
    w_start = np.random.choice(np.arange(0, max(w - target_w, 0) + 1))  

    h_end = min(h_start + target_h, h)
    w_end = min(w_start + target_w, w)

    # Extract patch
    extracted_patch = img[h_start:h_end, w_start:w_end, :]

    if extracted_patch.shape[:2] == (target_h, target_w):
        return extracted_patch

    else:
        # Zero pad if extracted patch is smaller than specified height and width    
        h, w, _ = extracted_patch.shape
        h_start = (target_h - h)//2
        w_start = (target_w - w)//2

        random_patch = np.zeros([target_h, target_w, img.shape[2]])
        random_patch[h_start:h_start + h, w_start:w_start + w, :] = extracted_patch

        return random_patch

def get_broadcasted_mean_array(mean_array: List[float], h: int, w: int) -> np.ndarray:
    """ Broadcasts mean array across height and width of image
    """
    
    c = len(mean_array)

    broadcast_mean_array = np.array([mean_array] * h * w).astype(np.float32)
    broadcast_mean_array = broadcast_mean_array.reshape([h, w, c])
    
    return broadcast_mean_array