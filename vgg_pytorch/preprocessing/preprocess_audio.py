import numpy as np
import scipy.io
import torch
from typing import Union

from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal import lfilter
from librosa.core import resample


def preprocess_audio(audio: Union[str, np.ndarray], target_sr: int = 16000, duration: float = 3.0,
                    ) -> torch.Tensor:
    """ Pre-processes raw waveform into a spectrogram for use by VGG models

    Parameters
    ----------
    audio: str or np.ndarray
        path to audio or audio
    
    target_sr: int
        target sampling rate
    
    duration: float
        target duration in seconds
        
    Returns
    -------
    torch.Tensor
        pre-processed spectrogram

    References
    ----------
    .. [1]  A. Nagrani, J. S. Chung, A. Zisserman, "VoxCeleb: a large-scale speaker identification dataset",
            INTERSPEECH, 2017, pp. 2616-2620
            
    .. [2]  J. S. Chung, A. Nagrani, A. Zisserman, "VoxCeleb2: Deep Speaker Recognition",
            INTERSPEECH, 2018, pp. 1086-1090
            
    .. [3]  A. Nagrani, https://github.com/a-nagrani/VGGVox
    """
    
    # Load audio and convert to mono
    if isinstance(audio, str):
        sr, audio = wavfile.read(audio)
    audio = convert_to_mono(audio)

    # Resample if needed
    final_sr = sr
    if target_sr:
        if target_sr!=sr:
            audio = resample(np.asfortranarray(audio), sr, target_sr)
            final_sr = target_sr
    
    # Extract and preprocess a random audio segment
    audio = extract_random_audio_segment(audio, final_sr, duration)
    audio = remove_dc_component(audio)
    audio = get_dithered_signal(audio)
    audio = pre_emphasis_filtering(audio)
    
    # Extract spectrogram
    spec = get_normalized_spectrogram(audio, final_sr)

    return torch.from_numpy(spec.astype(np.float32)).unsqueeze(0)

def convert_to_mono(audio: np.ndarray) -> np.ndarray:

    audio = audio.astype(np.float32)
    
    # Mono array already
    if len(audio.shape) == 1:
        return audio

    if audio.shape[1] == 1:
        return audio[:, 0]
        
    # Compute mono array from stereo
    audio_mono = np.sum(audio, axis = 1)/2
        
    # If the two channels cancel out return the first one
    criterion = np.abs(np.sum(audio_mono)/audio.shape[0])
        
    if criterion<0.001:
        return audio[:, 0]
    
    return audio_mono

def extract_random_audio_segment(audio: np.ndarray, sr: int, duration: float) -> np.ndarray:
    
    if not duration:
        return audio
    
    num_samples_required = int(sr * duration)
    diff = num_samples_required - audio.shape[0]
    
    if diff>0:        
        start_index = int(diff/2)
        end_index = start_index + audio.shape[0]
        audio_new = np.zeros([num_samples_required])        
        audio_new[start_index:end_index] = audio
        
        return audio_new
    
    else:        
        start_index = np.random.choice(-diff+1)
        end_index = start_index + num_samples_required
        audio_new = audio[start_index:end_index]  
        
        return audio_new

def get_normalized_spectrogram(audio: np.ndarray, sr: int, window_duration: float = 0.025,
                               step_size: float = 0.01) -> np.ndarray:

    # Get segment length and overlap number in terms of samples
    segment_length = int(window_duration * sr)
    n_overlap = segment_length - int(step_size * sr)
    
    # Short-time Fourier transform
    s_frequencies, s_times, s_stft = stft(audio,
                                          fs = 1.0,
                                          window = 'hamming',
                                          nperseg = segment_length,
                                          noverlap = n_overlap,
                                          nfft = 512,
                                          return_onesided = False,
                                          padded = False,
                                          boundary = None)

    # Spectrogram
    spec = np.abs(s_stft)
    
    # Remove mean and standard deviation across frequency buckets
    mean_spec = np.mean(spec, axis = 1)
    std_spec = np.std(spec, axis=1)

    num_windows = spec.shape[1]
    broadcast_mean_spec = np.array([mean_spec] * num_windows).T
    broadcast_std_spec = np.array([std_spec] * num_windows).T
    
    normalized_spec = (spec - broadcast_mean_spec)/broadcast_std_spec
    
    return normalized_spec

def remove_dc_component(audio: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    return lfilter([1, -1], [1, -alpha], audio, axis = 0).astype(np.float32)

def get_dithered_signal(audio: np.ndarray) -> np.ndarray:

    rand_array1 = np.random.uniform(size = audio.shape[0])
    rand_array2 = np.random.uniform(size = audio.shape[0])
    dither = rand_array1 + rand_array2 - 1

    spow = np.std(audio)
    adjustment = 1e-6 * spow * dither

    return audio + adjustment

def pre_emphasis_filtering(audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    return lfilter([1, -alpha], 1, audio)
