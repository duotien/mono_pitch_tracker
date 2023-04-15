"""
Copyright (c) 2012-2014 Department of Computational Perception,
Johannes Kepler University, Linz, Austria and Austrian Research Institute for
Artificial Intelligence (OFAI), Vienna, Austria.
All rights reserved.

https://github.com/CPJKU/madmom/blob/main/madmom/audio/signal.py#L169

"""

from typing import Tuple

import torchaudio
import torch
import librosa
import numpy as np


def load_audio_mono(file_path: str, sample_rate: int, keep_channel_dim: bool = True) -> Tuple[np.ndarray, int]:
    """Loads an audio file and returns a mono audio signal.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): The sample rate to load the audio file with.
        keep_channel_dim (bool, optional): Whether to keep the channel dimension of the audio signal. Defaults to True.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio signal and sample rate.
    """

    try:
        signal, sr = load_audio_with_torch(
            file_path, sample_rate, keep_channel_dim=keep_channel_dim)
        signal = signal.numpy()
    except Exception as e:
        print(f'Failed to load audio with `torchaudio`, fallback to `librosa` {e}')
        signal, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        if keep_channel_dim:
            signal = np.expand_dims(signal, -1)

    return signal, sr


def load_audio_with_torch(file_path, sample_rate, keep_channel_dim=True):
    """Load the audio file with target sample rate, down mix to mono.

    Args:
        file_path (str): the audio file path
        sample_rate (int): target sample rate
        keep_channel_dim (bool): if True, keep the channel dimension.

    Return:
        torch.Tensor: Audio array, [n_channels, n_samples] or [n_samples]
        int: sample rate 
    """
    signal, sr = torchaudio.load(file_path, channels_first=False)
    # resampling
    if sr != sample_rate:
        signal = resample(signal, sr, sample_rate)
        sr = sample_rate
    # downmix
    if signal.shape[1] == 1:
        signal = signal.squeeze(-1)
    else:
        signal = remix_torch(signal, 1)

    if keep_channel_dim:
        signal = torch.unsqueeze(signal, 0)
    return signal, sr


def resample(signal, old_sr, new_sr):
    resampler = torchaudio.transforms.Resample(old_sr, new_sr)
    signal = resampler(signal).mean(dim=0)
    return signal


def remix_torch(signal, num_channels):
    """
    This modified for torch, original source: https://github.com/CPJKU/madmom/blob/main/madmom/audio/signal.py#L169

    Remix the signal to have the desired number of channels.

    Parameters
    ----------
    signal : numpy array
        Signal to be remixed.
    num_channels : int
        Number of channels.

    Returns
    -------
    numpy array
        Remixed signal (same dtype as `signal`).

    Notes
    -----
    This function does not support arbitrary channel number conversions.
    Only down-mixing to and up-mixing from mono signals is supported.

    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    If the signal should be down-mixed to mono and has an integer dtype, it
    will be converted to float internally and then back to the original dtype
    to prevent clipping of the signal. To avoid this double conversion,
    convert the dtype first.

    """
    # convert to the desired number of channels
    if num_channels == signal.ndim or num_channels is None:
        # return as many channels as there are.
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        # down-mix to mono
        # Note: to prevent clipping, the signal is converted to float first
        #       and then converted back to the original dtype
        # TODO: add weighted mixing
        return torch.mean(signal, axis=-1).type(signal.dtype)
    elif num_channels > 1 and signal.ndim == 1:
        # up-mix a mono signal simply by copying channels
        return torch.tile(signal[:, None], num_channels)
    else:
        # any other channel conversion is not supported
        raise NotImplementedError((
            f"Requested {num_channels} channels, but got {signal.shape[1]} channels "
            "and channel conversion is not implemented."))

def midi_to_hz(midi_value:float):
    return 440.0 * (2.0 ** ((torch.as_tensor(midi_value) - 69.0) / 12.0))
