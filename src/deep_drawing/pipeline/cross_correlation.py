"""
Cross-Correlation / Cross Power Spectrum (GCC-PHAT) feature processing
================================
"""

import sys
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import torch

from deep_drawing import utils

# ------------------------- Settings -------------------------------------

NUM_MICS = 4
AUDIO_BIT_DEPTH = 24  # bits
AUDIO_SAMPLE_RATE = 48000  # samples per sec
VIDEO_SAMPLE_RATE = 30  # locations per sec

#class DataPoint:
#    """
#    Data point includes AudioInput and location (x, y).
#    """
#    audio: AudioInput
#    location: tuple[float, float]

#class AudioInput:
#    """
#    Audio input (1600-sample 24-bit, [-8,388,608, 8,388,607]).
#    """
#    samples: np.ndarray[1600]

# remainder = len(data) % 1600
#    if remainder > 0:
#        padding_needed = 1600 - remainder
#        data = np.pad(data, (0, padding_needed), mode='constant')

#    partitions = data.reshape(-1, 1600)
#    inputs = [AudioInput(samples=chunk) for chunk in partitions]


def inputs_from_audio(fname: str) -> np.ndarray:
    """
    Return (N, 1600) audio input array from an audio fname (.wav).
    """
    data, samplerate = sf.read(utils.get_dataset_file(fname), dtype="float32")
    if samplerate != AUDIO_SAMPLE_RATE:
        print(f"Error: Sample rate is {samplerate}, not {AUDIO_SAMPLE_RATE}")
        sys.exit(1)

    # mono check
    if len(data.shape) > 1:
        # data.shape is (samples, channels) for multichannel audio
        # data[:, 0] grabs the first microphone channel
        data = data[:, 0]

    # Assumption: audio and video feeds both begin at the exact same time
    # lop off remainder data
    remainder = len(data) % 1600
    if remainder > 0:
        inputs = data[:-remainder].reshape(-1, 1600)
    else:
        inputs = data.reshape(-1, 1600)

    return inputs


def cross_correlation_time_domain(window_size):
    """
    Cross correlation over time domain.
    
    """


def compute_gcc_phat(signal1, signal2, fs, interp=1):
    """Compute GCC-PHAT TDOA between two signals using pyroomacoustics."""
    # Get TDOA using pyroomacoustics (returns value in seconds)
    tdoa_seconds = pra.tdoa(signal1, signal2, interp=interp, phat=True, fs=fs)
    # Convert to samples
    tdoa_samples = tdoa_seconds * fs
    return tdoa_samples
