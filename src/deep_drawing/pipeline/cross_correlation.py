"""
Cross-Correlation / Cross Power Spectrum (GCC-PHAT) feature processing
================================
"""

import sys
from itertools import combinations
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim

from deep_drawing import utils

# ------------------------- Settings -------------------------------------

NUM_MICS = 4
AUDIO_BIT_DEPTH = 24  # bits
AUDIO_SAMPLE_RATE = 48000  # samples per sec
VIDEO_SAMPLE_RATE = 30  # locations per sec

PLYWOOD_SIDE_LENGTH = 0.6  # m


# Guesstimates
MIN_SPEED_OF_SOUND = 1000  # strict lower bound for m/s of sound in plywood
MAX_TDOA = PLYWOOD_SIDE_LENGTH * np.sqrt(2) / MIN_SPEED_OF_SOUND  # s, about 0.84 ms
MAX_TDOA_IN_AUDIO_SAMPLES = MAX_TDOA * AUDIO_SAMPLE_RATE  # samples, about 40.72

# sound MUST travel at least 343 m/s in a plywood solid,
# putting absolute largest-case MAX_TDOA_IN_AUDIO_SAMPLES at 118.7,
# or window size at 237.4 audio samples minimum. 81.44 recommended


class LiveAudioBuffer:
    """
    Audio buffer that can queue segments of audio and perform windowing operations.
    """
    def __init__(self, channels: int=2, window_size: int=1600, hop_size: int=400):
        self.channels = channels
        self.window_size = window_size
        self.hop_size = hop_size

        self.buffer = np.zeros((channels, 0), dtype=np.float32)
        self.timestamp = 0

    def push_audio(self, new_samples: np.ndarray):
        """Call this every time we push a new (channels, N) array of audio data."""
        self.buffer = np.hstack((self.buffer, apply_low_cut(new_samples)))

    def push_audio_from_files(self, fnames):
        """Push a new (channels, N) array of audio data from files."""
        if len(fnames) != self.channels:
            print("Error: # of audio files needs to match # of channels in buffer")
            return

        new_samples = np.vstack([input_from_audio(f) for f in fnames])
        self.buffer = np.hstack((self.buffer, apply_low_cut(new_samples)))

    def pop_window(self):
        """Pop (channels, window_size) audio input array from the buffer."""
        window = self.buffer[:, :self.window_size]
        self.buffer = self.buffer[:, self.hop_size:]

        return window

    def compute_pair_tdoa_tracking(self, i: int=0, j: int=1):
        """
        Compute time-domain cross-correlation for pair of mics (i,j) by processing all windows.
        Return time array (in s) and delay array (in samples/s).
        """
        timestamps = []
        delays_in_samples = []

        center_idx = self.window_size - 1

        while self.buffer.shape[1] >= self.window_size:
            # cross-correlation
            window = self.pop_window()
            cc_curve = np.correlate(window[i], window[j], mode='full')

            # find time lag yielding peak correlation
            max_idx = np.argmax(cc_curve)
            delay_samples = max_idx - center_idx

            timestamps.append(self.timestamp)
            delays_in_samples.append(delay_samples)

            # print(f"Mic {i+1} x Mic {j+1} time lag: {delay_samples}")
            self.timestamp += self.hop_size / AUDIO_SAMPLE_RATE  # hop seconds

        return timestamps, delays_in_samples

    def compute_pair_tdoa_tracking_time_domain(self, i: int=0, j: int=1):
        """
        Compute time-domain cross-correlation for pair of mics (i,j) by processing all windows.
        Return time array (in s) and delay array (in samples/s).
        """
        timestamps = []
        delays_in_samples = []

        center_idx = self.window_size - 1

        while self.buffer.shape[1] >= self.window_size:
            # cross-correlation
            window = self.pop_window()
            cc_curve = np.correlate(window[i], window[j], mode='full')

            # find time lag yielding peak correlation
            max_idx = np.argmax(cc_curve)
            delay_samples = max_idx - center_idx

            timestamps.append(self.timestamp)
            delays_in_samples.append(delay_samples)

            # print(f"Mic {i+1} x Mic {j+1} time lag: {delay_samples}")
            self.timestamp += self.hop_size / AUDIO_SAMPLE_RATE  # hop seconds

        return timestamps, delays_in_samples

    def compute_pair_tdoa_tracking_freq_domain(self, i: int=0, j: int=1):
        """
        GCC-PHAT for pair of mics (i,j) by processing all windows.
        Return time array (in s) and delay array (in samples/s).
        """
        timestamps = []
        delays_in_samples = []

        while self.buffer.shape[1] >= self.window_size:
            # cross-correlation
            window = self.pop_window()
            tdoa_samples = pra.tdoa(window[i], window[j], interp=1, phat=True)

            timestamps.append(self.timestamp)
            delays_in_samples.append(tdoa_samples)

            self.timestamp += self.hop_size / AUDIO_SAMPLE_RATE  # hop seconds

        return timestamps, delays_in_samples

    #for i, j in combinations(range(self.channels), 2):


def input_from_audio(fname: str) -> np.ndarray:
    """
    Return audio input array from an audio fname (.wav).
    """
    data, samplerate = sf.read(utils.get_dataset_file(fname), dtype="float32")
    if samplerate != AUDIO_SAMPLE_RATE:
        print(f"Error: Sample rate is {samplerate}, not {AUDIO_SAMPLE_RATE}")
        sys.exit(1)

    # mono check
    if len(data.shape) > 1:
        # data.shape is (samples, channels) for multichannel audio
        # data[:, 0] grabs the first microphone channel
        # need to ask about this later because the .wav files are stereo
        data = data[:, 0]

    return data


def apply_low_cut(amplitudes, cutoff_hz=250, sample_rate=AUDIO_SAMPLE_RATE, order=5):
    """
    Apply a low-cut (high-pass) Butterworth filter to a list of amplitudes.
    """
    sos = signal.butter(order, cutoff_hz, btype='highpass', fs=sample_rate, output='sos')
    filtered_amplitudes = signal.sosfilt(sos, amplitudes)

    return filtered_amplitudes



# Ignore

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
        # need to ask about this later because the .wav files are stereo
        data = data[:, 0]

    # Assumption: audio and video feeds both begin at the exact same time
    # lop off remainder data
    remainder = len(data) % 1600
    if remainder > 0:
        inputs = data[:-remainder].reshape(-1, 1600)
    else:
        inputs = data.reshape(-1, 1600)

    return inputs


def compute_gcc_phat(signal1, signal2, fs, interp=1):
    """Compute GCC-PHAT TDOA between two signals using pyroomacoustics."""
    # Get TDOA using pyroomacoustics (returns value in seconds)
    tdoa_seconds = pra.tdoa(signal1, signal2, interp=interp, phat=True, fs=fs)
    # Convert to samples
    tdoa_samples = tdoa_seconds * fs
    return tdoa_samples

class MultiLayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
#mlp_model = MultiLayerPerceptron(input_dim=8, hidden_dim=16, output_dim=1)

#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)

#model = nn.Sequential(
#    nn.Linear(in_features=8, out_features=16),
#    nn.ReLU(),
#    nn.Linear(in_features=16, out_features=1)
#)
