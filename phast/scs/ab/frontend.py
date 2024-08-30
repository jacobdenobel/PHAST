# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:24:42 2020

@author: Jbeim
"""

import numpy as np
from scipy.signal import lfilter
from scipy.io.wavfile import read as wavread
import librosa


def normalize_to_db(y, stim_db):
    return y / np.sqrt(np.mean(np.power(y, 2))) * 20e-6 * pow(10, stim_db / 20)


def process_stim(y, audio_fs, stim_db=65.0, **kwargs):
    if audio_fs is None:
        raise ValueError("audio_fs should be provided")

    if audio_fs != 17400:
        y = librosa.resample(y, orig_sr=audio_fs, target_sr=17400)

    if stim_db is not None:
        y = normalize_to_db(y, stim_db)

    return y.reshape(1, -1)


def read_wav(wav_file, fs=17400, i_channel=1, t_start_end=None, stim_db=65.0, **kwargs):
    y, sr = librosa.load(wav_file, sr=fs)
    y = process_stim(y, sr, stim_db)
    return y, sr


# this pre-emphasis filter has more values to optimize. Figure out difference
def td_filter(
    x,
    coeff_numerator=np.array([0.7688, -1.5376, 0.7688]),
    coeff_denominator=np.array([1, -1.5299, 0.5453]),
    **kwargs,
):
    if x.shape[0] > x.shape[1]:  #
        x = x.T

    coeff_dimension = len(coeff_numerator.shape)

    if coeff_dimension == 1:
        n_channels = 1
    elif coeff_dimension == 2:
        n_channels = coeff_numerator.shape[0]
    else:
        raise ValueError(
            "Filter coefficients must be organized in a vector or 2d matrix!"
        )

    if n_channels > 1:
        print("multichannel audio input")
        Y = np.zeros((n_channels, x.size))
        for i in np.arange(n_channels):
            Y[i, :] = lfilter(coeff_numerator[i, :], coeff_denominator[i, :], x[i, :])
    else:
        Y = lfilter(coeff_numerator, coeff_denominator, x)

    return Y
