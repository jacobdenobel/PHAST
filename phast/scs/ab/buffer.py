# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:27:28 2020

@author: Jbeim
"""

import numpy as np


def buffer_(X, n, p=0, opt=None):
    """
    Buffer signal vector into matrix of data frames

    In WinBufUnit (‘WB’), the sampled input signal is buffered into frames (according the FFT length and hop-size defined in the parent FftStrategy object)
    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap

    Returns
    -------
    result : (n,m) ndarray
        Buffer array created from X
    """

    if 0 < p & p < n:  # overlapping buffer, pad with p zeros at the beginning
        if opt is None:
            opt = np.zeros((p))
            Xb = np.concatenate((opt, X))
        elif len(opt) == 0:
            opt = np.zeros((p))
            Xb = np.concatenate((opt, X))
        elif opt == "nodelay":
            Xb = X
        else:
            Xb = np.concatenate((opt, X))
    elif p < 0:  # underlapping buffer (skips samples), skip opt samples if provided
        if opt is None:
            Xb = X
        elif len(opt) == 0:
            Xb = X
        elif opt == "nodelay":
            raise ValueError('"bufOpt" must not be set to no delay if p < 0')
        else:
            Xb = X[opt:]
    else:
        raise ValueError("p cannot be greater than n!")

    N = n
    M = np.ceil(len(X) / (n - p)).astype(int)
    b = np.zeros((N, M))

    for i in np.arange(M, dtype=int):
        if i == 0:
            b[:, i] = Xb[:n]
        else:
            x1 = i * (n - p)
            x2 = x1 + n
            if x2 > Xb.size:
                b[:, i] = np.concatenate((Xb, np.zeros((x2 - Xb.size))))[x1:x2]
            else:
                b[:, i] = Xb[x1:x2]
    return b


def window_buffer(signal_in, nFft=256, nHop=20, bufOpt=[], window=None, **kwargs):
    if window is None:
        window = 0.5 * (np.blackman(nFft) + np.hanning(nFft))
        window = window.reshape(1, window.size)

    M, N = signal_in.shape

    if N > M:
        signal_in = signal_in.T
        N = M

    return buffer_(signal_in[:, 0], nFft, nFft - nHop, bufOpt) * window.T
