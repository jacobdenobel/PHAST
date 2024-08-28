# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:44:51 2019

@author: beimx004
"""
import numpy as np
from .defaults import DEFAULT_BINS


def fft_filterbank(
    buf,
    nFft=256,
    combineDcNy=False,
    includeNyquistBin=False,
    compensateFftLength=False,
    **kwargs,
):

    X = np.fft.fft(buf, nFft, axis=0)

    if combineDcNy:
        NF = X[nFft // 2 + 1, :]
        DC = X[0, :]
        X[0, :] = (np.real(DC) + np.real(NF)) + 1j * (np.real(DC) - np.real(NF))
        X[nFft // 2 + 1, :] = 0

    if includeNyquistBin:
        X = X[0 : nFft // 2 + 1, :]
    else:
        X = X[0 : nFft // 2, :]

    if compensateFftLength:
        X = X / (nFft / 2)
    return X


def channel_energy(
    X,
    gain_agc,
    startBin=6,
    nBinLims=DEFAULT_BINS,
    nHop=20,
    gainDomain="linear",
    **kwargs,
):

    startBin = startBin - 1  # subtract 1 for python indexing

    nFrames = X.shape[1]
    nChan = nBinLims.size
    assert (
        isinstance(gain_agc, np.ndarray) or gain_agc.size == 0
    ), "gAgc, if supplied, must be a vector!"

    # determine if AGC is sample-based and deciimate to frame rate if necessary
    lenAgcIn = gain_agc.shape[1]
    if lenAgcIn > nFrames:
        gain_agc = gain_agc[:, nHop - 1 : -1 : nHop]
        assert (
            np.abs(gain_agc.shape[1] - nFrames) <= 3
        ), "Length of sample-based gAgc input incompatable with nr. frames in STFT matrix: length/nHop must = approx nFrames."
        if gain_agc.size < nFrames:
            gain_agc = np.concatenate(
                (
                    gain_agc,
                    gain_agc[:, -1:]
                    * np.ones((gain_agc.shape[0], nFrames - gain_agc.shape[1])),
                ),
                axis=1,
            )
            gain_agc = gain_agc[:, 0:nFrames]
        elif lenAgcIn > 0 and lenAgcIn < nFrames:
            raise ValueError(
                "Length of gAgc input incompatible with number of frames in STFT matrix: length must be >= nr. frames."
            )

    # compute roo-sum-squared FFT magnitudes per channel
    engy = np.zeros((nChan, nFrames))
    currentBin = startBin
    for iChan in np.arange(nChan):
        currBinIdx = np.arange(currentBin, currentBin + nBinLims[iChan])
        try:
            engy[iChan, :] = np.sum(np.abs(X[currBinIdx, :]) ** 2, axis=0)
        except Exception as err:
            print(err)
            breakpoint()
        currentBin += nBinLims[iChan]

    engy = np.sqrt(engy)

    # compensate AGC gain, if applicable
    if lenAgcIn > 0:
        if gainDomain.lower() == "linear" or gainDomain.lower() == "lin":
            pass
        elif gainDomain.lower() == "log" or gainDomain.lower() == "log2":
            gain_agc = 2 ** (gain_agc / 2)
        elif gainDomain.lower() == "db":
            gain_agc = 10 ** (gain_agc / 20)
        else:
            raise ValueError("Illegal value for parameter " "gainDomain" "")
        gain_agc = np.maximum(gain_agc, np.finfo(float).eps)
        engy = np.divide(engy, gain_agc)

    return engy


def hilbert_envelope(
    X,
    nChan=15,
    startBin=6,
    nBinLims=DEFAULT_BINS,
    upperBound=np.inf,
    lowerBound=0,
    outputOffset=0,
    **kwargs,
):

    startBin = startBin - 1  # correcting for matlab base-1 indexing

    Y = np.zeros(X.shape, dtype=complex)

    Y[np.arange(0, X.shape[0] - 1, 2), :] = -X[np.arange(0, X.shape[0] - 1, 2), :]

    Y[np.arange(0, X.shape[0] - 1, 2) + 1, :] = X[
        np.arange(0, X.shape[0] - 1, 2) + 1, :
    ]

    L = Y.shape[1]
    env = np.zeros((nChan, L))
    envNoLog = np.zeros((nChan, L))
    currentBin = startBin

    numFullFrqBin = np.floor(nBinLims / 4)
    numPartFrqBin = np.mod(nBinLims, 4)
    logFiltCorrect = np.array([2233, 952, 62, 0]) / (2**10)

    logCorrect = logFiltCorrect + outputOffset + 16

    for i in np.arange(nChan):
        for j in np.arange(numFullFrqBin[i]):
            sr = np.sum(np.real(Y[currentBin : currentBin + 4, :]), axis=0)
            si = np.sum(np.imag(Y[currentBin : currentBin + 4, :]), axis=0)
            env[i, :] = env[i, :] + sr**2 + si**2
            currentBin += 4
        sr = np.sum(np.real(Y[currentBin : currentBin + numPartFrqBin[i], :]), axis=0)
        si = np.sum(np.imag(Y[currentBin : currentBin + numPartFrqBin[i], :]), axis=0)

        env[i, :] = env[i, :] + sr**2 + si**2

        envNoLog[i, :] = env[i, :]
        env[i, :] = np.log2(env[i, :])

        if nBinLims[i] > logCorrect.size - 1:
            env[i, :] = env[i, :] + logCorrect[-1:]
        else:
            env[i, :] = (
                env[i, :] + logCorrect[nBinLims[i] - 1]
            )  # correcting here for matlab base-1 indexing
        currentBin += numPartFrqBin[i]

    ix = ~np.isfinite(env)
    env[ix] = 0

    return np.maximum(np.minimum(env, upperBound), lowerBound)
