# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:16:36 2019

@author: beimx004
"""
import numpy as np


def f120(
    carrier,
    envelope,
    weights,
    idxAudioFrame,
    nChan=15,
    M=500 * np.ones(16),
    T=50 * np.ones(16),
    IDR=60 * np.ones(16),
    gain=0 * np.ones(16),
    mapClip=None,
    chanToEl=np.arange(16),
    carrierMode=1,
    **kwargs,
):
    """
    ampWords = f120MappingFunc(par, carrier, env, weights, idxAudioFrame)

    Map envelope amplitudes to elec stimulation current according to
      f(x)  = (M-T)/IDR * (x - SAT + 12dB + IDR + G)) + T
            = (M-T)/IDR * (x - SAT + 12dB + G) + M
    with
          x - envelope value  [dB]  (per electode and frame)
          M - electric M-Level [uA] (per electrode)
          T - electric T-Level [uA] (per electrode)
        IDR - input dynamic range [dB] (per electrode)
          G - gain [dB] (per electrode)
        SAT - the envelope saturation level [dB]
    and apply fine-structure carrier signal. See Nogueira et al. (2009) for details.

    INPUT:
      carrier - nChan x nFtFrame matrix of carrier signals (range 0..1), sampled at FT rate
      env - nChan x nAudFrame matrix of channel envelopes (log2 power)
      weights - 2*nCh x nAudFrame matrix of current steering weights (in [0,1])
      idxAudioFrame - index of corresponding audio frame corresponding to
                      each FT (forward telemetry) frame / stimulation cycle

    FIELDS FOR PAR:
      parent.nChan - number of envelope channels
      mapM - M levels, 1 x nEl [uA]
      mapT - T levels, 1 x nEl [uA]
      mapIdr - IDRs, 1 x nEl [dB]
      mapGain - electrode gains, 1 x nEl [dB]
      mapClip - clipping levels, 1 x nl [uA]
      chanToElecPair - 1 x nChan vector defining mapping of logical channels
                       to electrode pairs (1 = E1/E2, ...)
      carrierMode - how to apply carrier [0/1/2] [default: 1]
                      0 - don't apply carrier (i.e. set carrier == 1)
                      1 - apply to channel envelopes (mapper input)  [default]
                      2 - apply to mapped stimulation amplitudes (mapper output)

    OUTPUT:
      ampWords - 30 x nFrames vector of current amplitudes with 2 successive
                 rows for each of the 15 physical electrode pairs; muAmp

    Copyright (c) 2015-2020 Advanced Bionics. All rights reserved.
    """
    nFtFrames = len(idxAudioFrame)
    if mapClip is None:
        mapClip = M * 4

    mSat = 30 * 10 * np.log(2) / np.log(10)
    mapA = (M - T) / IDR
    mapK = M + (M - T) / IDR * (-mSat + 12 + gain)

    envelope = envelope * 10 * np.log(2) / np.log(10)

    ampWords = np.zeros((30, nFtFrames))

    for iChan in np.arange(nChan):
        iElLo = chanToEl[iChan]
        iElHi = iElLo + 1
        iAmpLo = iElLo * 2  # remove+1 for 0 base indexing
        iAmpHi = iAmpLo + 1
        if carrierMode == 0:
            mappedLo = mapA[iElLo] * envelope[iChan, idxAudioFrame] + mapK[iElLo]
            mappedHi = mapA[iElHi] * envelope[iChan, idxAudioFrame] + mapK[iElHi]
        elif carrierMode == 1:
            mappedLo = (
                mapA[iElLo] * envelope[iChan, idxAudioFrame] * carrier[iChan, :]
                + mapK[iElLo]
            )
            mappedHi = (
                mapA[iElHi] * envelope[iChan, idxAudioFrame] * carrier[iChan, :]
                + mapK[iElHi]
            )
        elif carrierMode == 2:
            mappedLo = (
                mapA[iElLo] * envelope[iChan, idxAudioFrame] + mapK[iElLo]
            ) * carrier[iChan, :]
            mappedHi = (
                mapA[iElHi] * envelope[iChan, idxAudioFrame] + mapK[iElHi]
            ) * carrier[iChan, :]

        mappedLo = np.maximum(np.minimum(mappedLo, mapClip[iElLo]), 0)
        mappedHi = np.maximum(np.minimum(mappedHi, mapClip[iElHi]), 0)

        ampWords[iAmpLo, :] = mappedLo * weights[iChan, idxAudioFrame]
        ampWords[iAmpHi, :] = mappedHi * weights[iChan + nChan, idxAudioFrame]

    return ampWords
