# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:22:02 2019

@author: beimx004
"""
import numpy as np
from .defaults import DEFAULT_BINS, DEFAULT_BIN_TO_LOC_MAP


def spec_peak_locator(
    stftIn,
    nFft=256,
    fs=17400,
    nChan=15,
    startBin=6,
    nBinLims=None,
    binToLoc=None,
    **kwargs,
):
    """'takes the Short-Time Fourier Transform coefficients as input and estimates the dominant peak frequency for each channel and the
    corresponding electrode location along the cochlear axis. For more details on the frequency estimation,
    the reader is referred to Nogueira et al, 2009.

    Input:
    stftIn: every third FFT input frame in complex numbers [nFft/2 x floor(total_frames/3)]

    Output:
    freq : nChan x nFrames matrix of estimated peak frequencies [Hz]
    loc : nChan x nFrames matrix of corresponding cochlear locations [within [0,15]]"""

    if binToLoc is None:
        binToLoc = DEFAULT_BIN_TO_LOC_MAP

    if nBinLims is None:
        nBinLims = DEFAULT_BINS

    fftBinWidth = fs / nFft

    nBins, nFrames = stftIn.shape

    maxBin = np.zeros((nChan, nFrames), dtype=int)
    freqInterp = np.zeros((nChan, nFrames))
    loc = np.zeros((nChan, nFrames))
    binCorrection = np.zeros((1, nFrames))

    PSD = np.real(stftIn * np.conj(stftIn)) / 2
    PSD = np.maximum(PSD, 10 ** (-120 / 20))

    currentBin = startBin - 1  # account for matlab indexing

    for i in np.arange(nChan):
        currBinIdx = np.arange(currentBin, currentBin + nBinLims[i])
        argMaxPsd = np.argmax(PSD[currBinIdx, :], axis=0)
        maxBin[i, :] = currentBin + argMaxPsd
        currentBin += nBinLims[i]

    for i in np.arange(nChan):
        midVal = np.log2(PSD[maxBin[i, :], np.arange(nFrames)])
        leftVal = np.log2(PSD[maxBin[i, :] - 1, np.arange(nFrames)])
        rightVal = np.log2(PSD[maxBin[i, :] + 1, np.arange(nFrames)])

        maxLeftRight = np.maximum(leftVal, rightVal)
        midIsMax = midVal > maxLeftRight

        binCorrection[:, midIsMax] = (
            0.5
            * (rightVal[midIsMax] - leftVal[midIsMax])
            / (2 * midVal[midIsMax] - leftVal[midIsMax] - rightVal[midIsMax])
        )
        binCorrection[:, ~midIsMax] = 0.5 * (
            rightVal[~midIsMax] == maxLeftRight[~midIsMax]
        ) - 0.5 * (leftVal[~midIsMax] == maxLeftRight[~midIsMax])

        freqInterp[i, :] = fftBinWidth * (maxBin[i, :] + binCorrection)
        deltaLocIdx = maxBin[i, :] + np.sign(binCorrection).astype(int)

        loc[i, :] = binToLoc[maxBin[i, :]] + binCorrection * np.abs(
            binToLoc[maxBin[i, :]] - binToLoc[deltaLocIdx]
        )

    return freqInterp, loc


def upsample(signal, n_cols, **kwargs):
    signal = np.repeat(signal, 3, axis=1)
    return np.concatenate((np.zeros((signal.shape[0], 2)), signal), axis=1)[:, :n_cols]


def current_steering_weights(
    loc,
    nChan=15,
    nDiscreteSteps=9,
    steeringRange=1.0,
    current_steering=False,
    **kwargs,
):
    """Calculate pairs of current steering weights per channel given estimated
    cochlear location of the spectral peak frequency per channel.

    INPUT:
        par - parameter object/struct
        loc - estimated peak locations per channel in "electrode locations",
                i.e. ranging from 0 (most apical el.) to 15 (most basal el.) and
                confined to range [i-1, i] for each channel i

    FIELDS FOR PAR:
        nChan - number of filterbank channels
        nDiscreteSteps - number of discretization steps
                        integer >= 0; 0 -> no discretization
        steeringRange - range of steering between electodes; either
                            - scalar range (in [0,1]) around 0.5 for all channels
                            - 1 x nChan vector with range (in [0,1] around 0.5 per channel
                            - 2 x nChan matrix with (absolute) lower and upper steering
                                limits (within [0,1]) per channel

    OUTPUT:
        weights - (2*nChan) x nFrames matrix of current steering weights;
            weights for the lower and higher electrode of channel i are
            contained in rows i and (i+nChan), resp.
    Copyright (c) 2012-2020 Advanced Bionics. All rights reserved.
    """
    assert (
        np.isscalar(nDiscreteSteps) & np.mod(nDiscreteSteps, 1) == 0
    ), "nSteps must be an integer scalar."

    if np.isscalar(steeringRange):
        steeringRange = 0.5 + 0.5 * steeringRange * np.concatenate(
            (-np.ones((1, nChan)), np.ones((1, nChan)))
        )
    elif len(steeringRange) > 1:
        assert (
            len(steeringRange) == nChan
        ), 'Length of vector "steeringRange" must equal # of channels.'
        steeringRange = 0.5 + 0.5 * np.concatenate((-steeringRange, steeringRange))

    assert steeringRange.shape == (
        2,
        nChan,
    ), 'Matrix "steeringRange" must have dimensions 2xnChan.'
    assert np.all(steeringRange >= 0) & np.all(
        steeringRange <= 1
    ), 'Values in "steeringRange" must lie in [0,1]'
    assert np.all(
        np.diff(steeringRange, axis=0)
    ), "range[:,2] >= range[:,1] must be true for all channels"

    nFrames = loc.shape[1]
    weights = np.zeros((nChan * 2, nFrames))

    if current_steering == False:
        nDiscreteSteps = 1

    for iCh in np.arange(nChan):

        weightHiRaw = loc[iCh, :] - iCh
        weightHiRaw = np.maximum(np.minimum(weightHiRaw, 1), 0)

        if nDiscreteSteps == 1:
            weightHiRaw = 0.5
        elif nDiscreteSteps > 1:
            weightHiRaw = np.floor(weightHiRaw * (nDiscreteSteps - 1) + 0.5) / (
                nDiscreteSteps - 1
            )
            # add +.5 and use floor to force round-half-away-from-zero (python round uses round-half-towards-even),
            # only works for positive values
        weightHi = steeringRange[0, iCh] + weightHiRaw * np.diff(steeringRange[:, iCh])
        weights[iCh, :] = 1 - weightHi
        weights[iCh + nChan, :] = weightHi

    return weights


def carrier_synthesis(
    fPeak,  # num_channels x num_frames matrix of estimated peak frequencies per channel
    nChan=15,  # number of analysis channels
    fs=17400,  # sampling rate of signal [Hz]
    nHop=20,  # step size for new frame
    pulseWidth=18,
    deltaPhaseMax=0.5,  # maximum phase rotation per FT frame [turns, 0.0 .. 1.0]
    maxModDepth=1.0,
    fModOn=0.5,
    fModOff=1.0,
    **kwargs,
):
    """INPUT:

      par - parameter object/struct
      fPeak - nChan x nFrames matrix of estimated peak frequencies per channel

    FIELDS FOR par:
      parent.nChan - number of analysis channels
      parent.fs - sample rate of signalIn [Hz]
      stimRate - channel stimulation rate in pps or Hz, I think he meant that this is rateFT
      fModOn - peak frequency up to which max. modulation depth is applied [fraction of FT rate]
      fModOff - peak frequency beyond which no modulation is applied  [fraction of FT rate]
      maxModDepth - maximum modulation depth [0.0 .. 1.0]
      deltaPhaseMax - maximum phase rotation per FT frame [turns, 0.0 .. 1.0]

    OUTPUT:
      carrier  - nChan x nFrameFt square-wave carrier signals
      tFtFrame - start time of each FT frame, starting with 0 [s]
    """

    nFrame = fPeak.shape[1]
    durFrame = nHop / fs  # duration of 1 audio frame [s]
    durStimCycle = (
        2 * pulseWidth * nChan * 1e-6
    )  # duration of a full stimulation cycle [s].
    # Depends on number of channels because 1 period contains stimulation of all electrodes
    rateFt = np.round(
        1 / durStimCycle
    )  # stimulation cycles/sec = forward telemetry rate  [Hz]

    nFtFrame = (
        np.ceil(durFrame * nFrame / durStimCycle) - 1
    )  # number of output forward-telemetry frames
    tFtFrame = np.arange(nFtFrame) * durStimCycle  # starting time of each FT frame

    idxAudFrame = (np.floor(tFtFrame / durFrame)).astype(
        int
    )  # index of last audio frame for each FT frame
    fPeakPerFtFrame = fPeak[
        :, idxAudFrame
    ]  # latest peak frequency estimate for each channel and FT frame

    # compute phase accumulation per channel and frame
    deltaPhiNorm = np.minimum(
        fPeakPerFtFrame / rateFt, deltaPhaseMax
    )  # delta phase in turns, i.e. [rad/2*pi]
    phiNorm = np.mod(
        np.cumsum(deltaPhiNorm, axis=1), 1
    )  # accumulated phase, modulo 1 [turns]

    # compute modulation depth (per channel and frame)
    fModOn = (
        rateFt * fModOn
    )  # peak frequency up to which max. modulation depth is applied [fraction of FT rate] [0.5]
    fModOff = (
        rateFt * fModOff
    )  # peak frequency beyond which no modulation is applied  [fraction of FT rate] [1.0]
    modDepth = (
        maxModDepth
        * (
            fModOff - np.minimum(np.maximum(fPeakPerFtFrame, fModOn), fModOff)
        )  # replace minima and maxima with minimal and maximal frequency
        / (fModOff - fModOn)  # normalize the peak frequencies between the two
    )

    # sythesize carrier: phase-dependent alternation x modulation depth
    carrier = 1 - (modDepth * (phiNorm < 0.5))

    return carrier, idxAudFrame
