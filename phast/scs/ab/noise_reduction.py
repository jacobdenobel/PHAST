# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:19:09 2019

@author: beimx004
"""

# [G_out, A_out, Vn, Vs, Hold] = noiseReductionFunc(initState, A)
#
# Compute channel-by-channel noise reduction gains.
#
# INPUT:
#  initState - parameter object / struct
#  A - nCh x nFrames matrix of channel amplitudes (sqrt(power), linearly scaled)
#
# OUTPUT:
#  G_out - nCh x nFrames matrix of noise reduction gains (domain determined by initState.gainDomain)
#  A_out - nCh x nFrames matrix of channel amplitudes (sqrt(power), linearly scaled)
#  Vn   - nCh x nFrames matrix of noise estimates
#  Vs   - nCh x nFrames matrix of speech estimates
#  Hold - nCh x nFrames matrix of hold states
#
# FIELDS FOR initState:
#  parent.fs   - audio sample rate [int > 0]
#  parent.nHop - FFT hop size [int > 0]
#  gainDomain - domain of gain output ['linear','db','log2 ['linear
#  tau_speech - time constant of speech estimator [s]
#  tau_noise - time constant of noise estimator [s]
#  threshHold - hold threshold (onset detection criterion) [dB, > 0]
#  durHold - hold duration (following onset) [s]
#  maxAtt - maximum attenuation (applied for SNRs <= snrFloor) [dB]
#  snrFloor - SNR below which the attenuation is clipped [dB]
#  snrCeil  - SNR above which the gain is clipped  [dB]
#  snrSlope - SNR at which gain curve is steepest  [dB]
#  slopeFact  - factor determining the steepness of the gain curve [> 0]
#  noiseEstDecimation - down-sampling factor (re. frame rate) for noise estimate [int > 0]
#  enableContinuous - save final state for next execution? [boolean]

import numpy as np


def noise_reduction(
    A,
    fs=17400,
    nHop=20,
    gainDomain="log2",
    tau_speech=0.0258,
    tau_noise=0.219,
    threshHold=3,
    durHold=1.6,
    maxAtt=-12,
    snrFloor=-2,
    snrCeil=45,
    snrSlope=6.5,
    slopeFact=0.2,
    noiseEstDecimation=1,
    enableContinuous=False,
    initState={
        # 'V_s' : -30*np.ones((15)),'V_n' : -30*np.ones((15))
    },
    **kwargs,
):
    """Compute channel-by-channel noise reduction (NR) gains. This unit can be used
    with 1 to 4 outputs (see below).

    Input:
    A - nCh x nFrames matrix of channel amplitudes (sqrt(power), linearly scaled)

    Output:
    G_out - nCh x nFrames matrix of NR gains (domain determined by par.gainDomain)
    A_out - nCh x nFrames matrix of channel amplitudes (sqrt(power), linearly scaled)
    Vn   - nCh x nFrames matrix of noise estimates
    Vs   - nCh x nFrames matrix of speech estimates
    Hold - nCh x nFrames matrix of hold states"""

    dtFrame = nHop / fs

    nCh, nFrame = A.shape
    alpha_s = np.exp(-dtFrame / tau_speech)
    alpha_n = np.exp(-dtFrame / tau_noise)

    maxHold = durHold / (dtFrame * noiseEstDecimation)
    maxAttLin = 10 ** (-np.abs(maxAtt) / 20)

    gMin = 1 + (maxAttLin - 1) / (
        1 - 1 / (1 + np.exp(-slopeFact * (snrFloor - snrSlope)))
    )

    G = np.empty((nCh, nFrame))
    Vs_out = np.empty((nCh, nFrame))
    Vn_out = np.empty((nCh, nFrame))
    Hold_out = np.empty((nCh, nFrame))

    logA = np.maximum(-100, 20 * np.log10(A))

    V_s = initState.get("V_s", np.zeros(nCh))
    V_n = initState.get("V_n", np.zeros(nCh))
    Hold = initState.get("Hold", np.zeros(nCh, dtype=bool))
    HoldReady = initState.get("HoldReady", np.zeros(nCh, dtype=bool))
    HoldCount = initState.get("HoldCount", np.zeros(nCh) + maxHold)

    for iFrame in np.arange(nFrame):
        V_s = alpha_s * V_s + (1 - alpha_s) * logA[:, iFrame]
        if np.mod(iFrame - 1, noiseEstDecimation) == noiseEstDecimation - 1:
            maskSteady = (V_s - V_n) < threshHold
            maskOnset = ~maskSteady & HoldReady
            maskHold = ~maskSteady & ~HoldReady & Hold

            maskUpdateNoise = maskSteady | (~maskSteady & ~HoldReady & ~Hold)
            V_n[maskUpdateNoise] = (
                alpha_n * V_n[maskUpdateNoise] + (1 - alpha_n) * V_s[maskUpdateNoise]
            )

            Hold[maskOnset] = True
            HoldReady[maskOnset] = False
            HoldCount[maskOnset] = maxHold

            HoldCount[maskHold] -= 1
            Hold[np.squeeze(maskHold & [HoldCount <= 0])] = False
            Hold[maskSteady] = False
            HoldReady[maskSteady] = True

        # compute gains
        SNR = V_s - V_n
        G[:, iFrame] = gMin + np.divide(
            (1 - gMin),
            1
            + np.exp(
                -slopeFact * (np.minimum(np.maximum(SNR, snrFloor), snrCeil) - snrSlope)
            ),
        )

        Vn_out[:, iFrame] = V_n
        Vs_out[:, iFrame] = V_s
        Hold_out[:, iFrame] = Hold

    # Apply Gains
    A_out = np.multiply(A, G)

    if gainDomain.lower() == "linear" or gainDomain.lower() == "lin":
        G_out = G
    elif gainDomain.lower() == "log" or gainDomain.lower() == "log2":
        G_out = 2 * np.log2(G)
    elif gainDomain.lower() == "db":
        G_out = 20 * np.log10(G)
    else:
        raise ValueError("Illegal value for parameter " "gainDomain" "")

    if enableContinuous:
        initState = {
            "V_s": V_s,
            "V_n": V_n,
            "Hold": Hold,
            "HoldReady": HoldReady,
            "HoldCount": HoldCount,
        }

    return G_out, A_out, Vn_out, Vs_out, Hold_out
