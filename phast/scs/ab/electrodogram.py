# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:29:42 2019

@author: beimx004
"""

import numpy as np

from scipy.signal import lfilter
from scipy.interpolate import interp1d

from .defaults import DEFAULT_CHANNEL_ORDER


def f120(
    ampIn,
    nChan=15,
    pulseWidth=18,
    outputFs=None,
    channelOrder=None,
    cathodicFirst=True,
    weights=None,
    virtual_channels=True,
    charge_balanced=True,
    nDiscreteSteps=9,
    **kwargs,
):
    """
    elGram = f120ElectrodogramFunc(par, ampIn)
    Generate scope-like electrodogram from matrix of F120 amplitude frame.
    Amplitude frames are expected to represent the amplitude( pair)s for each
    channel by a pair of consecutive rows each (as provided e.g. by F120MappingUnit)

    Input:
      par - parameter object/struct
      ampIn - 2*nChan x nFtFrames matrix of stimulation amplitudes [uA]

    Fields of par:
      channelOrder - 1 x nChan vector defining the firing order among channels
                     [1..nChan, unique] [[1 5 9 13 2 6 10 14 3 7 11 15 4 8 12]]
      outputFs - output sampling frequency; [] for native FT rate  [Hz] [[]]
                 (resampling is done using zero-order hold method)
      cathodicFirst - start biphasic pulse with cathodic phase [bool] [true]
      enablePlot - generate electrodogram plot? [bool]
      colorScheme - color scheme for plot; [1..4] 1/2 more subdued, 3/4 more strident colors; odd/even affects color order

    Output:
      elGram - 16 x nSamp matrix of electrode current flow; [uA]

    Copyright (c) 2019-2020 Advanced Bionics. All rights reserved.
    """

    nFrameFt = ampIn.shape[1]
    phasesPerCyc = 2 * nChan
    dtIn = phasesPerCyc * pulseWidth * 1e-6
    durIn = nFrameFt * dtIn

    if channelOrder is None:
        channelOrder = DEFAULT_CHANNEL_ORDER

    assert nChan == 15, "only 15-channel strategies are supported."
    assert channelOrder.shape[0] == nChan, "length(channelOrder) must match nChan"

    n_samples = nFrameFt * phasesPerCyc

    idxLowEl = np.arange(nChan)
    idxHighEl = np.arange(nChan) + 1
    n_electrodes = 16

    pulse_train = np.zeros((n_electrodes, n_samples))

    # TODO: this should be the same as nDiscreteSteps
    n_virtual = nDiscreteSteps if kwargs.get("current_steering", True) else 1

    if n_virtual == 1:
        pt_v = np.zeros((nChan, n_samples))
    else:
        n_virtual_channels = nChan * n_virtual  # - (nChan - 1)
        pt_v = np.zeros((n_virtual_channels, n_samples))
        v_weights = np.arange(0, 1.1, 1 / (n_virtual - 1))[::-1]

    for iCh in np.arange(nChan):
        phaseOffset = 2 * (channelOrder[iCh] - 1)
        amp1 = ampIn[2 * iCh, :]
        amp2 = ampIn[2 * iCh + 1, :]

        pulse_train[idxLowEl[iCh], phaseOffset::phasesPerCyc] = amp1
        pulse_train[idxHighEl[iCh], phaseOffset::phasesPerCyc] = amp2

        if n_virtual == 1:
            pt_v[iCh, phaseOffset::phasesPerCyc] = amp1 + amp2
            continue

        aw1 = weights[iCh, :] * (amp1 != 0)
        aw2 = weights[iCh + nChan, :] * (amp2 != 0)

        for vidx, w1 in enumerate(v_weights):
            v_channel = vidx + ((v_weights.size) * iCh)
            w2 = 1 - w1
            mask = np.logical_and(aw1 == w1, aw2 == w2)
            pt_v[v_channel, phaseOffset::phasesPerCyc][mask] = amp1[mask] + amp2[mask]

    if virtual_channels:
        pulse_train = pt_v

    if charge_balanced:
        kernel = np.array([-1, 1]) if cathodicFirst else np.array([1, -1])
        pulse_train = lfilter(kernel, 1, pulse_train)

    if outputFs is not None:
        pulse_train = resample_to_fs(
            pulse_train, outputFs, n_samples, durIn, pulseWidth
        )

    pulse_train *= 1e-6

    return pulse_train


def resample_to_fs(
    pulse_train: np.ndarray,
    outputFs: int,
    nFrameOut: int,
    durIn: float,
    pulseWidth: float,
) -> np.ndarray:

    dtOut = 1 / outputFs
    tPhase = np.arange(nFrameOut) * pulseWidth * 1e-6
    tOut = np.arange(np.floor(durIn / dtOut)) * dtOut
    fElGram = interp1d(tPhase, pulse_train, kind="previous", fill_value="extrapolate")
    pulse_train = fElGram(tOut)
    return pulse_train


def pulse_train_to_virtual(
    pulse_train: np.ndarray,
    weights_matrix: np.ndarray,
    n_virtual: int = 8,
    charge_balanced: bool = False,
):
    (n_electrodes, n_samples) = weights_matrix.shape
    n_virtual_channels = (n_electrodes - 1) * n_virtual + 1
    pulse_times, pulse_electrodes = np.where(pulse_train.T < 0)
    pulse_train_virtual = np.zeros((n_virtual_channels, n_samples))
    pt_v = np.zeros((n_virtual_channels, n_samples))
    pt_neg = pulse_train.copy()
    pt_neg[pt_neg > 0] = 0

    print(pt_neg.sum(axis=1))

    weights_map = (
        {(0.5, 0.5): 0}
        if n_virtual == 1
        else {
            (float(x), 1 - float(x)): e
            for e, x in enumerate(np.arange(0, 1.1, 1 / n_virtual)[::-1], 1)
        }
    )

    if n_virtual == 1:
        weights = np.array([0.5])
    else:
        weights = np.arange(0, 1.1, 1 / n_virtual)[::-1]

    for target in range(n_virtual_channels):
        alpha = target % n_virtual
        e2 = int(np.floor(target / n_virtual))
        e1 = e2 - 1
        print(e1)
        w = float(weights[alpha])
        mask = np.logical_and(weights_matrix[e1] == w, weights_matrix[e2] == 1 - w)
        pt_v[target, mask] += (
            weights_matrix[e1, mask] * pulse_train[e1, mask]
            + weights_matrix[e2, mask] * pulse_train[e2, mask]
        )

    print(pt_v.sum(axis=1))
    breakpoint()

    for el in range(n_electrodes):
        pulse_times_electrode = pulse_times[pulse_electrodes == el]

        if el == 15:
            el -= 1  # only loop over electrode, don't add to count
            el_pair = [14, 15]
        else:
            el_pair = [el, el + 1]

        for pt in pulse_times_electrode:
            weights_pair = tuple(map(float, weights_matrix[el_pair, pt]))
            if weights_pair not in weights_map:
                continue
            virtual_channel_id = int(weights_map[weights_pair] + el * n_virtual - 1)

            pulse_pair = pulse_train[el_pair, pt]
            pulse_train_virtual[virtual_channel_id, pt] = np.sum(pulse_pair)

    if charge_balanced:
        return lfilter(np.array([1, -1]), 1, pulse_train_virtual)

    print(pulse_train_virtual.sum(axis=1))
    breakpoint()
    return pulse_train_virtual
