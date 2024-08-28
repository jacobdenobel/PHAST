# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:17:40 2019

[wavOut GExpand State C CSlow CFast Hold Env G EnvFast] = dualLoopTdAgcFunc(par, wavIn, [ctrl])

Apply the (single-channel) Harmony time-domain dual-loop AGC to input.
Implementation based on Teak model (agc17.c and agca.c)

INPUT:
  wavIn - input waveform (max. range [-1,1])
  ctrl  - (optional) additional "control" signal that is used to determine
          the gain to be applied to wavIn; wavIn itself is used as control
          is no other is explcitly provided

FIELDS FROM PAR: sampling frequency
 parent.fs - audio sample rate
 kneePt - compression threshold (in log2 power)
 compRatio -  compression ratio above kneepoint (in log-log space)
 tauRelFast - fast release [ms]
 tauAttFast - fast attack [ms]
 tauRelSlow - slow release [ms]
 tauAttSlow - slow attack [ms]
 maxHold - max. hold counter value
 g0 - gain for levels < kneepoint  (log2)
 fastThreshRel - relative threshold for fast loop [dB]
 clipMode   - output clipping behavior: 'none' / 'limit' / 'overflow'
 decFact    - decimation factor
 envBufLen  - buffer length for envelope computation
 gainBufLen - buffer length for gain smoothing
 cSlowInit - initial value for slow averager, 0..1, [] for auto-scale to avg signal amp.
 cFastInit - initial value for fast averager, 0..1, [] for auto-scale
 envCoefs - data window for envelope computation
 controlMode - how to use control signal, if provided on port #2? [string];
                 'naida'  - actual control signal is 0.75*max(abs(control, audio))
                 'direct' - control signal is used verbatim without further processing

OUTPUT:
 wavOut - output waveform
 G - gain vector (linear, sample-by-sample)
 State - state vector (0: release, 1: hold,  2:slow attack, fast release,3:slow attack, fast attack)
 C - vector of effective "input levels"
 CSlow - vector of slow averager values
 CFast - vector of fast averager values
 Hold - hold counter vector

See also: DualLoopTdAgcUnit

Change log:
27/11/2012, P.Hehrmann - created
14/01/2012, P.Hehrmann - renamed;
                        fixed: temporal alignment wavIn <-> gains (consistent with fixed-point GMT implementation / C model)
01/06/2015, PH - adapted to May 2015 framework: removed shared props
28/09/2015, PH - added 'auto' option for initial conditions
01/Dec/2017, PH - add "controlMode" property
14 Aug 2019, PH - swapped function arguments

@author: beimx004
"""


import numpy as np
from scipy import signal
from warnings import warn
from collections import namedtuple

from .defaults import DEFAULT_COEFFS

agc = namedtuple(
    "agc", ["smpGain", "State", "C", "CSlow", "CFast", "Hold", "Env", "G", "EnvFast"]
)


def dual_loop_td_agc(
    signal_in,  # signal still in time domain (max. range [-1,1])
    fs=17400,  # audio sample rate
    kneePt=4.476,  # compression threshold (in log2 power)
    compRatio=12,  # compression ratio above kneepoint (in log-log space)
    tauRelFast=-8 / (17400 * np.log(0.9901)) * 1000,  # fast release [ms]
    tauAttFast=-8 / (17400 * np.log(0.25)) * 1000,  # fast attack [ms]
    tauRelSlow=-8 / (17400 * np.log(0.9988)) * 1000,  # slow release [ms]
    tauAttSlow=-8 / (17400 * np.log(0.9967)) * 1000,  # slow attack [ms]
    maxHold=1305,  # max. hold counter value
    g0=6.908,  # gain for levels < kneepoint  (log2)
    fastThreshRel=8,  # relative threshold for fast loop [dB]
    cSlowInit=0.5e-3,  # initial value for slow averager, 0..1, [] for auto-scale to avg signal amp.
    cFastInit=0.5e-3,  # initial value for fast averager, 0..1, [] for auto-scale
    controlMode="naida",  # how to use control signal, if provided on port #2? [string];
    #'naida'  - actual control signal is 0.75*max(abs(control, audio))
    #'direct' - control signal is used verbatim without further processing
    clipMode="limit",  # output clipping behavior: 'none' / 'limit' / 'overflow'
    decFact=8,  # decimation factor, turns the whole time domain signal into 8 blocks
    envBufLen=32,  # buffer length for envelope computation
    gainBufLen=16,  # buffer length for gain smoothing
    envCoefs=None,  # data window for envelope computation
    ctrl=None,  # (optional) additional "control" signal that is used to determine
    # the gain to be applied to wavIn; wavIn itself is used as control
    # is no other is explcitly provided
    **kwargs,
):

    # check input dimensions
    assert isinstance(signal_in, np.ndarray), "wavIn must be a numpy array!"
    if envCoefs is None:
        envCoefs = DEFAULT_COEFFS

    if ctrl is None:  # no explicit control provided, use audio
        ctrl = signal_in
    else:  # control signal is provided, use the specified control mode option
        assert isinstance(ctrl, np.ndarray), "ctrl must be a numpy array!"
        nSamp = np.min([signal_in.size, ctrl.size])
        signal_in = signal_in[0 : nSamp - 1]
        if controlMode.lower() == "naida":
            ctrl = ctrl[0 : nSamp - 1]
            ctrl = 0.75 * np.maximum(np.abs(signal_in), np.abs(ctrl))
        elif controlMode.lower() == "direct":
            ctrl = ctrl[0 : nSamp - 1]
        else:
            raise ValueError("Unknown control mode setting: ", controlMode)

    # general parameters
    c0_log2 = kneePt - 15
    c0 = 2**c0_log2
    gainSlope = 1 / compRatio - 1
    fastHdrm = 10 ** (-fastThreshRel / 20)

    # averaging weights
    bAttSlow = np.exp(-decFact / fs * 1000 / tauAttSlow)
    bRelSlow = np.exp(-decFact / fs * 1000 / tauRelSlow)
    bAttFast = np.exp(-decFact / fs * 1000 / tauAttFast)
    bRelFast = np.exp(-decFact / fs * 1000 / tauRelFast)

    nSamp = ctrl.size
    nFrame = np.ceil(nSamp / decFact).astype(int)

    # pre-allocation
    Env = np.empty(nFrame)
    CSlow = np.empty(nFrame)
    CFast = np.empty(nFrame)
    C = np.empty(nFrame)
    G = np.empty(nFrame)
    Hold = np.empty(nFrame)
    State = np.empty(nFrame)
    EnvFast = np.empty(nFrame)

    # inital conditions
    cSlow_i = cSlowInit
    if isinstance(cSlow_i, np.ndarray) or isinstance(cSlow_i, list):
        if len(cSlow_i) == 0:
            cSlow_i = np.minimum(np.mean(np.abs(ctrl)) * np.sum(envCoefs), 1)
    cFast_i = cFastInit
    if isinstance(cFast_i, np.ndarray) or isinstance(cFast_i, list):
        if len(cFast_i) == 0:
            cFast_i = np.minimum(np.mean(np.abs(ctrl)) * np.sum(envCoefs) * fastHdrm, 1)

    cFastLowLimit_i = cFast_i
    hold_i = 0

    # loop over blocks
    for iFrame in np.arange(nFrame):
        idxWav = (iFrame + 1) * decFact + np.arange(-(envBufLen), 0) - 1

        idxWav = idxWav[idxWav >= 0]
        idxWav = idxWav[idxWav < nSamp]

        # compute envelope
        envLen = len(envCoefs[-idxWav.size :])
        envWin = envCoefs[-idxWav.size :]
        env_i = np.sum(np.abs(ctrl[0, idxWav]) * envCoefs[-idxWav.size :])
        # Use only first channel wavform needs to be single channel at this state
        envFast_i = clip1(env_i * fastHdrm)
        # update envelope averagers
        if env_i > cSlow_i:
            fastThr_i = clip1(cSlow_i * 10 ** (8 / 20))
            if env_i > fastThr_i:
                deltaHold = 0
                cFast_i = track(cFast_i, envFast_i, bAttFast)
                state_i = 3
            else:
                deltaHold = 2
                cFastLowLimit_i = cSlow_i * 10 ** (-10 / 20)
                cFast_i = track(cFast_i, envFast_i, bRelFast)
                state_i = 2
            cSlow_i = track(cSlow_i, min((env_i, fastThr_i)), bAttSlow)
        elif hold_i == 0:
            deltaHold = 0
            cFastLowLimit_i = cSlow_i * 10 ** (-10 / 20)
            cFast_i = track(cFast_i, envFast_i, bRelFast)
            cSlow_i = track(cSlow_i, env_i, bRelSlow)
            state_i = 0
        else:
            deltaHold = -1
            cFastLowLimit_i = cSlow_i * 10 ** (-10 / 20)
            cFast_i = track(cFast_i, envFast_i, bRelFast)
            state_i = 1

        hold_i = min((hold_i + deltaHold, maxHold))

        # clip values
        cSlow_i = max((cSlow_i, c0))
        cFast_i = max((cFast_i, cFastLowLimit_i))

        # select fast/slow averager for gain computation
        c_i = max((cFast_i, cSlow_i))

        # compute gain
        c_i_log2 = np.log2(max((c_i, 10**-16)))
        g_i = 2 ** (g0 + gainSlope * max((c_i_log2 - c0_log2, 0)))

        # store variables
        G[iFrame] = g_i
        Env[iFrame] = env_i
        C[iFrame] = c_i
        CSlow[iFrame] = cSlow_i
        CFast[iFrame] = cFast_i
        Hold[iFrame] = hold_i
        State[iFrame] = state_i
        EnvFast[iFrame] = envFast_i

    # apply gain
    idxExpand = np.concatenate(
        (
            np.ceil(np.arange(1 / decFact, nFrame + 1 / decFact, 1 / decFact)),
            np.array([nFrame]),
        )
    ).astype(int)
    smpGain = G[idxExpand - 1]
    smpGain = signal.lfilter(np.ones(gainBufLen) / gainBufLen, 1, smpGain)
    smpGain = smpGain[1 : nSamp + 2 - gainBufLen]
    smpGain = smpGain.reshape((1, smpGain.size))
    # agc['wavOut'] = np.concatenate((np.zeros((1,envBufLen)),wavIn[:,gainBufLen:nSamp-envBufLen+1]),axis=1)*agc['smpGain']

    wavPad = signal_in[:, np.max((0, gainBufLen - envBufLen)) : gainBufLen]
    zeroPad = np.zeros((1, envBufLen - wavPad.shape[1]))

    wavOut = (
        np.concatenate(
            (zeroPad, wavPad, signal_in[:, gainBufLen : nSamp - envBufLen + 1]), axis=1
        )
        * smpGain
    )

    nOutTooShort = signal_in.shape[1] - wavOut.shape[1]

    if nOutTooShort > 0:
        wavOut = np.concatenate(
            (
                wavOut,
                smpGain[:, -1]
                * signal_in[
                    :, nSamp - envBufLen + 1 : nSamp - envBufLen + nOutTooShort
                ],
            ),
            axis=1,
        )
    elif nOutTooShort != 0:
        wavOut = wavOut[:, : signal_in.shape[1]]

    wavOut = wavOut.reshape((1, wavOut.size))

    if clipMode.lower() == "none":
        pass
    elif clipMode.lower() == "limit":
        wavOut = np.maximum(-1, np.minimum(1, wavOut))
    elif clipMode.lower() == "overflow":
        wavOut = np.mod(1 + wavOut, 2) - 1
    else:
        warn("Unknown clipping mode: " + clipMode + " . Using " "none" " instead.")

    return wavOut, agc(smpGain, State, C, CSlow, CFast, Hold, Env, G, EnvFast)


def track(c_prev, In, weightPrev):
    weightIn = 1 - weightPrev
    return In * weightIn + c_prev * weightPrev


def clip1(In):
    return np.max([-1, np.min([1, In])])
