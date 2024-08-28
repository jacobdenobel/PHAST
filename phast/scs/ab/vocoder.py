# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:29:54 2019

@author: beimx004
"""
import time
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.io
import scipy.interpolate


def ActivityToPower(alpha, activity, audioPwr, blkSize):
    for k in np.arange(blkSize):
        audioPwr[:, k + 1] = np.maximum(
            audioPwr[:, k] * alpha + activity[:, k] * (1 - alpha), activity[:, k]
        )
    audioPwr[:, 0] = audioPwr[:, blkSize]
    return audioPwr


def ElFieldToActivity(efData, normOffset, nl, nlExp):
    efData = efData - normOffset
    electricField = np.maximum(0, efData)
    electricField = electricField / 0.4 * 0.5
    activity = np.maximum(0, np.minimum(np.exp(nl * electricField), nlExp) - 1) / (
        nlExp - 1
    )
    return activity


def NeurToBinMatrix(neuralLocsOct, nFFT, Fs):

    fGrid = np.arange(0, np.floor(nFFT / 2) + 1) * (Fs / nFFT)
    fBinsOct = np.log2(fGrid[1:])

    binCountPerOct = np.divide(1, np.diff(fBinsOct))
    x = np.ones((2, np.floor(nFFT / 2).astype(int) - 1))
    x[1, :] = np.arange(1, np.floor(nFFT / 2), 1)

    nNeuralLocs = len(neuralLocsOct)
    mNeurToBin = np.zeros((np.floor(nFFT / 2).astype(int), nNeuralLocs))

    I = np.zeros(nNeuralLocs)
    for k in np.arange(len(neuralLocsOct)):
        tmp = np.abs(fBinsOct - neuralLocsOct[k])
        I[k] = np.argmin(tmp)
        mNeurToBin[I[k].astype(int), k] = 1

    basepath = Path(__file__).parent.absolute()
    pFN = basepath / "MatlabSupportFiles/preemph.mat"
    emph = sp.io.loadmat(pFN)

    I = np.argmax(emph["emphDb"])
    emph["emphDb"][I + 1 :] = emph["emphDb"][I]
    emphDb = -emph["emphDb"]
    emphDb = emphDb - emphDb[0]

    scl = np.interp(
        np.arange(1, np.floor(nFFT / 2) + 1) * Fs / nFFT,
        np.append(0, emph["emphF"]),
        np.append(0, emphDb),
    )

    mNeurToBin = np.multiply(mNeurToBin.T, 10 ** (scl / 20))
    mNeurToBin = np.nan_to_num(mNeurToBin).T

    return mNeurToBin


def generate_cfs(lo, hi, n_bands):
    """
    Generates a series of 'bands' frequencies in Hz, linearely distributed
    on an ERB scale between the frequencies 'lo' and 'hi' (in Hz).
    These would are the centre frequencies (on an ERB scale) of the bands
    specifications made by 'generate_bands' with the same arguments
    """

    density = n_bands / (hz2erb(hi) - hz2erb(lo))
    bands = []
    for i in np.arange(1, n_bands + 1):
        bands.append(erb2hz(hz2erb(lo) + (i - 0.5) / density))
    return bands


def erb2hz(erb):
    """
    Convert equivalent rectangular bandwidth (ERB) to Hertz.
    """
    tmp = np.exp((erb - 43.0) / 11.17)
    return (0.312 - 14.675 * tmp) / (tmp - 1.0) * 1000.0


def hz2erb(hz):
    """
    Convert Hertz to equivalent rectangular bandwidth (ERB).
    """
    return 11.17 * np.log((hz + 312.0) / (hz + 14675.0)) + 43.0


def vocoder(electrodogram, audioFs=48000, saveOutput=False, outputFile=None, **kwargs):
    # these parameters will be used by the website. DO NOT CHANGE FROM DEFAULT VALUES
    captFs = 55556
    nCarriers = 20
    elecFreqs = None
    spread = None
    neuralLocsOct = None
    nNeuralLocs = 300
    MCLmuA = 500
    TmuA = 50
    tAvg = 0.005
    tauEnvMS = 10
    nl = 5

    # Process electrodogram before proceeding to vocoder pipeline
    if type(electrodogram) is str:
        if electrodogram[-4:] == ".mat":
            rawData = scipy.io.loadmat(electrodogram)
            if "elData" in rawData.keys():
                electrodogram = rawData["elData"]
                if type(electrodogram) is sp.sparse.csc.csc_matrix:
                    electrodogram = electrodogram.A
                    if 16 in electrodogram.shape and len(electrodogram.shape) == 2:
                        if electrodogram.shape[0] == 16:
                            pass
                        else:
                            electrodogram = electrodogram.T
                    else:
                        raise ValueError(
                            "Electrodogram must be shape 16xn, (16 electrodes x n total samples)"
                        )
            else:
                raise KeyError(
                    'The supplied .mat file must contain data saved as "elData"'
                )

        elif electrodogram[-4:] == ".npy":
            rawData = np.load(electrodogram)
            electrodogram = rawData
            if 16 in electrodogram.shape and len(electrodogram.shape) == 2:
                if electrodogram.shape[0] == 16:
                    pass
                else:
                    electrodogram = electrodogram.T
            else:
                raise ValueError(
                    "Electrodogram must be shape 16xn, (16 electrodes x n total samples)"
                )

        elif electrodogram[-4:] == ".npz":
            rawData = sp.sparse.load_npz(electrodogram)
            electrodogram = rawData.A
            if 16 in electrodogram.shape and len(electrodogram.shape) == 2:
                if electrodogram.shape[0] == 16:
                    pass
                else:
                    electrodogram = electrodogram.T
            else:
                raise ValueError(
                    "Electrodogram must be shape 16xn, (16 electrodes x n total samples)"
                )

        else:
            raise ValueError(
                "Invalid File format: Only .npy, scipy sparse .npz, .h5, or .mat files are allowed"
            )
    elif type(electrodogram) is np.ndarray:
        if 16 in electrodogram.shape:
            if electrodogram.shape[0] == 16:
                pass
            else:
                electrodogram = electrodogram.T
        else:
            raise ValueError(
                "Electrodogram must be shape 16xn, (16 electrodes x n total samples)"
            )
    else:
        raise ValueError("Expected str or numpy ndarray inputs.")

    # Scale and preprocess electrodogram data
    # scaletoMuA = 500/resistorValue
    scaletoMuA = 1
    # electrodeAmp = electrodogram[:,1:] # the first column in the matrix is actually channel similarity.
    electrodeAmp = electrodogram
    nElec = electrodeAmp.shape[0]
    elData = electrodeAmp * scaletoMuA
    captTs = 1 / captFs

    # compute electrode locations in terms of frequency
    if elecFreqs is None:
        elecFreqs = np.logspace(np.log10(381.5), np.log10(5046.4), nElec)
    else:
        if nElec != elecFreqs.size:
            raise ValueError("# of electrode frequencies does not match recorded data!")
        else:
            elecFreqs = elecFreqs
    # load electric field spread data
    if spread is None:
        elecPlacement = np.zeros(nElec).astype(
            int
        )  # change to zeros to reflect python indexing
        basepath = Path(__file__).parent.absolute()
        spreadFile = (basepath / "MatlabSupportFiles/spread.mat").__str__()
        spread = scipy.io.loadmat(spreadFile)
    else:  # This seciont may need reindexing if the actual spread mat data is passed through, for now let use the spread.mat data
        elecPlacement = spread["elecPlacement"]

    # Create octave location of neural populations
    if neuralLocsOct is None:
        neuralLocsOct = np.append(
            np.log2(np.linspace(150, 850, 40)),
            np.linspace(np.log2(870), np.log2(8000), 260),
        )

    neuralLocsOct = np.interp(
        np.linspace(1, neuralLocsOct.size, nNeuralLocs),
        np.arange(1, neuralLocsOct.size + 1),
        neuralLocsOct,
    )

    # tauEnvMS to remove carrier synthesis effect
    taus = tauEnvMS / 1000
    alpha = np.exp(-1 / (taus * captFs))

    # MCT and T levels in micro amp
    if MCLmuA is None:
        MCLmuA = 500 * np.ones(nElec) * 1.2
    else:
        if (type(MCLmuA) == int) or (type(MCLmuA) == float):
            MCLmuA = np.ones(nElec) * MCLmuA * 1.2
        elif (type(MCLmuA) == "numpy.ndarray") and (MCLmuA.size == nElec):
            MCLmuA = MCLmuA * 1.2
        else:
            raise ValueError("Wrong number of M levels!")

    if TmuA is None:
        TmuA = 50 * np.ones(nElec)
    else:
        if (type(TmuA) == int) or (type(TmuA) == float):
            TmuA = np.ones(nElec) * TmuA
        elif (type(TmuA) == "numpy.ndarray") and (TmuA.size == nElec):
            TmuA = TmuA
        else:
            raise ValueError("Wrong Number of T levels!")

    # Time constant for averaging neural activity to relate to frequency
    tAvg = np.ceil(tAvg / captTs) * captTs
    mAvg = np.round(tAvg / captTs)
    blkSize = mAvg.astype(int)

    # audio output frequency
    audioFs = np.ceil(tAvg * audioFs) / tAvg
    audioTs = 1 / audioFs
    tWin = 2 * tAvg
    nFFT = np.round(tWin / audioTs).astype(int)

    # create matrix to convert electrode charge to electric field
    charge2EF = np.zeros((nNeuralLocs, nElec))
    elecFreqOct = np.log2(elecFreqs)

    for iEl in np.arange(nElec):
        f = scipy.interpolate.interp1d(
            spread["fOct"][:, elecPlacement[iEl]] + elecFreqOct[iEl],
            spread["voltage"][:, elecPlacement[iEl]],
            fill_value="extrapolate",
        )
        steerVec = f(neuralLocsOct)
        steerVec[steerVec < 0] = 0
        charge2EF[:, iEl] = steerVec

    # matrix to map neural activity to FFT bin frequencies
    # here we call another function
    mNeurToBin = NeurToBinMatrix(neuralLocsOct, nFFT, audioFs)

    # Define auxilliary variables

    #    random phase
    phs = 2 * np.pi * np.random.rand(np.floor(nFFT / 2).astype(int))

    # preallocate audio power matrix
    audioPwr = np.zeros((nNeuralLocs, blkSize + 1))

    # interpolate M and T levels to match neural locations
    M = np.interp(neuralLocsOct, elecFreqOct, MCLmuA)
    M[neuralLocsOct < elecFreqOct[0]] = MCLmuA[0]
    M[neuralLocsOct > elecFreqOct[nElec - 1]] = MCLmuA[nElec - 1]

    T = np.interp(neuralLocsOct, elecFreqOct, TmuA)
    T[neuralLocsOct < elecFreqOct[0]] = TmuA[0]
    T[neuralLocsOct > elecFreqOct[nElec - 1]] = TmuA[nElec - 1]

    normRamp = np.multiply(charge2EF.T, 1 / (M - T)).T
    normOffset = (T / (M - T)).reshape((T.size, 1))

    elData[elData < 0] = 0
    nlExp = np.exp(nl)

    # Generate output carrier tone complex
    nBlocks = (nFFT / 2 * (np.floor(elData.shape[1] / blkSize + 1))).astype(int) - 1
    tones = np.zeros((nBlocks, nCarriers))
    toneFreqs = generate_cfs(20, 20000, nCarriers)
    t = np.arange(nBlocks) / audioFs

    for toneNum in np.arange(nCarriers):
        tones[:, toneNum] = np.sin(
            2.0 * np.pi * toneFreqs[toneNum] * t + phs[toneNum]
        )  # random phase

    interpSpect = np.zeros(
        (nCarriers, np.floor(elData.shape[1] / blkSize).astype(int)), dtype=complex
    )
    fftFreqs = np.arange(1, np.floor(nFFT / 2) + 1) * audioFs / nFFT
    # Loop through frames of electrode data, convert to electric field, calculate neural spectrum
    for blkNumber in np.arange(
        1, (np.floor(elData.shape[1] / blkSize).astype(int)) + 1
    ):
        # charge to electric field
        timeIdx = (
            np.arange((blkNumber - 1) * blkSize + 1, blkNumber * blkSize + 1, dtype=int)
            - 1
        )
        efData = np.dot(normRamp, elData[:, timeIdx])

        # Normalized EF to neural activity
        activity = ElFieldToActivity(efData, normOffset, nl, nlExp)  # JIT optimized

        # Neural activity to audio power
        audioPwr = ActivityToPower(
            alpha, activity, audioPwr, blkSize
        )  # JIT optimized inner loop

        # Average energy
        energy = np.sum(audioPwr, axis=1) / mAvg

        fMagInt = scipy.interpolate.interp1d(
            fftFreqs, np.dot(mNeurToBin, energy), fill_value="extrapolate"
        )

        # calculate tone
        toneMags = fMagInt(toneFreqs)
        interpSpect[:, blkNumber - 1] = toneMags

    # interpolated spectral envelope tone scaling

    specVec = np.arange(blkNumber) * nFFT / 2
    newTimeVec = np.arange(nBlocks - (nFFT / 2 - 1))
    modTones = np.zeros((len(toneFreqs), len(newTimeVec)))
    for freq in np.arange(len(toneFreqs)):
        fEnvMag = scipy.interpolate.interp1d(
            specVec, interpSpect[freq, :], fill_value="extrapolate"
        )
        tEnvMag = fEnvMag(newTimeVec)
        modTones[freq, :] = tones[: -(nFFT / 2 - 1).astype(int), freq] * np.abs(tEnvMag)

    audioOut = np.sum(modTones, axis=0)
    audioOut = audioOut / np.sqrt(np.mean(np.square(audioOut))) * 10 ** (-25 / 20)

    if saveOutput:
        if outputFile is None:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            relativeOutputFile = "Output/VocoderOutput_" + timestr + ".wav"
            outputFile = (basepath / relativeOutputFile).__str__()
        amplitude = np.iinfo(np.int16).max
        audioToSave = audioOut * amplitude

        scipy.io.wavfile.write(outputFile, audioFs.astype(int), np.int16(audioToSave))

    return (audioOut, audioFs.astype(int))
