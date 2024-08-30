import scipy
import numpy as np
from numpy.lib.stride_tricks import as_strided

from .parameters import Parameters


def buffer(signal, n, p=0):
    """Buffer a signal into overlapping frames with padding."""
    step = n - p
    pad_signal = np.concatenate((np.zeros(p), signal))  # Padding at the start
    num_frames = (len(pad_signal) - p + step - 1) // step  # Number of frames
    padded_length = num_frames * step + p

    # Zero-pad signal to make it fit into an integer number of frames
    pad_signal = np.concatenate((pad_signal, np.zeros(padded_length - len(pad_signal))))

    shape = (num_frames, n)
    strides = (pad_signal.strides[0] * step, pad_signal.strides[0])
    frames = as_strided(pad_signal, shape=shape, strides=strides)
    return frames


def filterbank(signal: np.ndarray, parameters: Parameters) -> np.ndarray:
    parameters.window = scipy.signal.windows.hann(parameters.block_length, False)
    buffer_opt = []

    block_shift = int(
        np.ceil(parameters.audio_sample_rate_Hz / parameters.analysis_rate_Hz)
    )
    parameters.analysis_rate_Hz = parameters.audio_sample_rate_Hz / block_shift

    num_bins = int(parameters.block_length / 2 + 1)

    parameters.bin_freq_Hz = parameters.audio_sample_rate_Hz / parameters.block_length
    parameters.bin_freqs_Hz = parameters.bin_freq_Hz * np.arange(num_bins)

    buff = buffer(
        signal, parameters.block_length, parameters.block_length - block_shift
    )
    spectrum = np.fft.fft(buff * parameters.window)[:, :num_bins]
    return spectrum


FFT_BAND_BINS = {
    22: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
    21: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8],
    20: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8],
    19: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
    18: [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
    17: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
    16: [1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7, 9, 11],
    15: [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 9, 13],
    14: [1, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 9, 13],
    13: [1, 2, 2, 2, 2, 3, 3, 4, 5, 7, 8, 10, 13],
    12: [1, 2, 2, 2, 2, 3, 4, 5, 7, 9, 11, 14],
    11: [1, 2, 2, 2, 3, 4, 5, 7, 9, 12, 15],
    10: [2, 2, 3, 3, 4, 5, 7, 9, 12, 15],
    9: [2, 2, 3, 3, 5, 7, 9, 13, 18],
    8: [2, 2, 3, 4, 6, 9, 14, 22],
    7: [3, 4, 4, 6, 9, 14, 22],
    6: [3, 4, 6, 9, 15, 25],
    5: [3, 4, 8, 16, 31],
    4: [7, 8, 16, 31],
    3: [7, 15, 40],
    2: [7, 55],
    1: [62],
}


def envelope_method(spectrum: np.ndarray, parameters: Parameters):
    if parameters.envelope_method == "power sum":
        return power_sum_envelope(spectrum, parameters)

    # TODO: We still need default vector sum method
    raise NotImplementedError()


def power_sum_envelope(
    spectrum: np.ndarray,
    parameters: Parameters,
):
    num_bins = spectrum.shape[1]
    num_bands = parameters.num_bands
    band_bins = np.array(FFT_BAND_BINS[num_bands]).T

    # Weights matrix for combining FFT bins into bands:
    weights = np.zeros((num_bands, num_bins))

    # Optionally incorporate frequency response equalisation:
    _, freq_response = scipy.signal.freqz(parameters.window / 2, 1, 128)
    power_response = (freq_response * freq_response.conj()).real

    P = (
        power_response[0],
        2 * power_response[1],
        power_response[0] + (2 * power_response[2]),
    )
    bin = 2  # We always ignore bins 0 (DC) & 1.
    for band, width in enumerate(band_bins):
        width = band_bins[band]
        weights[band, bin : (bin + width)] = 1
        if parameters.equalise:
            weights[band] = weights[band] / P[min(width, 3) - 1]
        bin += width

    # Not sure if needed
    cum_num_bins = np.r_[1.5, 1.5 + np.cumsum(band_bins)]
    crossover_freqs_Hz = cum_num_bins * parameters.bin_freq_Hz
    band_widths_Hz = np.diff(crossover_freqs_Hz)
    best_freqs_Hz = crossover_freqs_Hz[:num_bands] + (band_widths_Hz / 2)

    power_spectrum = (spectrum * np.conj(spectrum)).real
    power_spectrum = np.sqrt(weights @ power_spectrum.T)
    return power_spectrum
