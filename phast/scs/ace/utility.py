import numpy as np

from .parameters import Parameters, from_dB


def gain(
    channel_power: np.ndarray,
    parameters: Parameters,
) -> np.ndarray:
    gain = from_dB(parameters.amp.gain_dB)
    return channel_power * gain


def resample(
    channel_power: np.ndarray, parameters: Parameters, resample_method: str = "repeat"
) -> np.ndarray:
    if parameters.rate.channel_stim_rate_Hz != parameters.rate.analysis_rate_Hz:
        raise NotImplementedError
    return channel_power


def reject_smallest(
    channel_power: np.ndarray,
    parameters: Parameters,
) -> np.ndarray:

    num_bands, num_time_slots = channel_power.shape
    assert num_bands == parameters.rate.num_bands

    num_rejected = num_bands - parameters.rate.num_selected
    mask = np.zeros(channel_power.shape, dtype=bool)

    for n in range(num_rejected):
        min_idx = np.argmin(channel_power, axis=0)

        for t, idx in enumerate(min_idx):
            mask[idx, t] = True

    channel_power[mask] = np.nan
    return channel_power


