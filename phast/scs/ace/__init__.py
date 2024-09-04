import numpy as np

from .parameters import Parameters
from .audio import process_audio, freedom_mic
from .agc import agc
from .filterbank import filterbank, power_sum_envelope, envelope_method
from .utility import gain, resample, reject_smallest
from .lgf import lgf
from .mapping import collate_into_sequence, channel_mapping


def ace(
    wav_file: str = None,
    audio_signal: np.ndarray = None,
    audio_fs: int = None,
    parameters: Parameters = None,
    **kwargs
):
    if parameters is None:
        parameters = Parameters(**kwargs)

    audio_signal, *_ = process_audio(wav_file, audio_signal, audio_fs, parameters)
    signal = freedom_mic(audio_signal, parameters)
    signal = agc(signal, parameters)
    spectrum = filterbank(signal, parameters)

    channel_power = envelope_method(spectrum, parameters)
    channel_power = gain(channel_power, parameters)
    channel_power = resample(channel_power, parameters)
    channel_power = reject_smallest(channel_power, parameters)
    channel_power = lgf(channel_power, parameters)
    channels, magnitudes = collate_into_sequence(channel_power, parameters)
    electrode_seq = channel_mapping(channels, magnitudes, parameters)
    pulse_train = electrode_seq.to_pulse_table()
    pulse_train = np.flip(pulse_train, axis=0)
    return pulse_train, parameters, audio_signal
