import abc
from typing import Any

import numpy as np


class AudioRamp(abc.ABC):
    def __init__(self, Fs: float, ramp_duration: float) -> None:
        self.Fs = Fs
        self.ramp_duration = ramp_duration

    @abc.abstractmethod
    def __call__(self, audio_signal: np.ndarray, **kwds: Any) -> Any:
        "Input of time domain filters must be a vector"


class LinearRamp(AudioRamp):
    def __call__(self, audio_signal: np.ndarray, **kwds: Any) -> Any:
        if len(audio_signal.shape) > 1:
            audio_signal = np.squeeze(audio_signal)

        ramp_length = int(np.ceil(self.ramp_duration * self.Fs))
        ramp_vector = np.linspace(0, self.ramp_duration, ramp_length)
        ramped_audio_signal = audio_signal.copy()

        for ii in np.arange(ramp_length):
            ramped_audio_signal[ii] = (
                audio_signal[ii] * ramp_vector[ii] / ramp_vector.max()
            )
            ramped_audio_signal[-1 - ii] = (
                audio_signal[-1 - ii] * ramp_vector[ii] / ramp_vector.max()
            )

        return np.expand_dims(ramped_audio_signal, axis=0)


class CosineRamp(AudioRamp):
    def __call__(self, audio_signal: np.ndarray, **kwds: Any) -> Any:
        if len(audio_signal.shape) > 1:
            audio_signal = np.squeeze(audio_signal)
        ramp_length = int(np.ceil(self.ramp_duration * self.Fs))
        ramp_vector = np.linspace(0, np.pi, ramp_length)

        ramped_audio_signal = audio_signal.copy()

        ramp_up = -0.5 * np.cos(ramp_vector) + 0.5
        ramped_audio_signal[:ramp_length] = ramped_audio_signal[:ramp_length] * ramp_up
        ramped_audio_signal[-ramp_length:] *= ramp_up[::-1]
        ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)
        return ramped_audio_signal


class CosineSquaredRamp(AudioRamp):
    def __call__(self, audio_signal: np.ndarray, **kwds: Any) -> Any:
        if len(audio_signal.shape) > 1:
            audio_signal = np.squeeze(audio_signal)
        ramp_length = int(np.ceil(self.ramp_duration * self.Fs))
        ramp_vector = np.linspace(0, np.pi, ramp_length)

        ramped_audio_signal = audio_signal.copy()
        ramp_up = -0.5 * np.cos(ramp_vector) + 0.5
        ramped_audio_signal[:ramp_length] *= ramp_up
        ramped_audio_signal[-ramp_length:] *= ramp_up[::-1]
        ramped_audio_signal = np.expand_dims(ramped_audio_signal, axis=0)

        return ramped_audio_signal


def apply_ramp_type(
    audio_signal, ramp_type="cs", Fs=17400, ramp_duration=0.01, **kwargs
):
    if ramp_type is None:
        return audio_signal

    if ramp_type == "linear" or ramp_type == "l":
        ramped_audio_function = LinearRamp(ramp_duration=ramp_duration, Fs=Fs)
    elif ramp_type == "cosine" or ramp_type == "c":
        ramped_audio_function = CosineRamp(ramp_duration=ramp_duration, Fs=Fs)
    elif ramp_type == "cosine_squared" or ramp_type == "cs":
        ramped_audio_function = CosineSquaredRamp(ramp_duration=ramp_duration, Fs=Fs)
    else:
        ValueError(
            "This ramp type is not an option. Choose from: linear, cosine, cosine_squared"
        )

    return ramped_audio_function(audio_signal)
