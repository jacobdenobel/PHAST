import os
from dataclasses import dataclass

import librosa
import scipy
import numpy as np


from .parameters import Parameters, to_dB, from_dB
from ...constants import DATA_DIR


FAR_DATA = os.path.join(DATA_DIR, "freedom_avg_response_data.npy")


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(x, 2)))


def process_audio(
    wav_file: str, audio_signal: np.ndarray, audio_fs: int, parameters: Parameters
) -> np.ndarray:
    if wav_file is not None:
        signal, sr = librosa.load(wav_file, sr=parameters.audio_sample_rate_Hz)
    else:
        if audio_fs is None:
            raise ValueError("audio_fs must be provided")
        if audio_signal is None:
            raise ValueError("audio_signal must be provided")

        sr = parameters.audio_sample_rate_Hz
        signal = librosa.resample(audio_signal, orig_sr=audio_fs, target_sr=sr)

    # Sound pressure level is based on RMS value:
    audio_rms_dB = to_dB(rms(signal))

    # A full scale +-1 reference pure tone has RMS level of -3 dB relative to FS.
    # Therefore, the unscaled audio corresponds to the following sound pressure level:
    audio_dB_SPL = audio_rms_dB + parameters.reference_dB_SPL + to_dB(np.sqrt(2))

    # Calculate the calibration gain that will produce the desired sound pressure level:
    calibration_gain_dB = parameters.audio_dB_SPL - audio_dB_SPL

    # Apply gain:
    calibration_gain = from_dB(calibration_gain_dB)
    audio_dB_SPL = audio_dB_SPL + calibration_gain_dB
    return signal * calibration_gain, (
        sr,
        audio_rms_dB,
        audio_dB_SPL,
        calibration_gain,
    )


def freedom_mic(
    signal: np.ndarray,
    parameters: Parameters,
) -> np.ndarray:
    calibration_freq_Hz = parameters.audio_sample_rate_Hz / 16
    data = np.load(FAR_DATA, allow_pickle=True).item()

    mag = data[f"{parameters.directivity}_mag"]

    mag = mag - np.max(mag)
    amplitude = from_dB(mag)

    # Zero any magnitudes more than 50 dB below max:
    amplitude[mag < -50] = 0

    # Normalise the frequencies to the Nyquist frequency (fs/2).
    norm_freq = data["freq"] / (parameters.audio_sample_rate_Hz / 2)

    # Remove any frequencies which are greater than Nyquist
    # & Append Nyquist response (needed for fir2):
    mask = norm_freq < 1
    norm_freq = np.r_[norm_freq[mask], 1]
    amplitude = np.r_[amplitude[mask], 0]
    # Prepend DC response (needed for fir2):
    if norm_freq[0] > 0:
        norm_freq = np.r_[0, norm_freq]
        amplitude = np.r_[0, amplitude]

    # Create filter using frequency sampling:
    mic_numer = scipy.signal.firwin2(parameters.mic_order + 1, norm_freq, amplitude)
    mic_denom = 1
    # Calculate gain at calibration frequency.
    w, h = scipy.signal.freqz(
        mic_numer,
        mic_denom,
        [calibration_freq_Hz, calibration_freq_Hz * 2],
        fs=parameters.audio_sample_rate_Hz,
    )
    # Add gain so that response at calibration_freq_Hz is 0 dB:
    mic_numer = mic_numer / abs(h[0])

    # this is different from matlab in res
    return scipy.signal.lfilter(mic_numer, mic_denom, signal)
