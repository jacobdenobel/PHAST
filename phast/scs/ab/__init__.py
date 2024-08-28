import os

import numpy as np
from scipy.signal import lfilter

from . import frontend
from . import buffer
from . import automatic_gain_control
from . import filterbank
from . import noise_reduction
from . import post_filterbank
from . import mapping
from . import electrodogram
from . import vocoder
from . import audiomixer
from . import audio_ramp
from .defaults import virtual_channel_frequencies


def wav_to_electrodogram(
    wav_file: str = None,
    audio_signal: np.ndarray = None,
    audio_fs: int = None,
    apply_audiomixer=False,
    virtual_channels=True,
    charge_balanced=False,
    n_rep: int = 1,
    **kwargs,
):
    """
    Parameters
    ----------
        wav_file: str
            path to a .wav file
        audio_signal:  np.ndarray
            sound wave
        audio_fs: int
            input sample rate of audio signal
        apply_audiomixer: bool = False
            Whether to apply the audio mixer routine
        virtual_channels: bool = True
            Whether to output virtual channels, default AB is false
        charge_balanced: bool = False
            Whether to make the final pulse train charge balanced
        **kwargs: dict
            any keyword argument for any of the subroutines used in this function

    Returns
    -------
        np.ndarray, np.ndarray
            pulse_train, audio_signal

    """

    if wav_file is not None:
        audio_signal, *_ = frontend.read_wav(wav_file, **kwargs)
    else:
        audio_signal = frontend.process_stim(audio_signal, audio_fs, **kwargs)

    audio_signal = np.tile(audio_signal, n_rep)

    if apply_audiomixer:
        audio_signal, *_ = audiomixer.audiomixer(audio_signal, **kwargs)

    # add ramp to the stimulus to avoid any un-smoothness/discontinuity at onset
    audio_signal = audio_ramp.apply_ramp_type(audio_signal, **kwargs)

    # Applies an IIR (actually highpass filter) to the audio signal, but this works as a pre-emphasis filter:
    # (1) balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies,
    # (2) avoid numerical problems during the Fourier transform operation and
    # (3) may also improve the Signal-to-Noise Ratio (SNR). See https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    # Another way to apply pre-emphasis:
    # pre_emphasis = 0.97  # will become a hyperparameter for Jacob
    # emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    signal = frontend.td_filter(audio_signal, **kwargs)
    # AGC to reduce signals with too much strength, how can we alter this so that it can be optimized? Moreover, there is also another FIR
    # in this application?
    # dual loop for fast and slow acting part
    signal, agc = automatic_gain_control.dual_loop_td_agc(signal, **kwargs)
    # windowing to reduce spectral leakage
    signal = buffer.window_buffer(signal, **kwargs)
    # Makes the audio input of lenght 256, so fft over 0.0147 seconds. FFT length gives good compromise between spectral and temporal resolution
    signal_fft = filterbank.fft_filterbank(signal, **kwargs)
    # calculate the envelope per frequency band of each channel?
    signal_hilbert = filterbank.hilbert_envelope(signal_fft, **kwargs)
    # calculate amplitude just by taking the square root of the power per channel? Why first squaring + absolute and then square root?
    signal_energy = filterbank.channel_energy(signal_fft, agc.smpGain, **kwargs)
    # Compute channel-by-channel noise reduction gains.
    signal_energy, *_ = noise_reduction.noise_reduction(signal_energy, **kwargs)
    # why would you sum the envelope with the energy?
    signal = signal_hilbert + signal_energy
    # Find frequency band with largest amplitude of subsample (every third FFT input frame)
    peak_freq, peak_loc = post_filterbank.spec_peak_locator(
        signal_fft[:, 2::3], **kwargs
    )
    # upsample back to full framerate and add padding
    peak_freq = post_filterbank.upsample(peak_freq, signal_fft.shape[1], **kwargs)
    peak_loc = post_filterbank.upsample(peak_loc, signal_fft.shape[1], **kwargs)

    # # CS provides 8 "virtual" electrodes so 8*15 = 120 "virtual" electrodes in total
    weights = post_filterbank.current_steering_weights(peak_loc, **kwargs)

    # Create carrier function with period of 1/peak_freq, maximum depends on implant's maximal stimulation rate
    carrier, audio_idx = post_filterbank.carrier_synthesis(peak_freq, **kwargs)

    signal = mapping.f120(carrier, signal, weights, audio_idx, **kwargs)

    pulse_train = electrodogram.f120(
        signal,
        weights=weights[:, audio_idx],
        virtual_channels=virtual_channels,
        charge_balanced=charge_balanced,
        **kwargs,
    )

    return pulse_train, audio_signal
