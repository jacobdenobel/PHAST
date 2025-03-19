import os
import unittest

import numpy as np
import librosa

import phast
from phast.scs import ab as abt



def select_fibers(fiber_freq, mel_scale):
    grouped = np.digitize(fiber_freq, mel_scale, True)
    selected_fibers = []
    for fbin, nf in zip(*np.unique(grouped, return_counts=True)):
        selected_fibers.extend(
            sorted(
                np.random.choice(
                    np.where(grouped == fbin)[0], min(10, nf), replace=False
                )
            )
        )
    return np.array(selected_fibers)



def get_fiber_freq_specres(tp, max_freq, power=25):
    channel_freq = phast.scs.ab.defaults.virtual_channel_frequencies(
        tp.i_det.shape[1], max_freq + 500
    )
    w = (-tp.i_det / tp.i_det.sum(axis=1).reshape(-1, 1)) + (2 / len(channel_freq))
    w = np.power(w, power) / np.power(w, power).sum(axis=1).reshape(-1, 1)
    fiber_freq = w @ channel_freq
    return fiber_freq


def generate_specres(path):
    n_trials = 20
    cs = True
    min_freq = 450
    max_freq = 5500
    n_mels = 64
    mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)

    tp = phast.load_df120()
    fiber_freq = get_fiber_freq_specres(tp, max_freq)
    selected_fibers = select_fibers(fiber_freq, mel_scale)
    fiber_freq = fiber_freq[selected_fibers]

    audio_signal, audio_fs = phast.scs.ab.frontend.read_wav(path, stim_db=65.0)
    audio_signal += np.random.normal(0, 1e-20, size=len(audio_signal))

    (audio_signal, FS), pulse_train, neurogram = phast.ab_e2e(
        audio_signal=audio_signal,
        audio_fs=audio_fs,
        tp=tp,
        current_steering=cs,
        scaling_factor=1.4,
        ramp_duration=(audio_signal.size / audio_fs) * 0.05,
        n_trials=n_trials,
        accommodation_amplitude=0.07,
        adaptation_amplitude=7.142,
        accommodation_rate=2,
        adaptation_rate=19.996,
        selected_fibers=selected_fibers,
        spont_activity = 50.0,
    )
    audio_signal = audio_signal[0]
    neurogram_data = neurogram.data / n_trials
    ret_data = neurogram_data, neurogram.binsize, min_freq, max_freq, n_mels
    
    return ret_data


class TesDin(unittest.TestCase):
    def test_din_846(self):
        path = os.path.join(os.path.dirname(__file__), "846.wav")
        generate_specres(path)
        self.assertTrue(True)

    
if __name__ == "__main__":
    unittest.main()