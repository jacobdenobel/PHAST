import unittest

import numpy as np
import librosa

import matplotlib.pyplot as plt
import phast
from phast.scs import ab as abt

TOTAL_INJECTED_CURRENT = 3.6952946437


class TestPackage(unittest.TestCase):
    def evaluate(
        self,
        expected_n_channels: tuple,
        expected_output: float = TOTAL_INJECTED_CURRENT,
        **kwargs,
    ):
        name = "tone_1kHz"
        pulse_train, audio_signal = abt.wav_to_electrodogram(
            phast.SOUNDS[name],
            ramp_type=None,
            stim_db=None,
            apply_audiomixer=True,
            **kwargs,
        )
        self.assertEqual(pulse_train.shape[0], expected_n_channels)
        self.assertEqual(pulse_train.shape[1], 111090)
        self.assertAlmostEqual(np.abs(pulse_train).sum(), expected_output)
        return pulse_train, audio_signal

    def test_end_to_end_cs(self):
        sound, sr = librosa.load(phast.SOUNDS["defineit"], sr=8000)
        tp = phast.load_df120()
        _, _, ng = phast.ab_e2e(
            tp=tp, audio_signal=sound, audio_fs=sr, current_steering=True
        )
        self.assertEqual(ng.data.shape[0], 3200)
        self.assertEqual(ng.data.shape[1], 24764)

    def test_end_to_end_no_cs(self):
        tp = phast.load_df120()
        _, _, ng = phast.ab_e2e(
            phast.SOUNDS["defineit"], tp=tp, current_steering=False
        )
        self.assertEqual(ng.data.shape[0], 3200)
        self.assertEqual(ng.data.shape[1], 24764)

    def test_wav_to_electrodogram_current_steering_no_virtual(self):
        self.evaluate(
            16, current_steering=True, virtual_channels=False, charge_balanced=True
        )

    def test_wav_to_electrodogram_no_current_steering_no_virtual(self):
        pulse_train, audio_signal = self.evaluate(
            16, current_steering=False, virtual_channels=False, charge_balanced=True
        )
        self.assertTrue((np.abs(pulse_train).sum(axis=1) > 0).all())

    def test_virtual_current_steering(self):
        self.evaluate(
            135, current_steering=True, virtual_channels=True, charge_balanced=True
        )

    def test_virtual_no_current_steering(self):
        self.evaluate(
            15, current_steering=False, virtual_channels=True, charge_balanced=True
        )

    def test_wav_to_electrodogram_not_balanced(self):
        self.evaluate(
            16,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=True,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            16,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=False,
            virtual_channels=False,
            charge_balanced=False,
        )
        self.evaluate(
            135,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=True,
            virtual_channels=True,
            charge_balanced=False,
        )
        self.evaluate(
            15,
            TOTAL_INJECTED_CURRENT / 2,
            current_steering=False,
            virtual_channels=True,
            charge_balanced=False,
        )

    def test_ordering_virtual_channels(self):
        for cs in (False, True):
            n_channels = 135 if cs else 15
            pulse_train, _ = self.evaluate(
                n_channels,
                TOTAL_INJECTED_CURRENT / 2,
                current_steering=cs,
                virtual_channels=True,
                charge_balanced=False,
            )

            channel_freq = abt.virtual_channel_frequencies(n_channels)
            max_freq = channel_freq[pulse_train.sum(axis=1).argmax()]
            closest_to_1kHz = find_nearest(channel_freq, 1000)
            self.assertEqual(max_freq, closest_to_1kHz)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    idx = find_nearest_idx(array, value)
    return array[idx]


if __name__ == "__main__":
    unittest.main()
