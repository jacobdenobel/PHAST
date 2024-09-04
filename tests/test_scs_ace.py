import os
import unittest

import librosa
import numpy as np


from phast import SOUNDS, load_cochlear, load_df120, ace, ace_e2e


class TestAce(unittest.TestCase):
    def test_e2e(self):
        wav_file = SOUNDS["asa"]
        (audio, audio_fs), pulse_train, ng = ace_e2e(wav_file, some_bull=True)

        la, sr = librosa.load(wav_file)
        delta_d = abs(la.size * (1 / sr) - audio.size * (1 / audio_fs))
        self.assertLess(delta_d, 1e-4)
        self.assertEqual(pulse_train.shape, (22, 11604))
        self.assertEqual(ng.data.shape, (3200, 27525))

        (audio, audio_fs), pulse_train, ng = ace_e2e(audio_signal=la, audio_fs=sr)
        delta_d = abs(la.size * (1 / sr) - audio.size * (1 / audio_fs))
        self.assertLess(delta_d, 1e-4)
        self.assertEqual(pulse_train.shape, (22, 11604))
        self.assertEqual(ng.data.shape, (3200, 27525))

    def test_parameters(self):
        p = ace.Parameters()
        self.assertAlmostEqual(p.agc_kneepoint, 0.07933868577,  places=5)
        self.assertAlmostEqual(p.gain_dB, 36, places=5)
        self.assertAlmostEqual(p.dynamic_range_dB, 40, places=5)

        self.assertEqual(p.audio_sample_rate_Hz, 15625)
        self.assertEqual(p.channel_stim_rate_Hz, 976.5625)
        self.assertEqual(p.analysis_rate_Hz, 976.5625)
        self.assertEqual(p.block_shift, 16)
        self.assertEqual(p.num_bands, 22)
        self.assertEqual(p.num_selected, 12)
        self.assertEqual(p.interval_length, 1)
        self.assertEqual(p.implant_stim_rate_Hz, 11718.75)
        self.assertEqual(p.period_us, 85.4)

    def test_audio(self):
        wav_file = SOUNDS["asa"]
        parameters = ace.Parameters()
        audio, (
            sr,
            audio_rms_dB,
            audio_dB_SPL,
            calibration_gain,
        ) = ace.process_audio(wav_file, None, None, parameters)
        self.assertEqual(sr, parameters.audio_sample_rate_Hz)
        self.assertAlmostEqual(audio_rms_dB, -18.493992, places=5)
        self.assertEqual(audio_dB_SPL, 65.0)
        self.assertAlmostEqual(calibration_gain, 0.18801149, places=5)

    def test_freedom_mic(self):
        wav_file = SOUNDS["asa"]
        parameters = ace.Parameters()
        audio, _ = ace.process_audio(wav_file, None, None, parameters)

        res = ace.freedom_mic(audio, parameters)
        self.assertAlmostEqual(res.sum(), -0.11141796, places=5)

    def test_agc(self):
        wav_file = SOUNDS["asa"]
        parameters = ace.Parameters()
        signal, _ = ace.process_audio(wav_file, None, None, parameters)
        signal = ace.freedom_mic(signal, parameters)
        signal = ace.agc(signal, parameters)
        self.assertAlmostEqual(signal.sum(), -0.2375905, places=5)

    def test_ace(self):
        wav_file = SOUNDS["asa"]
        parameters = ace.Parameters()
        signal, _ = ace.process_audio(wav_file, None, None, parameters)
        signal = ace.freedom_mic(signal, parameters)
        signal = ace.agc(signal, parameters)
        spectrum = ace.filterbank(signal, parameters)
        self.assertAlmostEqual(np.abs(spectrum).sum(), 2746.54787571626, places=4)

        channel_power = ace.envelope_method(spectrum, parameters)
        channel_power = ace.gain(channel_power, parameters)
        channel_power = ace.resample(channel_power, parameters)
        channel_power = ace.reject_smallest(channel_power, parameters)
        self.assertEqual(np.isfinite(channel_power).sum(), 11604)

        channel_power = ace.lgf(channel_power, parameters)
        channels, magnitudes = ace.collate_into_sequence(channel_power, parameters)
        self.assertEqual(channels.size, magnitudes.size)
        self.assertEqual(channels.size, 11604)

        electrode_seq = ace.channel_mapping(channels, magnitudes, parameters)
        pulse_train = np.flip(electrode_seq.to_pulse_table(), axis=0)
        
        self.assertEqual(pulse_train.shape, (22, 11604))
        self.assertEqual(pulse_train.max(), 0.01)
        self.assertEqual(pulse_train.min(), 0.0)
        self.assertAlmostEqual(pulse_train.sum(), 61.7435,  places=5)

        pulse_train2, parameters2, _ = ace.ace(SOUNDS["asa"])
        self.assertTrue((pulse_train == pulse_train2).all())


if __name__ == "__main__":
    unittest.main()
