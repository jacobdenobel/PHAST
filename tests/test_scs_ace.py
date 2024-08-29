import os
import unittest
import numpy as np

from phast import SOUNDS 
from phast.scs import ace

import matplotlib.pyplot as plt

class TestAce(unittest.TestCase):
    
    # def test_amplitude_paramamters(self):
    #     amp = ace.AmplitudeParameters()
    #     self.assertAlmostEqual(amp.agc_kneepoint, 0.07933868577)
    #     self.assertAlmostEqual(amp.gain_dB, 36)
    #     self.assertAlmostEqual(amp.dynamic_range_dB, 40)
        
    # def test_rate_parameters(self):
    #     rate = ace.RateParameters()
    #     self.assertEqual(rate.audio_sample_rate_Hz, 15625)
    #     self.assertEqual(rate.channel_stim_rate_Hz, 976.5625)
    #     self.assertEqual(rate.analysis_rate_Hz, 976.5625)
    #     self.assertEqual(rate.block_shift, 16)
    #     self.assertEqual(rate.num_bands, 22)
    #     self.assertEqual(rate.num_selected, 12)
    #     self.assertEqual(rate.interval_length, 1)
    #     self.assertEqual(rate.implant_stim_rate_Hz, 11718.75)
    #     self.assertEqual(rate.period_us, 85.4)
        
    # def test_audio(self):
    #     wav_file = SOUNDS['asa']
    #     parameters = ace.Parameters()
    #     audio, (sr, audio_rms_dB, audio_dB_SPL, calibration_gain, ) = ace.process_audio(wav_file, parameters)
    #     self.assertEqual(sr, parameters.rate_parameters.audio_sample_rate_Hz)
    #     self.assertAlmostEqual(audio_rms_dB, -18.493992)
    #     self.assertEqual(audio_dB_SPL, 65.0)
    #     self.assertAlmostEqual(calibration_gain, 0.18801149)
        
    # def test_freedom_mic(self):
    #     wav_file = SOUNDS['asa']
    #     parameters = ace.Parameters()
    #     audio, _ = ace.process_audio(wav_file, parameters)
        
    #     res = ace.freedom_mic(audio, parameters)
    #     self.assertAlmostEqual(res.sum(), -0.11141796)
        
    # def test_agc(self):
    #     wav_file = SOUNDS['asa']
    #     parameters = ace.Parameters()
    #     signal, _ = ace.process_audio(wav_file, parameters)
    #     signal = ace.freedom_mic(signal, parameters)
    #     signal = ace.agc(signal, parameters)
    #     self.assertAlmostEqual(signal.sum(), -.2375905)
        
    def test_filterbank(self):
        wav_file = SOUNDS['asa']
        parameters = ace.Parameters()
        signal, _ = ace.process_audio(wav_file, parameters)
        signal = ace.freedom_mic(signal, parameters)
        signal = ace.agc(signal, parameters)
        spectrum = ace.filterbank(signal, parameters)        
        # self.assertAlmostEqual(np.abs(spectrum).sum(), 2735.721887637)
        
        channel_power = ace.power_sum_envelope(spectrum, parameters)
        # TODO: We still need default vector sum method
        
        channel_power = ace.gain(channel_power, parameters)
        channel_power = ace.resample(channel_power, parameters)
        channel_power = ace.reject_smallest(channel_power, parameters)
        channel_power = ace.lgf(channel_power, parameters)
        channels, magnitudes = ace.collate_into_sequence(channel_power, parameters)
        self.assertEqual(channels.size, magnitudes.size)
        breakpoint()

    
    
if __name__ == "__main__":
    unittest.main()