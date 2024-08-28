import os
import unittest
import numpy as np
import phast

class TestProblem(unittest.TestCase):
    def setUp(self):
        self.config = {
            "adap_ampl": 0.01,
            "acco_ampl": 0.0003,
            "adap_rate": 2.0,
            "acco_rate": 2.0,
            "arp": 0.4e-3,
            "rrp": 0.8e-3,
            "sigma_arp": 0.1e-3,
            "sigma_rrp": 0.5e-3,
            "rs": 0.06,
            "sigma_rs": 0.04,
        }
        self.pulse_train = np.load(os.path.join(phast.DATA_DIR, "pulse_train_121.npy"))
        self.i_det = np.load(os.path.join(phast.DATA_DIR, "I_det_121.npy"))

    def test_problem(self):
        stimulus = phast.PulseTrain(pulse_train=self.pulse_train, time_step=18e-6)
        fiber_idx = 0
        idet = self.i_det[fiber_idx, :]
        spatial_factor = np.nanmin(self.i_det, axis=0) / idet
        sigma = idet * self.config["rs"]

        refractoriness = phast.RefractoryPeriod(
            absolute_refractory_period=self.config["arp"],
            relative_refractory_period=self.config["rrp"],
            sigma_absolute_refractory_period=self.config["sigma_arp"],
            sigma_relative_refractory_period=self.config["sigma_rrp"],
        )

        decay = phast.LeakyIntegratorDecay(
            adaptation_amplitude=self.config["adap_ampl"],
            accommodation_amplitude=self.config["acco_ampl"],
            adaptation_rate=self.config["adap_rate"],
            accommodation_rate=self.config["acco_rate"],
        )

        fiber = phast.Fiber(
            i_det=idet,
            spatial_constant=spatial_factor,
            sigma=sigma,  # membrane sigma = I_det * RS
            fiber_id=fiber_idx,
            sigma_rs=self.config["sigma_rs"],
            refractory_period=refractoriness,  # object
            decay=decay, 
            store_stats=False,
        )
        fiber_stats = phast.phast([fiber], stimulus)
        spike_times = phast.spike_times(fiber_stats)
        self.assertEqual(len(spike_times), 0)
