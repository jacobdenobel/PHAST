import unittest
import numpy as np
import phast


class TestRandomRandy(unittest.TestCase):
    def test_seeding(self):
        # Leaky integrator decay

        idet = np.ones((1,), dtype=float) * 0.001

        fiber = phast.Fiber(
            i_det=idet,
            spatial_constant=([1]),
            sigma=idet * 0.06,  # membrane sigma = I_det * RS
            fiber_id=0,
            sigma_rs=0.04,
            store_stats=True,
        )
        Istim = 0.001
        Ntrials = 200
        SeedNumArray = [2477]  # , 2477, 5114, 83544]

        # Define a single pulse with amplitude=Istim
        pulse_train = np.ones(shape=(1, 1), dtype=float) * Istim
        stimulus = phast.PulseTrain(pulse_train=pulse_train.tolist(), time_step=1.0e-6)

        for SeedNum in SeedNumArray:

            # Calculate Ntrials in one call to phast
            phast.set_seed(SeedNum)
            fiber_stats = phast.phast([fiber], stimulus, n_trials=Ntrials)
            spike_times = phast.spike_times(fiber_stats)

            Na = len(spike_times)

            # Calculate Ntrials using seperate calls to phast
            phast.set_seed(SeedNum)
            Nb = 0

            for i in range(0, Ntrials):
                # phast.set_seed(SeedNum + i)
                fiber_stats = phast.phast([fiber], stimulus, n_trials=1)

                Nb += fiber_stats[0].n_spikes

            self.assertEqual(Nb, 100)
            self.assertEqual(Na, 108)
