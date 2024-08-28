import unittest
import multiprocessing

import numpy as np

import phast


class TestPhast(unittest.TestCase):
    def run_phast(
        self,
        decay: phast.Decay,
        random: bool = True,
        parallel: bool = False,
        store_stats: bool = False,
        n_trials: int = 10,
        duration: float = 0.4,
    ):
        pt = phast.ConstantPulseTrain(duration, 5000, 1e-3, 1e-6)
        phast.set_seed(42)
        fiber = phast.Fiber(
            i_det=[0.000774],
            spatial_constant=[0.866593],
            sigma=[0.000774 * 0.06],
            fiber_id=1200,
            decay=decay,
            store_stats=store_stats,
        )
        return phast.phast([fiber], pt, -1 if parallel else 1, n_trials, random)

    def compare_random(self, decay, n_total_spikes, parallel):
        fiber_stats1 = self.run_phast(decay, True, parallel)
        fiber_stats2 = self.run_phast(decay, True, parallel)
        self.assertListEqual(fiber_stats1, fiber_stats2)
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats1), {2000})
        self.assertGreaterEqual(sum(f.n_spikes for f in fiber_stats1), n_total_spikes)
        self.assertNotEqual(len(set(f.n_spikes for f in fiber_stats1)), 1)

    def test_no_random_powerlaw(self):
        decay = phast.Powerlaw()
        fiber_stats = self.run_phast(decay, False, False)
        self.assertSetEqual(set(f.n_spikes for f in fiber_stats), {48})
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats), {2000})

    def test_no_random_exponential(self):
        decay = phast.Exponential()
        fiber_stats = self.run_phast(decay, False, False)
        self.assertSetEqual(set(f.n_spikes for f in fiber_stats), {37})
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats), {2000})

    def test_random_exponential(self):
        self.compare_random(phast.Exponential(), 717, False)

    def test_random_powerlaw(self):
        self.compare_random(phast.Powerlaw(), 842, False)

    def test_random_exponential_parallel(self):
        self.compare_random(phast.Exponential(), 717, True)

    def test_random_powerlaw_parallel(self):
        self.compare_random(phast.Powerlaw(), 842, True)

    def test_fiber_no_random(self):
        fiber = phast.Fiber(
            i_det=[0.000774],
            spatial_constant=[0.866593],
            sigma=[0.000774 * 0.06],
            fiber_id=1200,
            sigma_rs=0.000,
        )
        fiber2 = fiber.randomize(1)

        self.assertTrue(np.all(np.array(fiber2.sigma) == fiber.sigma))

    def test_fiber_random(self):
        phast.set_seed(42)
        sigma = 1e-4

        fiber = phast.Fiber(
            i_det=[0.000774],
            spatial_constant=[0.866593],
            sigma=[0.000774 * 0.06],
            fiber_id=1200,
            sigma_rs=sigma,
        )
        fiber2 = fiber.randomize(1)
        self.assertFalse(np.all(np.array(fiber2.sigma) == fiber.sigma))

    def test_rng(self):
        phast.set_seed(42)
        r1 = phast.GENERATOR()
        phast.set_seed(42)
        r2 = phast.GENERATOR()
        self.assertEqual(r1, r2)

    def test_leaky(self):
        decay = phast.LeakyIntegratorDecay(2, 2, 2, 2)
        fiber_stats = self.run_phast(decay, False)
        self.assertSetEqual(set(f.n_spikes for f in fiber_stats), {60})
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats), {2000})

        self.compare_random(decay, 810, True)
        self.compare_random(decay, 810, False)

    def test_store_stats(self):
        decay = phast.LeakyIntegratorDecay(2, 2, 2, 2)
        fiber_stats, *_ = self.run_phast(decay, store_stats=True, n_trials=1)
        stat_names = (
            "accommodation",
            "adaptation",
            "pulse_times",
            "refractoriness",
            "scaled_i_given",
            "stochastic_threshold",
        )
        for name in stat_names:
            self.assertEqual(len(getattr(fiber_stats, name)), 2000)

    def test_dont_store_stats(self):
        decay = phast.LeakyIntegratorDecay(2, 2, 2, 2)
        fiber_stats, *_ = self.run_phast(decay, store_stats=False, n_trials=1)
        stat_names = (
            "accommodation",
            "adaptation",
            "pulse_times",
            "refractoriness",
            "scaled_i_given",
            "stochastic_threshold",
        )
        for name in stat_names:
            self.assertLessEqual(len(getattr(fiber_stats, name)), 1)

    def test_large_pt(self):
        duration = 100.0
        n_trials = multiprocessing.cpu_count()
        decay = phast.LeakyIntegratorDecay()
        fiber_stats = self.run_phast(
            decay, store_stats=False, parallel=True, n_trials=n_trials, duration=duration
        )
        spike_times = phast.spike_times(fiber_stats)
        spike_rate = phast.spike_rate(
            spike_times, num_bins=50, duration=duration, n_trials=n_trials
        )
        self.assertEqual(len(spike_rate), 50)

if __name__ == "__main__":
    unittest.main()
