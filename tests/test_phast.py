import unittest
import numpy as np
import phast

class TestPhast(unittest.TestCase):
    def run_phast(self, no_random: bool, no_power_law: bool, parallel: bool = False):
        decay= phast.Exponential() if no_power_law else phast.Powerlaw()
        pt = phast.ConstantPulseTrain(0.4, 5000, 1e-3, 1e-6)

        fiber = phast.Fiber(
            i_det=[0.000774], 
            spatial_constant=[0.866593], 
            sigma=[0.000774 * 0.06], 
            fiber_id=1200,
            n_max=pt.n_pulses, 
            decay=decay
        )
        return phast.phast([fiber], pt, parallel, 10, not no_random)
        

    def test_no_random_exponential(self):
        fiber_stats = self.run_phast(True, False)
        self.assertSetEqual(set(f.n_spikes for f in fiber_stats), {48})
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats), {2000})

    def test_no_random_powerlaw(self):
        fiber_stats = self.run_phast(True, True)
        self.assertSetEqual(set(f.n_spikes for f in fiber_stats), {37})
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats), {2000})

    def test_random_powerlaw(self):
        phast.set_seed(42)
        fiber_stats1 = self.run_phast(False, True)
        phast.set_seed(42)
        fiber_stats2 = self.run_phast(False, True)
        self.assertListEqual(fiber_stats1, fiber_stats2)
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats1), {2000})
        self.assertGreaterEqual(sum(f.n_spikes for f in fiber_stats1), 717)
        self.assertNotEqual(len(set(f.n_spikes for f in fiber_stats1)), 1)

    def test_random_exponential(self):
        phast.set_seed(42)
        fiber_stats1 = self.run_phast(False, False)
        phast.set_seed(42)
        fiber_stats2 = self.run_phast(False, False)
        self.assertListEqual(fiber_stats1, fiber_stats2)
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats1), {2000})
        self.assertGreaterEqual(sum(f.n_spikes for f in fiber_stats1), 842)
        self.assertNotEqual(len(set(f.n_spikes for f in fiber_stats1)), 1)

    def test_random_powerlaw_parallel(self):
        phast.set_seed(42)
        fiber_stats1 = self.run_phast(False, True, True)
        phast.set_seed(42)
        fiber_stats2 = self.run_phast(False, True, True)
        self.assertListEqual(fiber_stats1, fiber_stats2)
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats1), {2000})
        self.assertGreaterEqual(sum(f.n_spikes for f in fiber_stats1), 717)
        self.assertNotEqual(len(set(f.n_spikes for f in fiber_stats1)), 1)

    def test_random_exponential_parallel(self):
        phast.set_seed(42)
        fiber_stats1 = self.run_phast(False, False, True)
        phast.set_seed(42)
        fiber_stats2 = self.run_phast(False, False, True)
        self.assertListEqual(fiber_stats1, fiber_stats2)
        self.assertSetEqual(set(f.n_pulses for f in fiber_stats1), {2000})
        self.assertGreaterEqual(sum(f.n_spikes for f in fiber_stats1), 851)
        self.assertNotEqual(len(set(f.n_spikes for f in fiber_stats1)), 1)

    def test_fiber_no_random(self):
        fiber = phast.Fiber(
            i_det=[0.000774], 
            spatial_constant=[0.866593], 
            sigma=[0.000774 * 0.06], 
            fiber_id=1200,
            n_max=10_000,
            sigma_rs=0.000,
        )
        fiber2 = fiber.randomize()

        self.assertTrue(np.all(np.array(fiber2.sigma) == fiber.sigma))

    def test_fiber_random(self):
        phast.set_seed(42)
        sigma = 1e-4

        fiber = phast.Fiber(
            i_det=[0.000774], 
            spatial_constant=[0.866593], 
            sigma=[0.000774 * 0.06], 
            fiber_id=1200,
            n_max=10_000,
            sigma_rs=sigma,
        )
        fiber2 = fiber.randomize()
        self.assertFalse(np.all(np.array(fiber2.sigma) == fiber.sigma))
 
        
    def test_rng(self):
        phast.set_seed(42)
        r1 = phast.GENERATOR()
        phast.set_seed(42)
        r2 = phast.GENERATOR()
        self.assertEqual(r1, r2)
        

if __name__ == "__main__":
    unittest.main()