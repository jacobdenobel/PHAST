import unittest
import multiprocessing

import numpy as np

import phast


class TestSpontActivity(unittest.TestCase):
    def run_phast(
        self,
        decay: phast.Decay,
        random: bool = True,
        parallel: bool = False,
        store_stats: bool = False,
        n_trials: int = 1,
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
    
    def test_spont(self):
        decay = phast.LeakyIntegratorDecay(2, 2, 2, 2)
        fib_stats = self.run_phast(decay)
        # breakpoint()

if __name__ == "__main__":
    unittest.main()