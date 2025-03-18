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
        store_stats: bool = True,
        n_trials: int = 1,
        duration: float = 0.4,
        spont_activity: float = 150
    ):
        pt = phast.ConstantPulseTrain(duration, 5000, 1e-12, 1e-6)
        phast.set_seed(42)
        fiber = phast.Fiber(
            i_det=[0.000774],
            spatial_constant=[0.866593],
            sigma=[0.000774 * 0.06],
            fiber_id=1200,
            decay=decay,
            store_stats=store_stats,
            spont_activity=spont_activity 
        )
        
        fs = phast.phast([fiber], pt, -1 if parallel else 1, n_trials, random)
        return fs
    
    def test_spont(self):
        decay = phast.LeakyIntegratorDecay(2, 2, 2, 2)
        duration = 1.0
        n_trials = 50
        for sa in (0, 50, 100, 150, 200, 500, 1000):
            fs = self.run_phast(decay, duration=duration, n_trials=n_trials, spont_activity=sa)
            spike_times = phast.spike_times(fs)
            spike_rate = phast.spike_rate(spike_times, duration=duration, binsize=0.01, n_trials=n_trials)
            self.assertLessEqual(abs(sa - np.median(spike_rate)), sa * .05)
        
if __name__ == "__main__":  
    unittest.main()