import os 
import unittest

import numpy as np

import phast

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
IDET = np.load(os.path.join(phast.DATA_DIR,  "idet.npy"))


class TestThreading(unittest.TestCase):
    def setUp(self):
        self.stimulus = phast.ConstantPulseTrain(1, 5000, 1e-2, 1e-6)
        decay = phast.LeakyIntegratorDecay()
        self.fibers = []
        for fiber_idx in range(0, 3200, 100):
            self.fibers.append(
                phast.Fiber(
                    i_det=IDET[fiber_idx],
                    spatial_constant=[0.866593],
                    sigma=[0.000774 * 0.06],
                    fiber_id=fiber_idx,
                    decay=decay,  
                    store_stats=False,
                )
            )
            
    def test_thread_bleeding(self):
        fiber_stats = phast.phast(self.fibers, self.stimulus, n_trials = 10, n_jobs=-1)
        for fs in fiber_stats:
            self.assertEqual(fs.n_pulses, self.stimulus.n_pulses)
        
    def test_bleeding(self):
        fiber_stats = phast.phast(self.fibers, self.stimulus, n_trials = 10, n_jobs=1)
        for fs in fiber_stats:
            self.assertEqual(fs.n_pulses, self.stimulus.n_pulses)
        
if __name__ == "__main__":
    unittest.main()
