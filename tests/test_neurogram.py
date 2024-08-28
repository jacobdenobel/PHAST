import os
import unittest
import itertools

import numpy as np
import matplotlib.pyplot as plt

import phast

IDET = np.load(os.path.join(phast.DATA_DIR, "idet.npy"))


class TestThreading(unittest.TestCase):
    def run_phast(self, decay):
        self.time_step = 1e-6
        self.duration = 1.0
        self.stimulus = phast.ConstantPulseTrain(
            self.duration, 5000, 1e-2, self.time_step
        )
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
        fiber_stats = phast.phast(self.fibers, self.stimulus, n_trials=10, n_jobs=-1)
        ng = phast.Neurogram(fiber_stats, 1e-3)
        return ng

    def test_neurogram(self):
        ngs = []
        axes = [None] * 3
        for ax, decay in zip(
            axes, (phast.Powerlaw(), phast.Exponential(), phast.LeakyIntegratorDecay(2, 2, 4, 4,))
        ):
            ng = self.run_phast(decay)
            ngs.append(ng.data)
        
        for c1, c2 in itertools.combinations(range(3), 2):
            p_dist = np.linalg.norm(ngs[c1] - ngs[c2])
            self.assertLessEqual(p_dist, 600.0)
        

if __name__ == "__main__":
    unittest.main()
