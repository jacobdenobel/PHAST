import unittest
import multiprocessing

import numpy as np

import phast


class TestTP(unittest.TestCase):
    def test_can_load_df120(self):
        for ft in phast.FiberType:
            tp = phast.load_df120(ft)
            self.assertTrue(tp.fiber_type == ft)
            self.assertEqual(tp.n_fibers, 3200)
            self.assertEqual(tp.electrode.n_channels, 135)
            self.assertEqual(tp.electrode.n_electrodes, 16)
            self.assertEqual(tp.electrode.pw, 18e-6)
            self.assertEqual(tp.electrode.cs_enabled, True)
            
    def test_from_idet(self):
        i_det = np.load(phast.I_DET)
        tp = phast.ThresholdProfile.from_idet(i_det)
        self.assertTrue(tp.fiber_type == phast.FiberType.HEALTHY)
        self.assertEqual(tp.n_fibers, 3200)
        self.assertEqual(tp.electrode.n_channels, 16)
        self.assertEqual(tp.electrode.n_electrodes, 16)
        self.assertEqual(tp.electrode.pw, 18e-6)
        self.assertEqual(tp.electrode.cs_enabled, False)
        
        
        
        
            
if __name__ == "__main__":
    unittest.main()