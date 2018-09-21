#!/usr/bin/env python3
""" 
Test functions of confidence_calibration.py
"""

from confidence_calibration import TemperatureScaling
from math import log
import numpy as np
import numpy.testing as npt
import unittest

class TestTemperatureScaling(unittest.TestCase):

    def testNoECELoss(self):
        # Test with 4 bins (of which, the lower 2 will be empty), 5 examples each.
        # Boundaries: 0, 0.25, 0.5, 0.75, 1.0
        # (Conf, Acc) of 3rd bin = (0.6, 0.6), 4th bin = (0.8, 0.8)
        # Prepare inverse of softmax using log(), so that we get desired softmax values
        logits = np.array([
            # 3rd bin (3 out of 5 are correct):
            [log(0.35), log(0.65)],
            [log(0.4), log(0.6)],
            [log(0.6), log(0.4)],
            [log(0.6), log(0.4)],
            [log(0.55), log(0.45)],
            # 4th bin (4 out of 5 are correct):
            [log(0.16), log(0.84)],
            [log(0.2), log(0.8)],
            [log(0.8), log(0.2)],
            [log(0.8), log(0.2)],
            [log(0.76), log(0.24)]])
        labels = np.array([
            # In the 3rd bin, we want 3 correct:
            [0, 1], # Correct
            [0, 1], # Correct
            [0, 1],
            [1, 0], # Correct
            [0, 1],
            # In the 4th bin, we want 4 correct:
            [0, 1], # Correct
            [0, 1], # Correct
            [0, 1],
            [1, 0], # Correct
            [1, 0]]) # Correct

        temp_scaling = TemperatureScaling(4)
        self.assertAlmostEqual(0, temp_scaling.getECELoss(logits, labels)[0])


    def testTemperature(self):
        # Test with 4 bins (of which, the lower 2 will be empty), 5 examples each.
        # Boundaries: 0, 0.25, 0.5, 0.75, 1.0
        # Confidence numbers are over-estimates:
        # (Conf, Acc) of 3rd bin = (>0.6, 0.6), 4th bin = (>0.8, 0.8)
        # Prepare inverse of softmax using log(), so that we get desired softmax values
        logits = np.array([
            # 3rd bin (3 out of 5 are correct):
            [log(0.35), log(0.65)],
            [log(0.3), log(0.7)],
            [log(0.7), log(0.3)],
            [log(0.7), log(0.3)],
            [log(0.55), log(0.45)],
            # 4th bin (4 out of 5 are correct):
            [log(0.06), log(0.94)],
            [log(0.1), log(0.9)],
            [log(0.9), log(0.1)],
            [log(0.9), log(0.1)],
            [log(0.96), log(0.04)]])
        labels = np.array([
            # In the 3rd bin, we want 3 correct:
            [0, 1], # Correct
            [0, 1], # Correct
            [0, 1],
            [1, 0], # Correct
            [0, 1],
            # In the 4th bin, we want 4 correct:
            [0, 1], # Correct
            [0, 1], # Correct
            [0, 1],
            [1, 0], # Correct
            [1, 0]]) # Correct

        temp_scaling = TemperatureScaling(4)
        old_loss = temp_scaling.getECELoss(logits, labels)[0]
        temperature = temp_scaling.calculateTemperature(logits, labels)
        self.assertGreater(temperature, 1.0)
        new_loss = temp_scaling.getECELoss(logits, labels, temperature)[0]
        self.assertGreater(old_loss, new_loss)


if __name__ == '__main__':
    unittest.main()
