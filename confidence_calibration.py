"""
Modern neural nets produce softmax confidence numbers that are too optimistic.
This module calibrates the softmax output properly.
"""

import numpy as np
from scipy import optimize
import sys

class TemperatureScaling():
    """
    Based on this paper: https://arxiv.org/pdf/1706.04599.pdf
    Reference code for Torch is here:
    https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    def __init__(self, n_bins=30):
        """
        n_bins (int): number of confidence interval bins.
        Note that for binary classification, the lowest confidence possible is 0.5,
        so the bottom half bins will be empty.
        """
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]

    # Implement our own so we don't have to use tf's graph-based function.
    def _softmax(self, x_2d):
        """Compute softmax values for each sets of scores in x."""
        sm = np.zeros(x_2d.shape)
        for i in range(x_2d.shape[0]):
            x = x_2d[i]
            e_x = np.exp(x - np.max(x))
            sm[i] = e_x / e_x.sum()
        return sm


    def getECELoss(self, logits, labels, temperature=1.0):
        """
        Calculates the Expected Calibration Error of a model.
        The input to this loss is the logits of a model, NOT the softmax scores.
        This divides the confidence outputs into equally-sized interval bins.
        In each bin, we compute the confidence gap:
        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
        We then return a weighted average of the gaps, based on the number
        of samples in each bin
        See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. 2015.
        labels must be in one-hot form.
        Shapes of both logits and labels must be the same: num_examples * num_classes
        Returns ECE loss as well as info about the bins (that can be used for plots etc).
        """
        if not np.array_equal(logits.shape, labels.shape):
            print('logits/labels shapes are not equal!')
            sys.exit(1)

        softmaxes = self._softmax(logits/temperature)
        confidences = softmaxes.max(axis=1)  # Highest confidence value for each example
        predictions = np.argmax(softmaxes, axis=1)  # Class with highest confidence for each example
        # np.equal checks element-wise
        accuracies = np.equal(np.argmax(labels, axis=1), predictions).astype(int)

        ece = 0.0
        bin_accuracies = []
        weights_of_bins = []
        bin_confidence_midpoints = []
        # zip makes an iterator that aggregates elements from each of the iterables
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = [i for i in range(confidences.shape[0]) if confidences[i] >= bin_lower.item() and confidences[i] < bin_upper.item()]
            weight_of_bin = len(in_bin) / accuracies.shape[0]
            weights_of_bins.append(weight_of_bin)
            if weight_of_bin > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * weight_of_bin
            else:
                accuracy_in_bin = 0
            bin_accuracies.append(accuracy_in_bin)
            bin_confidence_midpoints.append((bin_lower + bin_upper) / 2)

        return ece, bin_confidence_midpoints, bin_accuracies, weights_of_bins


    def calculateTemperature(self, logits, labels, method='bounded'):
        """
        Calibrate the temperature of the model using logits/labels of the validation set.
        Optimize ECE Loss.
        Returns optimal temperature scalar.
        labels must be in one-hot form.
        Shapes of both logits and labels must be the same: num_examples * num_classes
        """
        def loss(temperature):
            return self.getECELoss(logits=logits, labels=labels, temperature=temperature)[0]

        #res = optimize.minimize(loss, [1.0], bounds=[(1.0, 3.0)], method=method)
        res = optimize.minimize_scalar(loss, bounds=(0.9, 3.0), method=method)
        if not res.success:
            print('Optimization failed: {}'.format(res.message))
            return 1.0
        else:
            return res.x
