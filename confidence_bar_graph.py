#!/usr/bin/env python3
"""
Program to generate a reliability diagram when logits and labels files are given.
Also outputs a table of minimum confidence values and their effect on the percentage
of spectra that would be left unclassified.
"""

import matplotlib.pyplot as plt
from confidence_calibration import TemperatureScaling
import numpy as np
import json
from os.path import expanduser

save_path = expanduser("~") + '/ast01/reliability_diagram.png'

logits = []
#path1 = expanduser("~") + '/ast01/logits_5conv2dense.txt'
path1 = expanduser("~") + '/ast01/logits_4conv2dense.txt'
with open(path1) as f:
    logits = np.array(json.loads(f.read()))

labels = []
#path2 = expanduser("~") + '/ast01/labels_5conv2dense.txt'
path2 = expanduser("~") + '/ast01/labels_4conv2dense.txt'
with open(path2) as f:
    labels = np.array(json.loads(f.read()))

num_bins = 26
graph_bins = int(num_bins/2)
temp_scaling = TemperatureScaling(n_bins=num_bins)
temperature = temp_scaling.calculateTemperature(logits, labels)
ece, bin_confidences, bin_accuracies, weights_of_bins = temp_scaling.getECELoss(logits, labels, temperature)
print("Temp: {:.2f}, ECE: {:0.2f}".format(temperature, ece))
#print(bin_confidences)
#print(bin_accuracies)
#print(weights_of_bins)

width = 0.02

overconfidences = []
bar_labels = []
min_conf_accuracies = []  # Weighted accuracy of the higher confidence bins
min_conf_undecided = []   # Fraction of the undecided class (can think of it as a 3rd prediction class)
for x in range(len(bin_confidences)):
    # If we use this bin as a min confidence (inclusive) for prediction, what's the accuracy of the remaining bins?
    if x >= num_bins / 2:
        remaining_weights = np.array([weights_of_bins[x:]])
        remaining_accuracies = np.array([bin_accuracies[x:]])
        min_conf_accuracies.append(np.sum(remaining_weights * remaining_accuracies) / np.sum(remaining_weights))
        min_conf_undecided.append(1.0 - np.sum(remaining_weights))

    if weights_of_bins[x] == 0.0:
        overconfidences.append(0)
        bar_labels.append('0%')
    else:
        bar_labels.append("{:d}%".format(int(max(1, round(weights_of_bins[x] * 100)))))
        if bin_accuracies[x] > bin_confidences[x]:
            overconfidences.append(0)
        else:
            overconfidences.append(bin_confidences[x] - bin_accuracies[x])

print('Min-confidence, Accuracy, % Undecided')
for conf, acc, undecided in zip(bin_confidences[graph_bins:], min_conf_accuracies, min_conf_undecided):
    # Recover the min confidence of each bin, rather than the midpoint confidence.
    print('{:.2f}, {:.1f}%, {:.1f}%'.format(conf - (1/num_bins/2), acc*100, undecided*100))

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(bin_confidences[graph_bins:], bin_accuracies[graph_bins:], width=width, color="#1455bc", label='Bin Accuracy')

# For each bar: Place a label
rects = ax.patches
i=-1
for rect in rects:
    i += 1
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label.
    space = 3
    # Vertical alignment for positive values
    va = 'bottom'
    label = bar_labels[graph_bins+i]
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.
ax.bar(bin_confidences[graph_bins:], overconfidences[graph_bins:], width=width, bottom=bin_accuracies[graph_bins:], color="#f95700", label='Overconfidence Gap')
ax2.plot(bin_confidences[graph_bins:], bin_confidences[graph_bins:], color='gray', linestyle='dashed', linewidth=2, label='Perfect Calibration')
ax.set_xlim([0.49, 1.01])
ax2.set_xlim([0.49, 1.01])
ax.set_ylim([0, 1.1])
ax2.set_ylim([0, 1.1])
ax.set_xlabel('Confidence')
ax.set_ylabel('Accuracy')
ax.legend(loc='lower right')
ax2.legend(loc='center right')

plt.title('Reliability Diagram')
plt.savefig(save_path)
plt.show()
