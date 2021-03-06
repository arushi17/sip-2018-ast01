#!/usr/bin/env python3
"""
1D CNN classifier.
Uses Estimator (new eager execution style).

To protect from being killed by terminal closing:
nohup $SHELL -c "python3 -u tf_trainer.py -m modelname [-a True] [-l 256] 2>&1 | grep --line-buffered -v gpu_device.cc > ../logs/training.log" &
tail -f ../logs/training.log

Works with TF v1.8 GPU, not with v1.9, as of 7/14/18.
To check TF version:
    python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
"""

from __future__ import division, print_function, absolute_import
from confidence_calibration import TemperatureScaling
from datetime import datetime
from os.path import expanduser
from spectrum_class import Spectrum
from spectrum_fits_utils import *
import argparse
import glob
import itertools
import math
import numpy as np
import os
import random
import re
import sys
import time
import tensorflow as tf
from tensorflow.python.client import device_lib

HOME_DIR = expanduser("~") + '/ast01/data'
# where is the dataset?
TRAIN_DATA_DIR = HOME_DIR + '/training_data'
DEV_DATA_DIR = HOME_DIR + '/dev_data'
STAR_DIR = 'star'
NONSTAR_DIR = 'nonstar'
# Use onehot style
STAR_LABEL = [0, 1]
NONSTAR_LABEL = [1, 0]
# Limit the lambda values to this region, as flux at extreme values tends to be very noisy.
MIN_BIN_LAMBDA=5200
MAX_BIN_LAMBDA=9500

# If specified, checkpoints are automatically saved to subdirs within this dir.
# Be careful that different training runs (different hyperparameters) don't load
# the last checkpointed model from a dir and 'continue' training! That is why
# we create unique (timestamped) directories for each training run.
# If set to None, TF automatically creates a temp dir and uses it.
MODEL_DIR = HOME_DIR + '/models'
#MODEL_DIR = None

MAX_NUM_EPOCHS = 200  # max epochs
SHUFFLE_BUFFER = 10000

NUM_CHECKPOINTS_SAVED = 20
# For early stopping, we look for an increase in dev error relative to the best model so far,
# expressed as a percentage.
ES_MIN_GENERALIZATION_LOSS = 1.0
ES_MIN_PROGRESS_QUOTIENT = 0.4

# from tensorflow.layers import ...
# We learn weights for these:
conv1d = tf.layers.conv1d
dense = tf.layers.dense
# No weights for these:
dropout = tf.layers.dropout
average_pooling1d = tf.layers.average_pooling1d
max_pooling1d = tf.layers.max_pooling1d
flatten = tf.layers.flatten
relu = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer

parser = argparse.ArgumentParser(description='Loads spectra from fits files, trains (or evaluates) model.')
parser.add_argument('-e', '--evalconfig', type=str, default='', help='File containing set of configs for evaluation (no training)')
parser.add_argument('-d', '--evaldata', type=str, default='', help='Path of dir containing dev/test data to be evaluated.')
parser.add_argument('-m', '--modelname', type=str, default='logBin4Conv2Dense', help='String name of model function to use.')
parser.add_argument('-a', '--adaptive', type=bool, default=False, help='Use adaptive gaussian smoothing that combines ivar and flux into one series.')
parser.add_argument('-l', '--loglam', type=int, default=0, help='Use adaptive gaussian smoothing AND log(lambda) rescaling and binning of adaptive flux into this pixel width.')
parser.add_argument('-i', '--layershidden', type=int, default=2, help='denseNetBuilder: # of hidden layers.')
parser.add_argument('-f', '--fraction', type=float, default=1.0, help='Fraction of training data to be loaded.')
ARGS = parser.parse_args()


# Inverse of argmax. Works only for 2 classes. 0 -> [1 0], 1 -> [0 1]
def classLabelsToOneHot(labels):
    out = np.zeros((labels.shape[0], 2), dtype=int)
    out[:, 0] = 1 - labels
    out[:, 1] = labels
    return out


# See https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays
# Given a directory, go to the "star" subdirectory (label=0),
# then "nonstar" subdirectory (label=1). Load features from each .fits file.
# Returns tf.data.Dataset, class labels.
# It is efficient to accumulate features in python lists and convert to np array at the end,
# to avoid creating copies of np arrays on each iteration.
def datasetFromFitsDirectory(directory_path, load_fraction=1.0):
    print('Loading data from dir: ' + directory_path)
    feature_data = []
    label_data = []
    star_files = glob.glob(directory_path + '/' + STAR_DIR + '/*.fits')
    nonstar_files = glob.glob(directory_path + '/' + NONSTAR_DIR + '/*.fits')
    filenames = []
    for fname in star_files:
        if random.random() > load_fraction:
            continue
        feature_data.append(featuresFromFits(fname, ARGS, MIN_BIN_LAMBDA, MAX_BIN_LAMBDA))
        label_data.append(STAR_LABEL)
        filenames.append(os.path.basename(fname))
    for fname in nonstar_files:
        if random.random() > load_fraction:
            continue
        feature_data.append(featuresFromFits(fname, ARGS, MIN_BIN_LAMBDA, MAX_BIN_LAMBDA))
        label_data.append(NONSTAR_LABEL)
        filenames.append(os.path.basename(fname))

    features = np.array(feature_data)
    labels = np.array(label_data)
    class_labels = np.argmax(labels, axis=1) # 0 or 1
    print('Shape of features: ' + str(features.shape))
    print('Shape of labels: ' + str(labels.shape))
    return {'dataset' : tf.data.Dataset.from_tensor_slices(({'features' : features}, labels)),
            'filenames' : filenames,
            'features' : features,
            'labels' : class_labels}


# Tensorflow network architecture definition
# Note that the last (typically dense) layer should not have any activation function, since we want
# to interpret that output as logits and apply softmax to it (later).

def denseNetBuilder(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    num_hidden_layers = params['hidden_dense_layers']
    input_width = params['input_width']

    # Calculate the width reduction factor (< 1.0) at each layer.
    # 2 = input_width * factor ^ (num_hidden_layers + 2)
    factor = math.exp((math.log(2) - math.log(input_width))/(num_hidden_layers+2))
    print('NetBuilder reduction factor: {:.3f}'.format(factor))

    net = flatten(x, name='flatten')
    output_width = int(input_width * factor)

    for layer_num in range(1, num_hidden_layers+2):
        input_width = output_width
        output_width = int(input_width * factor)
        name = 'fc' + str(layer_num)
        net = dense(net, output_width, activation=relu, name=name, kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
        name = name + '_dropout'
        net = dropout(net, rate=drop_rate, training=is_training, name=name)

    logits = dense(net, num_classes, name='fc_last', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin3Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']

    net = flatten(x, name='flatten')
    # Input: 128 x 1, outputs: 32 x 1, weights: 128 x 32 = 4096
    net = dense(net, 32, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    # Input: 32 x 1, outputs: 8 x 1, weights: 32 x 8 = 256
    net = dense(net, 8, activation=relu, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc2_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc_last', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin2Conv2Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']

    # Input: 70 x 1, outputs: 60 x 8, weights: 1 x 10 x 8 = 80
    net = conv1d(x, 8, conv1_width, strides=1, padding='valid', data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 60 x 8, outputs: 30 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Input: 30 x 8, outputs: 30 x 16, weights: 8 x 10 x 16 = 1280
    net = conv1d(net, 16, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 30 x 16, outputs: 15 x 16, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 15 x 16
    net = flatten(net, name='flatten')
    # Input: 40 x 1, outputs: 8 x 1, weights: 40 x 8 = 320
    net = dense(net, 8, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin3Conv2Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']

    # Input: 100 x 1, outputs: 90 x 4, weights: 1 x 10 x 4 = 40
    net = conv1d(x, 4, conv1_width, strides=1, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 90 x 4, outputs: 45 x 4, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Input: 45 x 4, outputs: 20 x 8, weights: 4 x 10 x 8 = 320
    net = conv1d(net, 8, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 20 x 8, outputs: 10 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 10 x 8, outputs: 10 x 16, weights: 8 x 5 x 16 = 640
    net = conv1d(net, 16, 5, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 10 x 16, outputs: 5 x 16, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')

    # Input: 5 x 16
    net = flatten(net, name='flatten')
    # Input: 80 x 1, outputs: 8 x 1, weights: 80 x 8 = 640
    net = dense(net, 8, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin4Conv2Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']

    # Input: 128 x 1, outputs: 128 x 4, weights: 1 x 11 x 4 = 44
    net = conv1d(x, 4, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 128 x 4, outputs: 64 x 4, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Input: 64 x 4, outputs: 64 x 8, weights: 4 x 11 x 8 = 352
    net = conv1d(net, 8, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 64 x 8, outputs: 32 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 32 x 8, outputs: 32 x 16, weights: 8 x 11 x 32 = 2816
    net = conv1d(net, 16, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 32 x 16, outputs: 16 x 16, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')

    # Input: 16 x 16, outputs: 16 x 32, weights: 16 x 7 x 32 = 3584
    net = conv1d(net, 32, 7, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv4', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 16 x 32, outputs: 4 x 32, weights: 0
    net = max_pooling1d(net, 4, 4, data_format='channels_last', name='pool4')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool4_dropout')

    # Input: 4 x 32
    net = flatten(net, name='flatten')
    # Input: 128 x 1, outputs: 8 x 1, weights: 128 x 8 = 1024
    net = dense(net, 8, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin5Conv2Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']

    # Input: 128 x 1, outputs: 128 x 4, weights: 1 x 11 x 4 = 44
    net = conv1d(x, 4, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 128 x 4, outputs: 64 x 4, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Input: 64 x 4, outputs: 64 x 8, weights: 4 x 11 x 8 = 352
    net = conv1d(net, 8, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 64 x 8, outputs: 32 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 32 x 8, outputs: 32 x 16, weights: 8 x 11 x 32 = 2816
    net = conv1d(net, 16, conv1_width, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 32 x 16, outputs: 16 x 16, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')

    # Input: 16 x 16, outputs: 16 x 32, weights: 16 x 7 x 32 = 3584
    net = conv1d(net, 32, 7, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv4', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 16 x 32, outputs: 8 x 32, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool4')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool4_dropout')

    # Input: 8 x 32, outputs: 8 x 64, weights: 32 x 5 x 64 = 10240
    net = conv1d(net, 32, 5, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv5', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 8 x 64, outputs: 2 x 64, weights: 0
    net = max_pooling1d(net, 4, 4, data_format='channels_last', name='pool5')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool5_dropout')

    # Input: 2 x 64
    net = flatten(net, name='flatten')
    # Input: 128 x 1, outputs: 8 x 1, weights: 128 x 8 = 1024
    net = dense(net, 8, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


# Expects full 8k input width (adaptive smoothing), looks for patterns like CA triplet.
def veryWideCNN(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']

    # Input: 8k x 1, outputs: 8188 x 2
    net = conv1d(x, 2, 7, strides=1, padding='valid', data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Input: 4144 x 1, outputs: ~ 3900 x 1
    net = conv1d(x, 1, conv1_width, strides=1, padding='valid', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 3900 x 1, outputs: ~ 1950 x 1
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Only 1 output
    net = max_pooling1d(net, 1800, 1500, data_format='channels_last', name='pool_wide')

    # Input: 1 x 1
    net = flatten(net, name='flatten')
    # Input: 1 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def verySmall1DCNN(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']

    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    # Smoothing layer (only 1 filter), not learning features in this layer.
    # Input: 8k x # channels, outputs: 810 x 1, weights: # channels x 61 x 2
    net = conv1d(x, 1, 61, strides=10, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))

    # Extract low-level features (individual absorption and emission lines)
    # Input: 810 x 2, outputs: 384 x 4, weights: 2 x 40 x 4 = 320
    net = conv1d(net, 4, 40, strides=2, data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 384 x 4, outputs: 96 x 4, weights: 0
    net = max_pooling1d(net, 4, 4, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Extract higher-level features (specific combinations of absorption and emission lines)
    # Input: 96 x 4, outputs: 56 x 8, weights: 4 x 40 x 8 = 1280
    net = conv1d(net, 8, 40, strides=1, data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 56 x 8, outputs: 1 x 8, weights: 0
    net = max_pooling1d(net, 56, 56, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 1 x 8
    net = flatten(net, name='flatten')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def small1DCNN(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']

    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    net = conv1d(x, 8, 61, strides=10, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # conv1d reshapes output to [batch, out_width, out_channels]
    # 820 x 8
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    # 410 x 8

    net = conv1d(net, 16, 61, strides=10, data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 41 x 16
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    # 20 x 16
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')
    # 20 x 16

    net = flatten(net, name='flatten')
    # 320
    net = dense(net, 64, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def medium1DCNN(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']

    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    net = conv1d(x, 8, 31, strides=5, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # conv1d reshapes output to [batch, out_width, out_channels]
    # 1633 x 8
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    # 820 x 8
    # net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')
    # 820 x 8

    net = conv1d(net, 16, 31, 5, data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 164 x 16
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    # 82 x 16
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')
    # 82 x 16

    net = conv1d(net, 32, 31, 5, data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 16 x 32
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    # 8 x 32
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')
    # 8 x 32

    net = flatten(net, name='flatten')
    # 256
    net = dense(net, 64, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def full8k5Conv2Dense(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    conv1_width = params['conv1_width']
    strides = params['strides']
    max_pool = params['max_pool']

    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    net = conv1d(x, 8, conv1_width, strides=strides, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # conv1d reshapes output to [batch, out_width, out_channels]
    # 1633 x 8
    net = max_pooling1d(net, max_pool, max_pool, data_format='channels_last', name='pool1')
    # 820 x 8
    # net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')
    # 820 x 8

    net = conv1d(net, 16, conv1_width, strides=strides, data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 164 x 16
    net = max_pooling1d(net, max_pool, max_pool, data_format='channels_last', name='pool2')
    # 82 x 16
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')
    # 82 x 16

    net = conv1d(net, 32, 7, strides=2, data_format='channels_last', activation=relu, name='conv3', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 16 x 32
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    # 8 x 32
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')
    # 8 x 32

    net = conv1d(net, 64, 5, strides=1, data_format='channels_last', activation=relu, name='conv4', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # 16 x 32
    net = max_pooling1d(net, 4, 4, data_format='channels_last', name='pool4')
    # 8 x 32
    net = dropout(net, rate=drop_rate, training=is_training, name='pool4_dropout')
    # 8 x 32

    net = flatten(net, name='flatten')
    # 256
    net = dense(net, 16, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


# Number of weights will be 4 * lstm_units * (lstm_units + 1)
def logBinLSTM1Layer(x, is_training, params):
    num_classes = params['num_classes']
    drop_rate = params['drop_rate']
    l2_scale = params['l2_scale']
    lstm_units = params['lstm_units']

    cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)  
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0-drop_rate)
    outputs, final_states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    logits = dense(final_states.h, num_classes, name='fc', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))  
    return logits, x


# Print summary about the learned network, including weights.
# Input: Estimator (model). Requires latest checkpoint to be available.
def printNetworkInfo(model, print_weights=False):
    print('\nLEARNED WEIGHTS INFO:')
    total_weights = 0
    for layer_name in model.get_variable_names():
        # Care about weights of only conv and dense (fc) layers
        if not re.match(r'conv.*/kernel$', layer_name) and not re.match(r'fc.*/kernel$', layer_name):
            continue
        weights = model.get_variable_value(layer_name)
        num_weights = np.prod(weights.shape)
        print('Layer {} has {} weights {}'.format(layer_name, num_weights, str(weights.shape)))
        total_weights += num_weights
        if print_weights:
            print('{}: {}'.format(layer_name, weights))

    print('This model has {} trainable parameters/weights.'.format(total_weights))


# Needs to follow this signature, per tf.estimator. Should support TRAIN, EVAL, PREDICT
# Assumes labels follows the onehot style.
def modelFn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # Allow network model to be picked dynamically
    logits, x = globals()[params['model_name']](
                features['features'],
                is_training,
                params)

    # Apply temperature scaling to logits to get properly calibrated probabilities (confidence).
    # See https://arxiv.org/pdf/1706.04599.pdf
    logits = logits / params['temperature']

    # Use argmax to convert probabilities to 0, 1.
    probabilities = tf.nn.softmax(logits)
    predictions=tf.argmax(probabilities, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Package all useful output into a map
        predict_output = {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predict_output)

    # Negative Log Likelihood (NLL) loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=labels))
    # TF does not automatically add regularization losses.
    l2_loss = tf.losses.get_regularization_loss()  # Scalar value
    loss += l2_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Assume onehot labels. Returns the metric and the update op.
    # Give the variables different train/dev prefixes so that they are tracked separately.
    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                              predictions=predictions,
                              name='train_accuracy' if is_training else 'dev_accuracy')
    precision = tf.metrics.precision(labels=tf.argmax(labels, axis=1),
                              predictions=predictions,
                              name='train_precision' if is_training else 'dev_precision')
    recall = tf.metrics.recall(labels=tf.argmax(labels, axis=1),
                              predictions=predictions,
                              name='train_recall' if is_training else 'dev_recall')
    auc = tf.metrics.auc(labels=tf.argmax(labels, axis=1),
                              predictions=predictions,
                              name='train_auc' if is_training else 'dev_auc')

    # Used in EVAL mode. Loss is calculated by default. Add others here.
    eval_metrics = {'accuracy': acc,
               'precision': precision,
               'recall': recall,
               'auc': auc}
    # Make data available to Tensorboard. http://localhost:6006
    tf.summary.scalar('accuracy', acc[1])
    tf.summary.scalar('precision', precision[1])
    tf.summary.scalar('recall', recall[1])
    tf.summary.scalar('auc', auc[1])
    # Draw the histograms separately.
    tf.summary.histogram('training probabilities' if is_training else 'dev probabilities', probabilities)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metrics)

    # TRAIN mode
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics)


# Returns the index of the checkpoint entry with the highest dev accuracy
def bestCheckpointIndex(checkpoint_tracker):
    cp_max_acc = 0
    i = 0
    best_i = 0
    for cp_file, dev_loss, dev_acc in checkpoint_tracker:
        if dev_acc > cp_max_acc:
            cp_max_acc = dev_acc
            best_i = i
        i = i + 1
    return best_i


# Returns True if early stopping criteria are met (that is, we should stop training now).
# See https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
# Using the second approach in this paper, which waits for the training rate to
# plateau before considering early stopping.
# TODO: Better early stopping. See https://github.com/tensorflow/tensorflow/issues/18394
# (the feature will be released in TF 1.10).
def shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, epoch, params):
    if epoch < params['es_min_epochs']:
        # Clearly too early to stop training
        return False
    lowest_dev_loss = np.min(dev_losses[0:epoch+1])
    latest_dev_loss = dev_losses[epoch]
    generalization_loss = 100.0 * (latest_dev_loss/lowest_dev_loss - 1.0)

    # Check the training rate progress, has it plateaued in the last few epochs?
    avg_train_loss = np.mean(training_losses[epoch-4:epoch+1])
    min_train_loss = np.min(training_losses[epoch-4:epoch+1])
    if min_train_loss > 0:
        progress = 1000.0 * (avg_train_loss / min_train_loss - 1.0)
    else:
        # If training loss = 0, we are unlikely to learn anything more and make more progress
        progress = 0.01  # Some small number to cause progress_quotient to be exceeded

    progress_quotient = generalization_loss / progress
    if training_accuracies[epoch] < 0.999 and (progress_quotient < ES_MIN_PROGRESS_QUOTIENT or generalization_loss < ES_MIN_GENERALIZATION_LOSS):
        return False

    # Should stop now. Find the best saved checkpoint using accuracy (not loss).
    print('Progress quotient: {:.3f}, generalization loss: {:.3f}, stopping early!'.format(progress_quotient, generalization_loss))
    return True


# Given a checkpoint_path like /home/arushi/ast01/models/2018-07-25-07-19-13/model.ckpt-1581
# deletes all checkpoints in that directory other than the one given. To save disk space.
def deleteExtraCheckpoints(cp_path):
    print('Deleting extra checkpoints other than {}'.format(cp_path))
    dirname = os.path.dirname(cp_path)
    for filepath in glob.glob(dirname + '/*'):
        if not filepath.startswith(cp_path):
            try:
                os.remove(filepath)
            except OSError:
                pass


def trainModel(train_metadata, dev_metadata, params, results):
    batch_size = params['batch_size']
    print('==========================================================================')
    print('\nBEGIN MODEL: {}, LEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}, L2_SCALE: {}, '
    'CONV1_WIDTH: {}, STRIDES: {}, MAX_POOL: {}, LSTM_UNITS: {}'
    ''.format(params['model_name'], params['learning_rate'], params['batch_size'], params['drop_rate'],
    params['l2_scale'], params['conv1_width'], params['strides'], params['max_pool'], params['lstm_units']))
    start_time = time.time()

    # The training dataset needs to last for all epochs, for use with one_shot_iterator.
    raw_train_dataset = train_metadata['dataset']
    eval_of_train_dataset = raw_train_dataset.batch(batch_size)
    # Returns features, label each time it is called. Do not need to repeat and shuffle for eval.
    def evalOfTrainInputFn():
        return eval_of_train_dataset.make_one_shot_iterator().get_next()

    train_dataset = raw_train_dataset.shuffle(SHUFFLE_BUFFER).repeat(MAX_NUM_EPOCHS).batch(batch_size)
    # Returns features, label each time it is called.
    def trainInputFn():
        return train_dataset.make_one_shot_iterator().get_next()

    # For the dev set, we generally want to evaluate/predict the entire set.
    # Does the batch size even matter? conv1d gives dimension mismatch error without batching.
    dev_dataset = dev_metadata['dataset']
    dev_dataset = dev_dataset.batch(batch_size)

    dev_features = dev_metadata['features']
    dev_labels = dev_metadata['labels']
    dev_filenames = dev_metadata['filenames']

    # Returns features, label each time it is called.
    def devInputFn():
        return dev_dataset.make_one_shot_iterator().get_next()

    # Creates additional configurations for the estimator
    num_steps_in_epoch = max(1, int(len(train_metadata['labels'])/batch_size))
    print('Number of steps in epoch: {}'.format(num_steps_in_epoch))

    # TODO: Looks like a bug: Estimator is producing one extra checkpoint per requested
    # checkpoint (one step after the requested one). So double the number of checkpoints.
    my_checkpointing_config = tf.estimator.RunConfig(
        save_summary_steps = 10000,  # Write to events.out.tfevents file less often than default 100.
        save_checkpoints_steps = num_steps_in_epoch,  # Save checkpoints every epoch.
        keep_checkpoint_max = 2*NUM_CHECKPOINTS_SAVED,  # Retain so many recent checkpoints.
    )

    # define saver object
    model_save_dir = MODEL_DIR + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print('Saving checkpoints to: ' + model_save_dir)

    # Build
    model = tf.estimator.Estimator(
        modelFn,
        params=params,
        model_dir=model_save_dir,
        config=my_checkpointing_config)

    # Keeps track of loss and accuracy after each epoch
    checkpoint_tracker = []

    # Train
    training_accuracies = np.zeros(MAX_NUM_EPOCHS)
    training_losses = np.zeros(MAX_NUM_EPOCHS)
    dev_accuracies = np.zeros(MAX_NUM_EPOCHS)
    dev_losses = np.zeros(MAX_NUM_EPOCHS)
    for epoch in range(MAX_NUM_EPOCHS):
        # Some params may result in negative dimension error. Catch and discard this training run.
        try:
            model.train(input_fn=trainInputFn, steps=num_steps_in_epoch)
        except:
            print('!!!!!! Training: Incompatible model params?')
            print(params)
            return

        # On Tensorboard, these will show up as 'eval_training' and 'eval_validation'
        eval_of_train = model.evaluate(evalOfTrainInputFn, name='training')
        eval_of_dev = model.evaluate(devInputFn, name='validation')

        latest_train_acc = np.asscalar(eval_of_train['accuracy'])
        latest_train_loss = np.asscalar(eval_of_train['loss'])
        latest_dev_acc = np.asscalar(eval_of_dev['accuracy'])
        latest_dev_loss = np.asscalar(eval_of_dev['loss'])
        print('Epoch: {:02d}/{} => Train Loss: {:.3f}, Dev Loss: {:.3f}. Train Acc: {:.3f}, Dev Acc: {:.3f}'
              ''.format(epoch+1, MAX_NUM_EPOCHS, latest_train_loss, latest_dev_loss, latest_train_acc, latest_dev_acc))

        training_accuracies[epoch] = latest_train_acc
        training_losses[epoch] = latest_train_loss
        dev_accuracies[epoch] = latest_dev_acc
        dev_losses[epoch] = latest_dev_loss

        checkpoint_tracker.append((model.latest_checkpoint(), latest_dev_loss, latest_dev_acc))
        # Keep only as much data as we have checkpoints saved
        checkpoint_tracker = checkpoint_tracker[-NUM_CHECKPOINTS_SAVED:]

        if shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, epoch, params):
            break

    best_cp_index = bestCheckpointIndex(checkpoint_tracker)
    cp_path, cp_dev_loss, cp_dev_acc = checkpoint_tracker[best_cp_index]

    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_minutes = int((elapsed_time - (elapsed_hours * 3600)) / 60)

    # Restore the best model from the saved checkpoints and re-evaluate dev on it using model.predict().
    predict_output = model.predict(devInputFn, checkpoint_path=cp_path)
    _, temperature = evaluatePredictOutput(predict_output, dev_labels, dev_filenames, params, results, do_temp_scaling=True)

    # Print network info. This uses latest checkpoint, so delete extra checkpoints after this.
    printNetworkInfo(model, print_weights=False)
    deleteExtraCheckpoints(cp_path)

    results['accurate_runs'].append((cp_dev_acc, 'cp_dev_acc: {:.3f} train_acc: {:.3f} batch_size: {} '
    'learning_rate: {:.4f} drop_rate: {:.3f} l2_scale: {:.3f} conv1_width: {} strides: {} max_pool: {} '
    'lstm_units: {} temperature: {:.2f} cp_path: {} ({} epochs, {} hours {} mins)'
    ''.format(cp_dev_acc, latest_train_acc, params['batch_size'], params['learning_rate'],
    params['drop_rate'], params['l2_scale'], params['conv1_width'], params['strides'], params['max_pool'],
    params['lstm_units'], temperature, cp_path, epoch+1, elapsed_hours, elapsed_minutes)))
    print('\nRUN RESULTS: {}'.format(results['accurate_runs'][-1][1]))

    print('=================== Done training ========================================')


# predict_output is a generator, which will produce a dict for each example.
# Will append overall stats to results, and print evaluation results.
# Can optionally do TemperatureScaling.
# Returns overall accuracy, temperature.
def evaluatePredictOutput(predict_output, dev_labels, dev_filenames, params, results, do_temp_scaling=False, print_logits_labels=False):
    dev_labels_onehot = classLabelsToOneHot(dev_labels)
    predict_output = list(predict_output)  # So we can iterate over it multiple times
    predicted_classes = [pred_dict['predictions'] for pred_dict in predict_output]  # Class labels, NOT one-hot
    predicted_classes = np.array(predicted_classes)
    logits = [pred_dict['logits'] for pred_dict in predict_output]
    logits = np.array(logits)
    pre_ece_probabilities = np.array([pred_dict['probabilities'] for pred_dict in predict_output])

    temperature = 1.0
    if do_temp_scaling:
        temp_scaling = TemperatureScaling(n_bins=30)
        print('Before temperature calibration, ECE loss: {:0.3f}'
              ''.format(temp_scaling.getECELoss(logits, dev_labels_onehot)))
        temperature = temp_scaling.calculateTemperature(logits, dev_labels_onehot)
        print('After temperature (={:0.2f}) calibration, ECE loss: {:0.3f}'
              ''.format(temperature, temp_scaling.getECELoss(logits, dev_labels_onehot, temperature)))
        # Recompute probabilities using temperature
        post_ece_probabilities_tf = tf.nn.softmax(logits / temperature)
        with tf.Session() as sess:
            post_ece_probabilities = sess.run(post_ece_probabilities_tf)

    if print_logits_labels:
        print(logits / temperature)
        print(dev_labels_onehot)

    # Print the filenames of the fits we got wrong, for further debugging/analysis
    # Class labels need to be 0 or 1, not onehot.
    truth_labels = dev_labels.tolist()
    print('\nDEV EXAMPLES WE GOT WRONG:')
    num_wrong = 0
    for i in range(len(truth_labels)):
        dev_filename = dev_filenames[i]
        if (truth_labels[i] != predicted_classes[i]):
            num_wrong += 1
            if not dev_filename in results['incorrect']:
                results['incorrect'][dev_filename] = 1
            else:
                results['incorrect'][dev_filename] = results['incorrect'][dev_filename] + 1
            if do_temp_scaling:
                print('Predicted {} (conf {:.2f} -->temp-scaling--> {:.2f}), true {}: {}'.format(predicted_classes[i],
                    pre_ece_probabilities[i].max(), post_ece_probabilities[i].max(), truth_labels[i], dev_filename))
            else:
                print('Predicted {} (conf {:.2f}), true {}: {}'.format(predicted_classes[i],
                    pre_ece_probabilities[i].max(), truth_labels[i], dev_filename))

    dev_acc = (len(truth_labels) - num_wrong) / len(truth_labels)
    # Print the dev confusion matrix using the best checkpointed model.
    with tf.Session() as sess:
        confusion_matrix = tf.confusion_matrix(truth_labels, predicted_classes)
        matrix_to_print = sess.run(confusion_matrix)
        print('\nCONFUSION MATRIX:')
        print(matrix_to_print)

    return dev_acc, temperature


def printResults(results):
    results['accurate_runs'].sort(reverse=True)
    print('\nHYPERPARAMETERS WITH HIGHEST DEV ACCURACY:')
    for acc, params in results['accurate_runs']:
        print(params)
    most_wrong = sorted(results['incorrect'].items(), key=lambda kv: kv[1], reverse=True)
    print('\nSPECTRA THAT WERE PREDICTED INCORRECTLY THE MOST:')
    for name, numwrong in most_wrong:
        print('{}: {}'.format(name, numwrong))


def runMainTrain():
    print('This version of TF built with CUDA? {}'.format(tf.test.is_built_with_cuda()))
    print(device_lib.list_local_devices())
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Don't log Info and Warning level messages (too many of them)
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ES_MIN_EPOCHS = 20
    if ARGS.adaptive:
        print('Will use adaptive gaussian smoothing using ivar and flux.')
    elif ARGS.loglam:
        print('Will use adaptive gaussian smoothing, and {} binning by log(lambda).'.format(ARGS.loglam))
        ES_MIN_EPOCHS = 40
    else:
        print('Will use raw ivar and flux channels.')

    if not ARGS.modelname in globals():
        print('ERROR: Model {} does not exist!'.format(ARGS.modelname))
        sys.exit(1)

    print('Will use model {}'.format(ARGS.modelname))
    print('\nStarting at datetime: {}'.format(str(datetime.now())))

    # Adjust early stopping min epochs up if we are given a smaller dataset, to make sure
    # we train for a similar number of steps at least.
    #ES_MIN_EPOCHS = int(ES_MIN_EPOCHS/ARGS.fraction)

    # hyperparameters should be optimized using the dev set.
    # High learning rates cause error/accuracy to jump around and interfere with early stopping,
    # but strong regularization seems to help.
    LEARNING_RATES = [4e-4, 1e-4]  # Use with conv
    #LEARNING_RATES = [1e-2, 3e-3, 1e-3]  # Use with logbinned, lstm
    BATCH_SIZES = [32, 48]

    DROP_RATES = [0.007]  # Use with logbinned, conv
    #DROP_RATES = [0.1, 0.01, 0]  # Use with logbinned, fully connected
    #DROP_RATES = [0.2, 0.1]  # Use with logbinned, lstm
    #DROP_RATES = [0.25]  # Use with full 8k

    # L2_SCALE cannot be 0
    L2_SCALES = [0.001, 0.003]  # Use with logbinned, conv
    #L2_SCALES = [0.001, 0.01, 0.1]  # Use with logbinned, fully connected
    #L2_SCALES = [0.003, 0.01]  # Use with logbinned, lstm
    #L2_SCALES = [0.1]  # Use with full 8k

    CONV1_WIDTH = [5]  # Use with logbinned, conv
    #CONV1_WIDTH = [5, 7, 9]  # Use with logbinned, conv
    #CONV1_WIDTH = [15, 31, 61]  # Use with full 8k

    STRIDES = [3]
    #STRIDES = [3, 5]  # Use with full 8k
    MAX_POOL = [2]
    #MAX_POOL = [2, 4]  # Use with full 8k

    #LSTM_UNITS = [100, 200]
    LSTM_UNITS = [100]

    combinations = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROP_RATES, L2_SCALES, CONV1_WIDTH, STRIDES, MAX_POOL, LSTM_UNITS))
    random.shuffle(combinations)
    all_params_combs = []
    print('Will run the following {} combinations of hyperparams:'.format(len(combinations)))
    for (learning_rate, batch_size, drop_rate, l2_scale, conv1_width, strides, max_pool, lstm_units) in combinations:
        # All of these params may not be used by the model.
        train_params={
                # Fixed params:
                'model_name': ARGS.modelname,
                'num_classes': 2,
                'es_min_epochs': ES_MIN_EPOCHS,
                'hidden_dense_layers': ARGS.layershidden,
                'input_width': ARGS.loglam,
                'temperature': 1.0,
                # hyperparams:
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'drop_rate': drop_rate,
                'l2_scale': l2_scale,
                'conv1_width': conv1_width,
                'strides': strides,
                'max_pool': max_pool,
                'lstm_units': lstm_units,
                }
        all_params_combs.append(train_params)
        print(train_params)

    train_metadata = datasetFromFitsDirectory(TRAIN_DATA_DIR, ARGS.fraction)
    #dev_metadata = datasetFromFitsDirectory(DEV_DATA_DIR, 1.0)
    dev_metadata = datasetFromFitsDirectory(DEV_DATA_DIR, ARGS.fraction)
    print('Finished loading data at datetime: {}'.format(str(datetime.now())))

    results = {'incorrect': {},
               'accurate_runs' : [],
              }
    for train_params in all_params_combs:
        trainModel(train_metadata, dev_metadata, train_params, results)

    printResults(results)

    print('Finished at datetime: {}'.format(str(datetime.now())))


# Loads checkpoint from saved_model_dir, evaluates it on dev_metadata using modelFn.
def evaluateModel(dev_metadata, params, results):
    dev_dataset = dev_metadata['dataset']
    dev_dataset = dev_dataset.batch(params['batch_size'])

    dev_features = dev_metadata['features']
    dev_labels = dev_metadata['labels']
    dev_filenames = dev_metadata['filenames']

    # Returns features, label each time it is called.
    def devInputFn():
        return dev_dataset.make_one_shot_iterator().get_next()

    cp_path=params['cp_path']
    print('Will evaluate checkpoint: {}'.format(cp_path))
    saved_model_dir = os.path.dirname(cp_path)

    model = tf.estimator.Estimator(
        modelFn,
        params=params,
        model_dir=saved_model_dir)

    try:
        predict_output = model.predict(devInputFn, checkpoint_path=cp_path)
        test_acc, _ = evaluatePredictOutput(predict_output, dev_labels, dev_filenames, params, results, do_temp_scaling=True, print_logits_labels=True)
    except:
        print('!!!!!! Likely incompatible model params for {}'.format(cp_path))
        return

    print('Architecture: {} model_dir: {} input_width: {} conv1_width: {} strides: {} max_pool: {} Test-Acc: {:.3f}'
          ''.format(params['model_name'], os.path.basename(saved_model_dir), params['input_width'],
          params['conv1_width'], params['strides'], params['max_pool'], test_acc))
    print('==========================================================================')


def runMainEval():
    all_params = []
    # Read eval configs from file. Each line looks like:
    #conv1_width:5 input_width:128 cp_path:/Users.../2018-08-10-02-19-43/model.ckpt-1275
    with open(ARGS.evalconfig) as f:
        for line in f.readlines():
            params = {
                # Default (and unused) params that are needed only during training
                'num_classes': 2,
                'drop_rate': 0,
                'l2_scale': 0.01,  # Cannot be 0
                'learning_rate': 0.001,
                'batch_size': 32,
                # Params that actually affect the architecture and output in PREDICT/EVAL modes:
                'model_name': ARGS.modelname,
                'hidden_dense_layers': ARGS.layershidden,
                'input_width': ARGS.loglam,
                'conv1_width': 7,
                'strides': 5,
                'max_pool': 2,
                'lstm_units': 1,
                'temperature': 1.0,
            }
            for phrase in line.rstrip().split():
                key, val = phrase.split(':')
                # Some values could be int (no use case for float yet)
                try:
                    val = int(val)
                except:
                    pass
                params[key] = val
            print(params)
            all_params.append(params)

    dev_metadata = datasetFromFitsDirectory(DEV_DATA_DIR, ARGS.fraction)
    print('Finished loading data at datetime: {}'.format(str(datetime.now())))
    results = {'incorrect': {},
               'accurate_runs' : [],
              }
    for params in all_params:
        evaluateModel(dev_metadata, params, results)
                
    printResults(results)


if __name__=='__main__':
    if ARGS.evaldata:
        DEV_DATA_DIR = ARGS.evaldata
        print('Will load dev/test data for evaluation from: {}'.format(DEV_DATA_DIR))

    if ARGS.evalconfig:
        runMainEval()
    else:
        runMainTrain()
