#!/usr/bin/env python3
""" 
1D CNN classifier.
Uses Estimator (new eager execution style).

To protect from being killed by terminal closing:
nohup $SHELL -c "python3 -u tf_trainer.py [-a True] 2>&1 | grep --line-buffered -v gpu_device.cc > ../logs/training.log" &
tail -f ../logs/training.log

Works with TF v1.8 GPU, not with v1.9, as of 7/14/18.
To check TF version:
    python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
"""

from __future__ import division, print_function, absolute_import
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
import time
import tensorflow as tf
from tensorflow.python.client import device_lib

HOME_DIR = expanduser("~") + '/ast01'
# where is the dataset?
TRAIN_DATA_DIR = HOME_DIR + '/training_data'
DEV_DATA_DIR = HOME_DIR + '/dev_data'
STAR_DIR = 'star'
NONSTAR_DIR = 'nonstar'
# Use onehot style
STAR_LABEL = [0, 1]
NONSTAR_LABEL = [1, 0]

# If specified, checkpoints are automatically saved to subdirs within this dir.
# Be careful that different training runs (different hyperparameters) don't load
# the last checkpointed model from a dir and 'continue' training! That is why
# we create unique (timestamped) directories for each training run.
# If set to None, TF automatically creates a temp dir and uses it.
MODEL_DIR = HOME_DIR + '/models'
#MODEL_DIR = None

NUM_EPOCHS = 200  # max epochs
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
max_pooling1d = tf.layers.max_pooling1d
flatten = tf.layers.flatten
relu = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer

parser = argparse.ArgumentParser(description='Loads spectra from fits files, trains model.')
parser.add_argument('-a', '--adaptive', type=bool, default=False, help='Use adaptive gaussian smoothing that combines ivar and flux into one series.') 
parser.add_argument('-l', '--loglam', type=int, default=0, help='Use adaptive gaussian smoothing AND log(lambda) rescaling and binning of adaptive flux into this pixel width.') 
parser.add_argument('-f', '--fraction', type=float, default=1.0, help='Fraction of training data to be loaded.') 
ARGS = parser.parse_args()

# Data loading code. Load from *.fits files.
# See https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays
# Given a directory, go to the "star" subdirectory (label=0),
# then "nonstar" subdirectory (label=1).
# For each file:
# - Use Spectrum class to load flux and ivar into numpy arrays.
# - Standardize each series. Optionally truncate outlier ivar values.

# If adaptive smoothing, returns a numpy array of 8kx1 dimension (float32)
# Otherwise returns a numpy array of 8kx2 dimension
# TODO: Need to check for invalid ivar and reject the fits file.
def featuresFromFits(filepath):
    print('Reading file {}'.format(filepath))
    spec = Spectrum(filepath)
    if ARGS.adaptive:
        # We have only one channel, smoothed flux.
        smoothed_flux = np.float32(standardize(adaptiveSmoothing(spec.flux, spec.ivar)))
        return smoothed_flux.reshape((len(smoothed_flux), 1))
    elif ARGS.loglam:
        # We have only one channel, smoothed flux. Also stretch/compress it so that pixel
        # distance between any known emission/absorption features is the same irrespective of
        # amount of redshift (z). Also re-bin the flux into a smaller number of pixels.
        smoothed_flux = adaptiveSmoothing(spec.flux, spec.ivar)
        flux, lam = logBinPixels(ARGS.loglam, spec.lam, smoothed_flux)
        log_binned_flux = np.float32(standardize(flux))
        return log_binned_flux.reshape((len(log_binned_flux), 1))
    else:
        # Make both raw ivar and flux as channels.
        flux = cleanValues(spec.flux)
        # TODO: Try just scaling flux from 0 to 1
        flux = np.float32(standardize(flux))

        ivar = cleanValues(spec.ivar)
        ivar = np.sqrt(ivar)
        ivar = limitOutliers(ivar, 2.5)
        ivar = np.float32(standardize(ivar))
        # TODO: ivar of 0 has a special meaning (don't trust the flux)
        # ivar = scale(ivar, 0, 1.0)

        if flux.shape != ivar.shape:
            raise ValueError(filepath + ': flux.shape: ' + flux.shape + ', ivar.shape: ' + ivar.shape)
        # Channels should be last for most tf layers
        return np.transpose(np.array([flux, ivar]))


# Returns tf.data.Dataset, class labels
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
        feature_data.append(featuresFromFits(fname))
        label_data.append(STAR_LABEL)
        filenames.append(os.path.basename(fname))
    for fname in nonstar_files:
        if random.random() > load_fraction:
            continue
        feature_data.append(featuresFromFits(fname))
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
# TODO: Make some of the conv1d sizes etc hyperparameters

def logBin2Conv2Dense(x, num_classes, is_training, drop_rate, l2_scale):
    # Extract low-level features (individual absorption and emission lines)
    # Input: 20 x 1, outputs: 20 x 4, weights: 1 x 10 x 4 = 40
    net = conv1d(x, 4, 10, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 20 x 4, outputs: 10 x 4, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Extract higher-level features (specific combinations of absorption and emission lines)
    # Input: 10 x 4, outputs: 10 x 8, weights: 4 x 10 x 8 = 320
    net = conv1d(net, 8, 10, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 10 x 8, outputs: 5 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Input: 5 x 8
    net = flatten(net, name='flatten')
    # Input: 40 x 1, outputs: 8 x 1, weights: 40 x 8 = 320
    net = dense(net, 8, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')
    # Input: 8 x 1, outputs: 2 x 1, weights: 16
    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


def logBin3Conv2Dense(x, num_classes, is_training, drop_rate, l2_scale):
    # Extract low-level features (individual absorption and emission lines)
    # Input: 100 x 1, outputs: 90 x 4, weights: 1 x 10 x 4 = 40
    net = conv1d(x, 4, 10, strides=1, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 90 x 4, outputs: 45 x 4, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool1_dropout')

    # Extract higher-level features (specific combinations of absorption and emission lines)
    # Input: 45 x 4, outputs: 20 x 8, weights: 4 x 10 x 8 = 320
    net = conv1d(net, 8, 10, strides=1, padding='same', data_format='channels_last', activation=relu, name='conv2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    # Input: 20 x 8, outputs: 10 x 8, weights: 0
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')

    # Extract more higher-level features (specific combinations of absorption and emission lines)
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


def verySmall1DCNN(x, num_classes, is_training, drop_rate, l2_scale):
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


def small1DCNN(x, num_classes, is_training, drop_rate, l2_scale):
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


def medium1DCNN(x, num_classes, is_training, drop_rate, l2_scale):
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
    logits, x = logBin2Conv2Dense(x=features['features'], 
                            num_classes=params['num_classes'], 
                            is_training=is_training,
                            drop_rate=params['drop_rate'],
                            l2_scale=params['l2_scale'])
    y_hat = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.argmax(y_hat, axis=1))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, 
        labels=labels))
    # TF does not automatically add regularization losses.
    l2_loss = tf.losses.get_regularization_loss()  # Scalar value
    loss += l2_loss

    train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
        loss, global_step=tf.train.get_global_step(), 
        name='train_acc' if is_training else 'dev_acc')

    # Assume onehot
    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), 
                              predictions=tf.argmax(y_hat, axis=1),
                              name='train_acc' if is_training else 'dev_acc')

    metrics = {'accuracy': acc}
    # Make available to tensorboard in TRAIN mode. http://localhost:6006
    tf.summary.scalar('accuracy', acc[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=tf.argmax(y_hat, axis=1),
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


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
def shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, epoch):
    if epoch < 10:
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


def trainModel(train_metadata, dev_metadata, learning_rate, batch_size, drop_rate, l2_scale, results):
    print('==========================================================================')
    print('\nBEGIN LEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}, L2_SCALE: {}'.format(learning_rate, batch_size, drop_rate, l2_scale))
    start_time = time.time()

    # The training dataset needs to last for all epochs, for use with one_shot_iterator.
    train_dataset = train_metadata['dataset']
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).repeat(NUM_EPOCHS).batch(batch_size)

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
        params={'learning_rate': learning_rate, 
                'batch_size': batch_size,
                'drop_rate': drop_rate,
                'l2_scale': l2_scale,
                'num_classes': 2},
        model_dir=model_save_dir,
        config=my_checkpointing_config)

    # Keeps track of loss and accuracy after each epoch
    checkpoint_tracker = []

    # Train
    training_accuracies = np.zeros(NUM_EPOCHS)
    training_losses = np.zeros(NUM_EPOCHS)
    dev_accuracies = np.zeros(NUM_EPOCHS)
    dev_losses = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        model.train(input_fn=trainInputFn,
                    steps=num_steps_in_epoch)

        eval_of_train = model.evaluate(trainInputFn)       
        eval_of_dev = model.evaluate(devInputFn)

        latest_train_acc = np.asscalar(eval_of_train['accuracy'])
        latest_train_loss = np.asscalar(eval_of_train['loss'])
        latest_dev_acc = np.asscalar(eval_of_dev['accuracy'])
        latest_dev_loss = np.asscalar(eval_of_dev['loss'])
        print("Epoch: {:02d}/{} => Train Loss: {:.3f}, Dev Loss: {:.3f}. Train Acc: {:.3f}, Dev Acc: {:.3f}".format(epoch+1, NUM_EPOCHS, latest_train_loss, latest_dev_loss, latest_train_acc, latest_dev_acc))
        
        training_accuracies[epoch] = latest_train_acc
        training_losses[epoch] = latest_train_loss
        dev_accuracies[epoch] = latest_dev_acc
        dev_losses[epoch] = latest_dev_loss

        checkpoint_tracker.append((model.latest_checkpoint(), latest_dev_loss, latest_dev_acc))
        # Keep only as much data as we have checkpoints saved
        checkpoint_tracker = checkpoint_tracker[-NUM_CHECKPOINTS_SAVED:]

        if shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, epoch):
            break

    best_cp_index = bestCheckpointIndex(checkpoint_tracker)
    cp_path, cp_dev_loss, cp_dev_acc = checkpoint_tracker[best_cp_index]

    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_minutes = int((elapsed_time - (elapsed_hours * 3600)) / 60)
    results['accurate_runs'].append((cp_dev_acc, 'dev_acc: {:.3f} batch_size: {} learn_rate: {:.4f} drop_rate: {:.3f} l2_scale: {:.3f} cp_path: {} ({} epochs, {} hours {} mins)'.format(cp_dev_acc, batch_size, learning_rate, drop_rate, l2_scale, cp_path, epoch+1, elapsed_hours, elapsed_minutes)))

    print('\nRUN RESULTS: {}'.format(results['accurate_runs'][-1][1]))

    # Restore the best model from the saved checkpoints and re-evaluate dev on it using model.predict().
    predicted_classes = list(model.predict(devInputFn, checkpoint_path=cp_path))

    # Print the filenames of the fits we got wrong, for further debugging/analysis
    # Class labels need to be 0 or 1, not onehot.
    truth_labels = dev_labels.tolist()
    print('\nDEV EXAMPLES WE GOT WRONG:')
    for i in range(len(truth_labels)):
        dev_filename = dev_filenames[i]
        if (truth_labels[i] != predicted_classes[i]):
            if not dev_filename in results['incorrect']:
                results['incorrect'][dev_filename] = 1
            else:
                results['incorrect'][dev_filename] = results['incorrect'][dev_filename] + 1
            print('Predicted {}, true {}: {}'.format(predicted_classes[i], truth_labels[i], dev_filename))

    # Print the dev confusion matrix using the best checkpointed model.
    with tf.Session() as sess:
        confusion_matrix = tf.confusion_matrix(truth_labels, predicted_classes)
        matrix_to_print = sess.run(confusion_matrix)
        print('\nCONFUSION MATRIX:')
        print(matrix_to_print)

    # Print network info. This uses latest checkpoint, so delete extra checkpoints after this.
    printNetworkInfo(model)
    deleteExtraCheckpoints(cp_path)

    print('\nEND LEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}, L2_SCALE: {}'.format(learning_rate, batch_size, drop_rate, l2_scale))
    print('==========================================================================')


def printResults(results):
    results['accurate_runs'].sort(reverse=True)
    print('\nHYPERPARAMETERS WITH HIGHEST DEV ACCURACY:')
    for acc, params in results['accurate_runs']:
        print(params)
    most_wrong = sorted(results['incorrect'].items(), key=lambda kv: kv[1], reverse=True)
    print('\nSPECTRA THAT WERE PREDICTED INCORRECTLY THE MOST:')
    for name, numwrong in most_wrong:
        print('{}: {}'.format(name, numwrong))


if __name__=='__main__':
    print('This version of TF built with CUDA? {}'.format(tf.test.is_built_with_cuda()))
    print(device_lib.list_local_devices())
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Don't log Info and Warning level messages (too many of them)
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # hyperparameters should be optimized using the dev set.
    # High learning rates cause error/accuracy to jump around and interfere with early stopping,
    # but strong regularization seems to help.
    LEARNING_RATES = [4e-4, 2e-4]
    BATCH_SIZES = [32]
    DROP_RATES = [0.01, 0.005]
    L2_SCALES = [0.01, 0.003]

    if ARGS.adaptive:
        print('Will use adaptive gaussian smoothing using ivar and flux.')

    print('\nStarting at datetime: {}'.format(str(datetime.now())))

    train_metadata = datasetFromFitsDirectory(TRAIN_DATA_DIR, ARGS.fraction)
    dev_metadata = datasetFromFitsDirectory(DEV_DATA_DIR, 1.0)
    print('Finished loading data at datetime: {}'.format(str(datetime.now())))

    results = {'incorrect': {},
               'accurate_runs' : [],
              }
    combinations = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROP_RATES, L2_SCALES))
    random.shuffle(combinations)
    for (learn_rate, batch, drop_rate, l2_scale) in combinations:
        trainModel(train_metadata, dev_metadata, learn_rate, batch, drop_rate, l2_scale, results)

    printResults(results)

    print('Finished at datetime: {}'.format(str(datetime.now())))
