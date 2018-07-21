#!/usr/bin/env python3
""" 
1D CNN classifier.
Uses Estimator (new eager execution style).

To run:
python3 -u tf_trainer.py [-a True] 2>&1 | tee training.log

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
import tensorflow as tf
from tensorflow.python.client import device_lib

HOME_DIR = expanduser("~") + '/ast01'
# where is the dataset?
TRAIN_DATA_DIR = HOME_DIR + '/training_data'
TEST_DATA_DIR = HOME_DIR + '/test_data'
STAR_DIR = 'star'
NONSTAR_DIR = 'nonstar'
# Use onehot style
STAR_LABEL = [0, 1]
NONSTAR_LABEL = [1, 0]

# If specified, checkpoints are automatically saved to this dir.
# This should be a commandline arg so different runs don't clash on the model dir.
# Be careful that different training runs (different hyperparameters) can load the
# last checkpointed model from this dir and 'continue' training!
#MODEL_DIR = HOME_DIR + '/models'
MODEL_DIR = None

SHUFFLE_BUFFER = 10000
NUM_EPOCHS = 100  # max epochs

# from tensorflow.layers import ...
dropout = tf.layers.dropout
conv1d = tf.layers.conv1d
max_pooling1d = tf.layers.max_pooling1d
dense = tf.layers.dense
flatten = tf.layers.flatten
relu = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer

parser = argparse.ArgumentParser(description='Loads spectra from fits files, trains model.')
parser.add_argument('-s', '--sigma', type=int, default=0, help='Sigma to be used in gaussian smoothing. 0 if no smoothing.') 
parser.add_argument('-a', '--adaptive', type=bool, default=False, help='Use adaptive gaussian smoothing that combines ivar and flux into one series.') 
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
    else:
        # Make both raw ivar and flux as channels.
        flux = cleanValues(spec.flux)
        flux = np.float32(standardize(flux))

        ivar = cleanValues(spec.ivar)
        ivar = np.sqrt(ivar)
        ivar = limitOutliers(ivar, 2.5)
        ivar = np.float32(standardize(ivar))

        if flux.shape != ivar.shape:
            raise ValueError(filepath + ': flux.shape: ' + flux.shape + ', ivar.shape: ' + ivar.shape)
        # Channels should be last, for most tf layers
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
def small1dcnn(x, num_classes, is_training, drop_rate):
    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    net = conv1d(x, 8, 61, strides=10, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init())
    # conv1d reshapes output to [batch, out_width, out_channels]
    # 820 x 8
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    # 410 x 8

    net = conv1d(net, 16, 61, 10, data_format='channels_last', activation=relu, name='conv2')
    # 41 x 16
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    # 20 x 16
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')
    # 20 x 16

    net = flatten(net, name='pool3d_flattened')
    # 320
    net = dense(net, 64, activation=relu, name='fc1')
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2')
    return logits, x


# Tensorflow network architecture definition
def medium1dcnn(x, num_classes, is_training, drop_rate):
    # Channels refers to flux/ivar (2 channels), or if adaptive smoothing, a single channel.
    net = conv1d(x, 8, 30, strides=5, data_format='channels_last', activation=relu, name='conv1', kernel_initializer=xavier_init())
    # conv1d reshapes output to [batch, out_width, out_channels]
    # 1633 x 8
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool1')
    # 820 x 8

    net = conv1d(net, 16, 30, 5, data_format='channels_last', activation=relu, name='conv2')
    # 164 x 16
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool2')
    # 82 x 16
    net = dropout(net, rate=drop_rate, training=is_training, name='pool2_dropout')
    # 82 x 16

    net = conv1d(net, 32, 30, 5, data_format='channels_last', activation=relu, name='conv3')
    # 16 x 32
    net = max_pooling1d(net, 2, 2, data_format='channels_last', name='pool3')
    # 8 x 32
    net = dropout(net, rate=drop_rate, training=is_training, name='pool3_dropout')
    # 8 x 32

    net = flatten(net, name='pool3d_flattened')
    # 256
    net = dense(net, 64, activation=relu, name='fc1')
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2')
    return logits, x


# Needs to follow this signature, per tf.estimator. Should support TRAIN, EVAL, PREDICT
# Assumes labels follows the onehot style.
def modelFn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits, x = medium1dcnn(x=features['features'], 
                            num_classes=params['num_classes'], 
                            is_training=is_training,
                            drop_rate=params['drop_rate'])
    y_hat = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.argmax(y_hat, axis=1))

    # https://www.tensorflow.org/get_started/custom_estimators#predict
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, 
        labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
        loss, global_step=tf.train.get_global_step(), 
        name='train_acc' if is_training else 'test_acc')

    # Assume onehot
    acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), 
                              predictions=tf.argmax(y_hat, axis=1),
                              name='train_acc' if is_training else 'test_acc')

    metrics = {'accuracy': acc}
    # Make available to tensorboard in TRAIN mode. http://localhost:6006
    tf.summary.scalar('accuracy', acc[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=tf.argmax(y_hat, axis=1),
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def trainModel(train_metadata, test_metadata, LEARNING_RATE, BATCH_SIZE, DROP_RATE, results):
    print('==========================================================================')
    print('\nLEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}'.format(LEARNING_RATE, BATCH_SIZE, DROP_RATE))
    # The training dataset needs to last for all epochs, for use with one_shot_iterator.
    train_dataset = train_metadata['dataset']
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).repeat(NUM_EPOCHS).batch(BATCH_SIZE)

    # Returns features, label each time it is called.
    def trainInputFn():
        return train_dataset.make_one_shot_iterator().get_next()

    # For the test set, we generally want to evaluate/predict the entire set.
    # Does the batch size even matter? conv1d gives dimension mismatch error without batching.
    test_dataset = test_metadata['dataset']
    test_dataset = test_dataset.batch(BATCH_SIZE)

    test_features = test_metadata['features']
    test_labels = test_metadata['labels']
    test_filenames = test_metadata['filenames']

    # Returns features, label each time it is called.
    def testInputFn():
        return test_dataset.make_one_shot_iterator().get_next()

    # Build
    model = tf.estimator.Estimator(
        modelFn,
        params={'learning_rate': LEARNING_RATE, 
                'batch_size': BATCH_SIZE,
                'drop_rate': DROP_RATE,
                'num_classes': 2},
        model_dir=MODEL_DIR)

    # Train
    validation_accuracies = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        model.train(input_fn=trainInputFn,
                    steps=len(train_metadata['labels'])/BATCH_SIZE)
                        # hooks=hooks)
        t = model.evaluate(trainInputFn)       
        v = model.evaluate(testInputFn)
        pad = len("Epoch: {}/{}".format(epoch, NUM_EPOCHS)) * " "
        print("\nEpoch: {}/{} |    Test Loss: {:.3f} |    Test Accuracy: {:.3f}"
              "".format(epoch+1, NUM_EPOCHS, np.asscalar(v['loss']), np.asscalar(v['accuracy'])))
        print(pad + " |Training Loss: {:.3f} | Training Accuracy: {:.3f}"
              "".format(np.asscalar(t['loss']), np.asscalar(t['accuracy'])))
        validation_accuracies[epoch] = v['accuracy']
        # TODO: Better early stopping. See https://github.com/tensorflow/tensorflow/issues/18394
        # (the feature will be released in TF 1.10).
        # Should save and restore the previous best model.
        if epoch > 8 and t['accuracy'] > 0.985 and t['accuracy'] > v['accuracy'] and np.mean(validation_accuracies[epoch-3:epoch+1]) < np.mean(validation_accuracies[epoch-5:epoch-1]):
            break

    results['accurate_runs'].append((v['accuracy'], 'batch: {} learn-rate: {:.4f} drop_rate: {:.2f}'.format(BATCH_SIZE, LEARNING_RATE, DROP_RATE)))

    # Print the test confusion matrix.
    # Class labels need to be 0 or 1, not onehot.
    truth_labels = test_labels.tolist()
    predictions = list(model.predict(testInputFn))
    predicted_classes = predictions

    # Print the filenames of the fits we got wrong, for further debugging/analysis
    print('\nTEST CASES WE GOT WRONG:')
    for i in range(len(truth_labels)):
        test_filename = test_filenames[i]
        if (truth_labels[i] != predicted_classes[i]):
            if not test_filename in results['incorrect']:
                results['incorrect'][test_filename] = 1
            else:
                results['incorrect'][test_filename] = results['incorrect'][test_filename] + 1
            print('Predicted {}, true {}: {}'.format(predicted_classes[i], truth_labels[i], test_filename))

    with tf.Session() as sess:
        confusion_matrix = tf.confusion_matrix(truth_labels, predicted_classes)
        matrix_to_print = sess.run(confusion_matrix)
        print('\nCONFUSION MATRIX:')
        print(matrix_to_print)

    print('\nLEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}'.format(LEARNING_RATE, BATCH_SIZE, DROP_RATE))
    print('==========================================================================')


def printResults(results):
    results['accurate_runs'].sort(reverse=True)
    print('HYPERPARAMETERS WITH HIGHEST TEST ACCURACY:')
    for acc, params in results['accurate_runs']:
        print('Accuracy: {:.3f}, {}'.format(acc, params))
    most_wrong = sorted(results['incorrect'].items(), key=lambda kv: kv[1], reverse=True)
    print('SPECTRA THAT WERE PREDICTED INCORRECTLY THE MOST:')
    for name, numwrong in most_wrong:
        print('{}: {}'.format(name, numwrong))


if __name__=='__main__':
    print('This version of TF built with CUDA? {}'.format(tf.test.is_built_with_cuda()))
    print(device_lib.list_local_devices())
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Don't log Info and Warning level messages (too many of them)
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # hyperparameters should be learned too.
    LEARNING_RATES = [1e-3, 3e-4]
    BATCH_SIZES = [32, 48]
    DROP_RATES = [0.2, 0.25]

    if ARGS.adaptive:
        print('Will use adaptive gaussian smoothing using ivar and flux.')

    print('\nStarting at datetime: {}'.format(str(datetime.now())))

    train_metadata = datasetFromFitsDirectory(TRAIN_DATA_DIR, ARGS.fraction)
    test_metadata = datasetFromFitsDirectory(TEST_DATA_DIR)
    print('Finished loading data at datetime: {}'.format(str(datetime.now())))

    results = {'incorrect': {},
               'accurate_runs' : [],
              }
    for (learn_rate, batch, drop_rate) in itertools.product(LEARNING_RATES, BATCH_SIZES, DROP_RATES):
        trainModel(train_metadata, test_metadata, learn_rate, batch, drop_rate, results)

    printResults(results)

    print('Finished at datetime: {}'.format(str(datetime.now())))
