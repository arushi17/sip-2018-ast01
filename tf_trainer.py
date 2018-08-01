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

NUM_EPOCHS = 100  # max epochs
SHUFFLE_BUFFER = 10000

NUM_CHECKPOINTS_SAVED = 20
# For early stopping, we look for an increase in dev error relative to the best model so far,
# expressed as a percentage.
ES_MIN_GENERALIZATION_LOSS = 1.0
ES_MIN_PROGRESS_QUOTIENT = 0.5

# from tensorflow.layers import ...
# We learn weights for these:
conv1d = tf.layers.conv1d
dense = tf.layers.dense
dropout = tf.layers.dropout
# No weights for these:
max_pooling1d = tf.layers.max_pooling1d
flatten = tf.layers.flatten
relu = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer

parser = argparse.ArgumentParser(description='Loads spectra from fits files, trains model.')
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
def small1dcnn(x, num_classes, is_training, drop_rate, l2_scale):
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

    net = flatten(net, name='pool3d_flattened')
    # 320
    net = dense(net, 64, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


# Tensorflow network architecture definition
def medium1dcnn(x, num_classes, is_training, drop_rate, l2_scale):
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

    net = flatten(net, name='pool3d_flattened')
    # 256
    net = dense(net, 64, activation=relu, name='fc1', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    net = dropout(net, rate=drop_rate, training=is_training, name='fc1_dropout')

    logits = dense(net, num_classes, name='fc2', kernel_initializer=xavier_init, kernel_regularizer=regularizer(scale=l2_scale))
    return logits, x


# Needs to follow this signature, per tf.estimator. Should support TRAIN, EVAL, PREDICT
# Assumes labels follows the onehot style.
def modelFn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits, x = medium1dcnn(x=features['features'], 
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


# Returns the best checkpoint_tracker entry if early stopping criteria are met
# (that is, we should stop training now). Otherwise returns None.
# See https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
# Using the second approach in this paper, which waits for the training rate to
# plateau before considering early stopping.
# TODO: Better early stopping. See https://github.com/tensorflow/tensorflow/issues/18394
# (the feature will be released in TF 1.10).
def shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, checkpoint_tracker, epoch):
    if epoch < 10 or training_accuracies[epoch] < dev_accuracies[epoch]:
        # Clearly too early to stop training
        return None
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
        # TODO: Remove. Debugging
        print('Progress quotient: {:.3f}, generalization loss: {:.3f}, not stopping.'.format(progress_quotient, generalization_loss))
        return None

    # Should stop now. Find the best saved checkpoint using accuracy (not loss).
    print('Progress quotient: {:.3f}, generalization loss: {:.3f}, stopping early!'.format(progress_quotient, generalization_loss))
    cp_max_acc = 0
    i = 0
    best_i = epoch
    for cp_file, dev_loss, dev_acc in checkpoint_tracker:
        if dev_acc > cp_max_acc:
            cp_max_acc = dev_acc
            best_i = i
        i = i + 1

    return checkpoint_tracker[best_i]


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
    num_steps_in_epoch = int(len(train_metadata['labels'])/batch_size)
    print('Number of steps in epoch: {}'.format(num_steps_in_epoch))

    # TODO: Looks like a bug: Estimator is producing one extra checkpoint per requested
    # checkpoint (one step after the requested one). So double the number of checkpoints.
    my_checkpointing_config = tf.estimator.RunConfig(
        save_summary_steps = 1000,  # Write to events.out.tfevents file less often than default 100.
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
        print("Epoch: {}/{}\nDev Loss:      {:.3f}   |   Dev Accuracy:      {:.3f}"
              "".format(epoch+1, NUM_EPOCHS, latest_dev_loss, latest_dev_acc))
        print("Training Loss: {:.3f}   |   Training Accuracy: {:.3f}"
              "".format(latest_train_loss, latest_train_acc))
        
        training_accuracies[epoch] = latest_train_acc
        training_losses[epoch] = latest_train_loss
        dev_accuracies[epoch] = latest_dev_acc
        dev_losses[epoch] = latest_dev_loss

        checkpoint_tracker.append((model.latest_checkpoint(), latest_dev_loss, latest_dev_acc))
        # Keep only as much data as we have checkpoints saved
        checkpoint_tracker = checkpoint_tracker[-NUM_CHECKPOINTS_SAVED:]

        best_checkpoint = shouldStopEarly(training_accuracies, training_losses, dev_accuracies, dev_losses, checkpoint_tracker, epoch)
        if best_checkpoint:
            break

    if not best_checkpoint:
        # Exceeded NUM_EPOCHS. TODO: Select the best from saved epochs.
        best_checkpoint = checkpoint_tracker[-1]

    cp_path, cp_dev_loss, cp_dev_acc = best_checkpoint
    results['accurate_runs'].append((cp_dev_acc, 'dev_acc: {:.3f} batch_size: {} learn_rate: {:.4f} drop_rate: {:.2f} l2_scale: {:.2f} cp_path: {}'.format(cp_dev_acc, batch_size, learning_rate, drop_rate, l2_scale, cp_path)))

    print('\nRUN RESULTS: {}'.format(results['accurate_runs'][-1][1]))

    deleteExtraCheckpoints(cp_path)

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

    elapsed_time = time.time() - start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_minutes = int((elapsed_time - (elapsed_hours * 3600)) / 60)
    print('\nEND LEARNING_RATE: {}, BATCH_SIZE: {}, DROP_RATE: {}, L2_SCALE: {} ({} epochs, {} hours {} mins)'.format(learning_rate, batch_size, drop_rate, l2_scale, epoch, elapsed_hours, elapsed_minutes))
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
    LEARNING_RATES = [3e-4, 1e-3]
    BATCH_SIZES = [32]
    DROP_RATES = [0.2, 0.25]
    L2_SCALES = [0.05, 0.1]

    if ARGS.adaptive:
        print('Will use adaptive gaussian smoothing using ivar and flux.')

    print('\nStarting at datetime: {}'.format(str(datetime.now())))

    train_metadata = datasetFromFitsDirectory(TRAIN_DATA_DIR, ARGS.fraction)
    dev_metadata = datasetFromFitsDirectory(DEV_DATA_DIR, ARGS.fraction)
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
