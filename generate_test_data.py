#!/usr/bin/env python3
# move about 20% of training data into test data, and 20% into dev data.

import random
import os
import shutil

SOURCE_DIR = os.path.expanduser("~") + '/ast01/training_data/'
DEV_DATA_DIR = os.path.expanduser("~") + '/ast01/dev_data/'
TEST_DATA_DIR = os.path.expanduser("~") + '/ast01/test_data/'

def move_from_dir(source, test_dir, dev_dir):
    for file in os.listdir(source):
        filename = os.fsdecode(file)
        randnum = random.random()
        print(randnum)
        if randnum < 0.20:
            print('move to TEST data: {}'. format(filename))
            shutil.move(source + '/' +  filename, test_dir)
        elif randnum < 0.40:
            print('move to DEV data: {}'. format(filename))
            shutil.move(source + '/' +  filename, dev_dir)

move_from_dir(SOURCE_DIR + '/star', TEST_DATA_DIR + '/star', DEV_DATA_DIR + '/star')
move_from_dir(SOURCE_DIR + '/nonstar', TEST_DATA_DIR + '/nonstar', DEV_DATA_DIR + '/nonstar')
