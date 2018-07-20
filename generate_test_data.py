#!/usr/bin/env python3
# move about 20% of training data into test data
# last modified: 7/1/18

import random
import os
import shutil

SOURCE_DIR = os.path.expanduser("~") + '/ast01/training_data/'
TARGET_DIR = os.path.expanduser("~") + '/ast01/test_data/'

def move_from_dir(source, target):
    for file in os.listdir(source):
        filename = os.fsdecode(file)
        randint = random.randint(1,10)
        print(randint)
        if randint < 3:
            print('move to test data')
            shutil.move(source + '/' +  filename, target)

move_from_dir(SOURCE_DIR + '/star', TARGET_DIR + '/star')
move_from_dir(SOURCE_DIR + '/nonstar', TARGET_DIR + '/nonstar')
