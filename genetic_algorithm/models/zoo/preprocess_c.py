# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import ReLU, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial

import random
import math
import sys


def get_parse_example_func(target_size=(224,224,3), num_classes=10):
    resize = resize_repeat(target_size=tuple(target_size), training=False)
    one_hot = partial(tf.one_hot, depth=num_classes)
    def _parse_example(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = resize(x)
        y = one_hot(y)
        return x,y
    return _parse_example


#     def resize(self):
# def preprocess_data(data: tf.data.Dataset, target_size=None, num_classes=None, batch_size=1, buffer_size=1024): #class_encoder=None):
#     parse_example = get_parse_example_func(target_size=target_size, num_classes=num_classes) #class_encoder=class_encoder)
#     return data.map(lambda x,y: parse_example(x, y), num_parallel_calls=-1) \
#                 .shuffle(buffer_size) \
#                 .batch(batch_size) \
#                 .prefetch(-1)

class Preprocess:
    ''' Preprocess base (super) class for Composable Models '''

    def __init__(self):
        """ Constructor
        """
        pass

    ###
    # Preprocessing
    ###

    def normalization(self, x_train, x_test=None, centered=False):
        """ Normalize the input
            x_train : training images
            y_train : test images
        """
        if x_train.dtype == np.uint8:
            if centered:
                x_train = ((x_train - 1) / 127.5).astype(np.float32)
                if x_test is not None:
                    x_test  = ((x_test  - 1) / 127.5).astype(np.float32)
            else:
                x_train = (x_train / 255.0).astype(np.float32)
                if x_test is not None:
                    x_test  = (x_test  / 255.0).astype(np.float32)
        return x_train, x_test

    def standardization(self, x_train, x_test=None):
        """ Standardize the input
            x_train : training images
            x_test  : test images
        """
        self.mean = np.mean(x_train)
        self.std  = np.std(x_train)
        x_train = ((x_train - self.mean) / self.std).astype(np.float32)
        if x_test is not None:
            x_test  = ((x_test  - self.mean) / self.std).astype(np.float32)
        return x_train, x_test

    def label_smoothing(self, y_train, n_classes, factor=0.1):
        """ Convert a matrix of one-hot row-vector labels into smoothed versions. 
            y_train  : training labels
            n_classes: number of classes
            factor   : smoothing factor (between 0 and 1)
        """
        if 0 <= factor <= 1:
            # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
            y_train *= 1 - factor
            y_train += factor / n_classes
        else:
            raise Exception('Invalid label smoothing factor: ' + str(factor))
        return y_train

    
    
    
## class-Weighted label smoothing    
# def class_weight(labels_dict,mu=0.15):
#     total = np.sum(labels_dict.values())
#     keys = labels_dict.keys()
#     weight = dict()
# for i in keys:
#         score = np.log(mu*total/float(labels_dict[i]))
#         weight[i] = score if score > 1 else 1
# return weight
# # random labels_dict
# labels_dict = df[target_Y].value_counts().to_dict()
# weights = class_weight(labels_dict)