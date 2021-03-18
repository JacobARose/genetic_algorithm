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

import random
import math
import sys
from typing import Tuple
from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial

from omegaconf import DictConfig
from genetic_algorithm.datasets.plant_village import ClassLabelEncoder, load_and_preprocess_data

# from genetic_algorithm.datasets import pnas
from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.data import extant, pnas
import wandb


from .layers_c import Layers
from .preprocess_c import Preprocess
from .pretraining_c import Pretraining
from .hypertune_c import HyperTune
from .training_c import Training

class Dataset(object):
    ''' Dataset base (super) class for Models '''

    def __init__(self):
        """ Constructor
        """
        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None
        self.n_classes = 0
        self._dataset_initialized = False

    @property
    def data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def load_data(self, train, test=None, std=False, onehot=False, smoothing=0.0):
        """ Load in memory data
            train: expect form: (x_train, y_train)
        """
        self.x_train, self.y_train = train
        if test is not None:
            self.x_test, self.y_test   = test
        if std:
            self.x_train, self.x_test = self.standardization(self.x_train, self.x_test)

        if self.y_train.ndim == 2:
            self.n_classes = np.max(self.y_train) + 1
        else:
            self.n_classes = self.y_train.shape[1]
        if onehot:
            self.y_train = to_categorical(self.y_train, self.n_classes)
            self.y_test  = to_categorical(self.y_test, self.n_classes)
        if smoothing > 0.0:
            self.y_train = self.label_smoothing(self.y_train, self.n_classes, smoothing)

    def cifar10(self, epochs=10, decay=('cosine', 0), save: str=None):
        """ Train on CIFAR-10
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = self.standardization(x_train, x_test)
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test, 10)
        y_train = self.label_smoothing(y_train, 10, 0.1)

        # compile the model
        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train, save=save)

        lr, batch_size = self.random_search(x_train, y_train, x_test, y_test, save=save)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay, save=save)
        self.evaluate(x_test, y_test)

    def cifar100(self, epochs=20, decay=('cosine', 0), save: str=None):
        """ Train on CIFAR-100
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_test = self.normalization(x_train, x_test)
        y_train = to_categorical(y_train, 100)
        y_test  = to_categorical(y_test, 100)
        y_train = self.label_smoothing(y_train, 100, 0.1)
        
        
        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train, save=save)

        lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test, save=save)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay, save=save)
        self.evaluate(x_test, y_test)

    def coil100(self, epochs=20, decay=('cosine', 0), save: str=None):
        """
        Columbia University Image Library (COIL-100)
        """
        # Get TF.dataset generator for COIL100
        train, info = tfds.load('coil100', split='train', shuffle_files=True, with_info=True, as_supervised=True)
        n_classes = info.features['label'].num_classes
        n_images = info.splits['train'].num_examples
        input_shape = info.features['image'].shape

        # Get the dataset into memory
        train = train.shuffle(n_images).batch(n_images)
        for images, labels in train.take(1):
            pass
        images = np.asarray(images)
        images, _ = self.standardization(images, None)
        labels = to_categorical(np.asarray(labels), n_classes)

        # split the dataset into train/test
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train, save=save)

        lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test, save=save)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay, save=save)
        self.evaluate(x_test, y_test)

        
        
    def plant_village(self, target_size=[256,256], batch_size=32, epochs=20, decay=('cosine', 0), save: str=None, allow_resume: bool=False): #, skip_full_run: bool=False):
        """
        Plant Village leaf disease dataset (2016)
        """
        

        if not self._dataset_initialized:
            self.plant_village_init(target_size=target_size, batch_size=batch_size)

        self.set_wandb_env_vars(project='ResNet50_v2', group=f'resnet50_v2-plant_village-res{target_size[0]}')

        print('compiling')        
        self.compile(loss='categorical_crossentropy', metrics=['acc', 'recall','precision'])
        
        with wandb.init(reinit=True, job_type='warmup', tags=['warmup']) as run:
            print('initiating warmup')
            self.warmup(self.x_train, self.y_train, save=save, allow_resume=allow_resume)
            
#         with wandb.init(reinit=True, job_type='grid_search', tags=['grid_search']) as run:
        print('initiating grid_search. dir(self):\n',dir(self))
        lr, batch_size = self.grid_search(self.x_train, self.y_train, self.x_val, self.y_val, save=save,
                                          batch_range=[16, 32], allow_resume=allow_resume)
        
        with wandb.init(reinit=True, job_type='training', tags=['training']) as run:
            print('initiating training. dir(self):\n')
            self.training(self.x_train, self.y_train, validation_data = (self.x_val, self.y_val), epochs=epochs, batch_size=batch_size,
                          lr=lr, decay=decay, save=save)
            
            print('initiating evaluate. dir(self):\n',dir(self))
            result = self.evaluate(self.x_test, self.y_test)
            run.log({'test_results':result})

            
            
    def plant_village_init(self, target_size=[256,256], batch_size=32):
        """Plant Village leaf disease dataset (2016)
        
        Run this function to initialize the dataset without performing any model training or evaluation.
        
        
        # TODO Introduce a wandb dataset artifact logging option here
        """
        
        data_config = DictConfig({
                            'load':{'dataset_name':'plant_village',
                                    'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
                                    'data_dir':'/media/data/jacob/tensorflow_datasets'},
                            'preprocess':{'batch_size':batch_size,
                                          'target_size':target_size}
                                })
        
        data, self.class_encoder = load_and_preprocess_data(data_config)

        train_dataset = data['train']
        val_dataset = data['val']
        test_dataset = data['test']
        
        batch_size = data_config.preprocess.batch_size
        steps_per_epoch = len(data['train'])
        validation_steps = len(data['val'])
        test_steps = len(data['test'])
        num_classes = train_dataset.element_spec[1].shape[1]
        
        
        num_samples = batch_size*steps_per_epoch
        train_dataset = train_dataset.unbatch().shuffle(num_samples).batch(num_samples)
        for images, labels in train_dataset.take(1):
            pass
        self.x_train, self.y_train = np.asarray(images), np.asarray(labels)

        print(f'Loaded {num_samples} samples into memory from plant_village train')
        num_samples = batch_size*validation_steps
        val_dataset = val_dataset.unbatch().batch(num_samples)
        for images, labels in val_dataset.take(1):
            pass
        self.x_val, self.y_val = np.asarray(images), np.asarray(labels)
        print(f'Loaded {num_samples} samples into memory from plant_village val')

        num_samples = batch_size*test_steps
        test_dataset = test_dataset.unbatch().batch(num_samples)
        for images, labels in test_dataset.take(1):
            pass
        self.x_test, self.y_test = np.asarray(images), np.asarray(labels)
        print(f'Loaded {num_samples} samples into memory from plant_village test')
        
        self._dataset_initialized = True

        
#     def pnas(self, target_size=[256,256], batch_size=32, epochs=20, decay=('cosine', 0), initial_frozen_layers: Tuple[int]=None,  threshold=100, save: str=None, allow_resume: bool=False, search_frozen_layers=False): #, skip_full_run: bool=False):

            
            
    def fit_paleoai_dataset(self,
                            dataset_name: str='PNAS',
                            target_size=[256,256],
                            batch_size=32,
                            grayscale=False,
                            num_epochs: int=20,
                            num_warmup_epochs: int=5,
                            num_epochs_per_hp_search: int=4,
                            decay=('cosine', 0),
                            initial_frozen_layers: Tuple[int]=None,
                            threshold=100,
                            val_split: float=0.2,
                            save: str=None,
                            allow_resume: bool=False, 
                            search_frozen_layers=False,
                            seed: int=None):
        
        if dataset_name == 'PNAS':
            self.pnas(target_size=target_size,
                      grayscale=grayscale,
                      batch_size=batch_size,
                      threshold=threshold,
                      val_split=val_split,
                      seed=seed)
                      
        elif dataset_name == 'Extant':
            self.extant(target_size=target_size,
                        batch_size=batch_size,
                        seed=seed)
            
            
        self.set_wandb_env_vars(project='ResNet50_v2',
                                group=f'resnet50_v2-{dataset_name}-res{target_size[0]}')

        print('compiling')
        metrics = self.get_metrics()
        self.compile(loss='categorical_crossentropy', metrics=metrics)        
        
        with wandb.init(reinit=True, job_type='warmup', tags=['warmup']) as run:
            print('initiating warmup')
            self.warmup(self.x_train,
                        self.y_train,
                        epochs=num_warmup_epochs,
                        frozen_layers=initial_frozen_layers,
                        metrics=metrics,
                        save=save,
                        allow_resume=allow_resume)
            
#         with wandb.init(reinit=True, job_type='grid_search', tags=['grid_search']) as run:
        print('initiating grid_search.')
        if search_frozen_layers:
            frozen_layers_hp_range = [(0,-4),(0,-17), (0,-28), (0,-40)]
        else:
            frozen_layers_hp_range = [None]
#         lr, batch_size, best_loss
        best_hp = self.grid_search(self.x_train, self.y_train,
                                   self.x_val, self.y_val,
                                   epochs=num_epochs_per_hp_search,
                                   num_train_samples=self.num_samples['train'],
                                   num_test_samples=self.num_samples['val'],
                                   lr_range=[0.0001, 0.001, 0.01],
                                   batch_range=[16, 32],
                                   frozen_layers_hp_range=frozen_layers_hp_range,
                                   metrics=metrics,
                                   save=save,
                                   allow_resume=allow_resume)
        
        batch_size = best_hp.batch_size
        lr = best_hp.lr
        frozen_layers = best_hp.frozen_layers
        
        steps_per_epoch=np.floor(self.num_samples['train']/batch_size)
        validation_steps=np.floor(self.num_samples['val']/batch_size)
        test_steps=np.floor(self.num_samples['test']/batch_size)
        with wandb.init(reinit=True, job_type='training', tags=['training']) as run:
            print('initiating training.\n')
            self.training(self.x_train,
                          self.y_train,
                          validation_data = (self.x_val, self.y_val),
                          epochs=num_epochs,
                          batch_size=batch_size,
                          num_train_samples=self.num_samples['train'],
                          num_test_samples=self.num_samples['val'],
                          lr=lr,
                          decay=decay,
                          frozen_layers=frozen_layers,
                          metrics=metrics,
                          save=save)
            
#             print('initiating evaluate. dir(self):\n',dir(self))
            result = self.evaluate(self.x_test,
                                   self.y_test,
                                   steps=test_steps)
            run.log({'test_results':result,
                     'best_hp':best_hp})
            
            
    def pnas(self,
             target_size=[256,256],
             grayscale=False,
             batch_size=32,
             threshold=100,
             validation_split: float=0.2,
             seed: int=None):
        """
        PNAS leaf disease dataset (2016)
        """

        if not self._dataset_initialized:
            self.pnas_init(target_size=target_size, grayscale=grayscale, batch_size=batch_size, threshold=threshold, validation_split=validation_split, seed=seed)
            
            
    def pnas_init(self,
                  target_size=[256,256],
                  grayscale=False,
                  batch_size=32,
                  threshold=100,
                  validation_split=0.2,
                  seed: int=None):
        """
        PNAS leaf disease dataset (2016)
        
        Run this function to initialize the dataset without performing any model training or evaluation.
        
        # TODO Introduce a wandb dataset artifact logging option here

        Args:
            target_size (list, optional): [description]. Defaults to [256,256].
            batch_size (int, optional): [description]. Defaults to 32.
            threshold (int, optional): [description]. Defaults to 100.
            validation_split (float, optional): [description]. Defaults to 0.2.
            seed (int, optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        
        # data_config = DictConfig({
        #                     'load':{
        #                         'dataset_name':'PNAS'
        #                         },
        #                     'preprocess':{
        #                                   'batch_size':batch_size,
        #                                   'target_size':target_size,
        #                                   'threshold':threshold
        #                                   },
        #                     'augment':   {
        #                                   'augmix_batch_size':max([batch_size*2, 48])
        #                                   }
        #                   })        
        # data, self.class_encoder, self.preprocess_data = pnas.load_and_preprocess_data(data_config)

        data, self.class_encoder = pnas.get_supervised(target_size=target_size,
                                                       grayscale=grayscale,
                                                       batch_size=batch_size,
                                                       val_split=validation_split,
                                                       threshold=threshold,
                                                       seed=seed,
                                                       return_label_encoder=True)
        
        train_data = data[0]#.map(lambda sample: (sample['x'], sample['y']))
        val_data = data[1]#.map(lambda sample: (sample['x'], sample['y']))
        test_data = data[2]#.map(lambda sample: (sample['x'], sample['y']))
        
        steps_per_epoch = len(train_data) #data['train'])
        validation_steps = len(val_data) #data['val'])
        test_steps = len(test_data) #data['test'])
        # num_classes = train_data.element_spec[0].shape[1]

        # num_samples = batch_size*steps_per_epoch
        # train_dataset = train_dataset.shuffle(num_samples).batch(num_samples)
        # self.train_data = next(iter(train_dataset.take(1)))

        # print(f'Loaded {num_samples} samples into memory from PNAS train')
        # num_samples = batch_size*validation_steps
        # val_dataset = val_dataset.batch(num_samples)
        # self.val_data = next(iter(val_dataset.take(1)))
        # print(f'Loaded {num_samples} samples into memory from PNAS val')

        # num_samples = batch_size*test_steps
        # test_dataset = test_dataset.batch(num_samples)
        # self.test_data = next(iter(test_dataset.take(1)))
        # print(f'Loaded {num_samples} samples into memory from PNAS test')
        
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.x_train, self.x_val, self.x_test = train_data, val_data, test_data
        self.y_train, self.y_val, self.y_test = None, None, None

        self.num_samples = {'train':batch_size*steps_per_epoch,
                            'val':batch_size*validation_steps,
                            'test':batch_size*test_steps}

        # if len(self.train_data)==3:
        #     self.catalog_numbers_train, self.x_train, self.y_train = self.train_data
        #     self.catalog_numbers_val, self.x_val, self.y_val = self.val_data
        #     self.catalog_numbers_test, self.x_test, self.y_test = self.test_data
        # elif len(self.train_data)==2:
        #     self.x_train, self.y_train = self.train_data
        #     self.x_val, self.y_val = self.val_data
        #     self.x_test, self.y_test = self.test_data
        # else:
        #     raise Exception('Invalid train_dataset format')
            
        self._dataset_initialized = True
        
        
        
        
        
    def extant(self,
               target_size=[256,256],
               batch_size=32,
               seed: int=None):
        """
        Extant leaf disease dataset (2021)
        """

        if not self._dataset_initialized:
            self.extant_init(target_size=target_size, batch_size=batch_size, seed=seed)
            
            
    def extant_init(self, target_size=[256,256], batch_size=32, seed: int=None):
        """Extant leaf disease dataset (2021)
        
        Run this function to initialize the dataset without performing any model training or evaluation.
        
        
        # TODO Introduce a wandb dataset artifact logging option here
        """
        
        train_data, val_data, test_data = get_dataset(dataset='extant_sup',
                                                      batch_size=batch_size,
                                                      target_size=(*target_size,3),
                                                      seed=seed)
        
#         train_data = train_data.map(preprocess_input, num_parallel_calls=-1)
#         val_data = val_data.map(preprocess_input, num_parallel_calls=-1)
#         test_data = test_data.map(preprocess_input, num_parallel_calls=-1)
        
#         self.num_samples = {'train':batch_size*len(train_data),
#                             'val':batch_size*len(val_data),
#                             'test':batch_size*len(test_data)}

        self.num_samples = extant.NUM_SAMPLES
    
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.x_train, self.x_val, self.x_test = train_data, val_data, test_data
        self.y_train, self.y_val, self.y_test = None, None, None
            
        self._dataset_initialized = True
        
        

        
        
        
        
#         num_samples = batch_size*steps_per_epoch
#         train_dataset = train_dataset.unbatch().shuffle(num_samples).batch(num_samples)
#         for images, labels in train_dataset.take(1):
#             pass
#         self.x_train, self.y_train = np.asarray(images), np.asarray(labels)

#         print(f'Loaded {num_samples} samples into memory from PNAS train')
#         num_samples = batch_size*validation_steps
#         val_dataset = val_dataset.unbatch().batch(num_samples)
#         for images, labels in val_dataset.take(1):
#             pass
#         self.x_val, self.y_val = np.asarray(images), np.asarray(labels)
#         print(f'Loaded {num_samples} samples into memory from PNAS val')

#         num_samples = batch_size*test_steps
#         test_dataset = test_dataset.unbatch().batch(num_samples)
#         for images, labels in test_dataset.take(1):
#             pass
#         self.x_test, self.y_test = np.asarray(images), np.asarray(labels)
#         print(f'Loaded {num_samples} samples into memory from PNAS test')
        
#         self._dataset_initialized = True

        
        
        
        
        
        
        
        
        
        
        
        
        
#         data_config = DictConfig({
#                             'load':{'dataset_name':'plant_village',
#                                     'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
#                                     'data_dir':'/media/data/jacob/tensorflow_datasets'},
#                             'preprocess':{'batch_size':batch_size,
#                                           'target_size':target_size}
#                                 })
        
#         data, self.class_encoder = load_and_preprocess_data(data_config)

#         train_dataset = data['train']
#         val_dataset = data['val']
#         test_dataset = data['test']
        
#         batch_size = data_config.preprocess.batch_size
#         steps_per_epoch = len(data['train'])
#         validation_steps = len(data['val'])
#         test_steps = len(data['test'])
#         num_classes = train_dataset.element_spec[1].shape[1]
        
        
# #         x_train, y_train = next(iter(train_dataset.unbatch().batch(batch_size*steps_per_epoch).take(1)))

#         # Get the dataset into memory
#         num_samples = batch_size*steps_per_epoch
#         train_dataset = train_dataset.unbatch().shuffle(num_samples).batch(num_samples)
#         for images, labels in train_dataset.take(1):
#             pass
#         self.x_train, self.y_train = np.asarray(images), np.asarray(labels)
# #         input_shape = self.x_train.shape[1:]
#         print(f'Loaded {num_samples} samples into memory from plant_village train')
#         # Get the dataset into memory
#         num_samples = batch_size*validation_steps
#         val_dataset = val_dataset.unbatch().batch(num_samples)
#         for images, labels in val_dataset.take(1):
#             pass
#         self.x_val, self.y_val = np.asarray(images), np.asarray(labels)
#         print(f'Loaded {num_samples} samples into memory from plant_village val')

#         # Get the dataset into memory
#         num_samples = batch_size*test_steps
#         test_dataset = test_dataset.unbatch().batch(num_samples)
#         for images, labels in test_dataset.take(1):
#             pass
#         self.x_test, self.y_test = np.asarray(images), np.asarray(labels)
#         print(f'Loaded {num_samples} samples into memory from plant_village test')