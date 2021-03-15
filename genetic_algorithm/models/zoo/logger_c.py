

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
import os
from sklearn.model_selection import train_test_split

import random
import math
import sys

from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial

from omegaconf import DictConfig
from genetic_algorithm.datasets.plant_village import ClassLabelEncoder, load_and_preprocess_data
import wandb


from .layers_c import Layers
from .preprocess_c import Preprocess
from .pretraining_c import Pretraining
from .hypertune_c import HyperTune
from .training_c import Training

class Logger(object):
    ''' Logger base (super) class for Models '''

    def __init__(self):
        """ Constructor
        """
        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None
        self.n_classes = 0
        
    def set_wandb_env_vars(self, project='ResNet50_v2', group='', job_type=''):
        os.environ['WANDB_ENTITY'] = 'jrose'
        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_RUN_GROUP'] = group
        os.environ['WANDB_JOB_TYPE'] = job_type

        
        
        
        
        self.run.log({'Best chromosome':BestOrganism.chromosome}, commit=False)
        self.run.log({'population_size':len(fitness)}, commit=False)
        self.run.log({'Best fitness': fitness[0]}, commit=False)
        self.run.log({'Average fitness': sum(fitness)/len(fitness)}, commit=False)

        self.population[0].show()
        logger.info(f'BEST ORGANISM: {BestOrganism.name}')
#         k=16
        max_rows=10000
        if self.debug:
            max_rows = min(BestOrganism.config.output_size*30,1000)
#                 print('SKIPPING Evaluate & plotting due to debug flag')
#                 return BestOrganism
        model_dir = BestOrganism.config.model_dir or '.'
        model_path = os.path.join(model_dir,f'model-phase_{self.phase}.jpg')
        results_dir = os.path.join(model_dir,'results')
        os.makedirs(results_dir, exist_ok=True)
        chromosome = BestOrganism.chromosome
        test_data = BestOrganism.test_data
        model = BestOrganism.model

        if last:
            k=64
            model_path = os.path.join(model_dir,f'best-model-phase_{self.phase}.jpg')
            logger.info(f'Currently logging the model to {model_path}')
            tf.keras.utils.plot_model(model, to_file=model_path, expand_nested=True)
            model_img = cv2.imread(model_path)
            model_structure_image = wandb.Image(model_img, caption=f"Best Model phase_{self.phase}")

            run.log({"best_model": model_structure_image})#, commit=False)
            log_high_loss_examples(test_data,
                                   model,
                                   k=k,
                                   log_predictions=True,
                                   max_rows=max_rows,
                                   run=self.run)#,
#                                        commit=False)

            log_classification_report(test_data,
                                      model,
                                      data_split_name='test',
                                      class_encoder=self.class_encoder,
                                      run=self.run)

            logger.info(f'SAVING BEST MODEL: {BestOrganism.name}\nat {BestOrganism.model_dir}')
            BestOrganism.log_model_artifact(run=self.run)


        prevBestOrganism = generation.evaluate(last=True)
        keras.utils.plot_model(prevBestOrganism.model, to_file='best.png')
        wandb.log({"best_model": [wandb.Image('best.png', caption="Best Model")]})

        log_multiclass_metrics(test_data, 
                               model,
                               data_split_name='test', 
                               class_encoder=self.class_encoder,
                               log_predictions=True,
                               max_rows=max_rows,
                               run=self.run,
                               commit=True,
                               output_path=results_dir,
                               metadata=chromosome)
    
    
    def log_model_artifact(self, run=None):
        '''
        Logs a
        
        # TODO log chromosome along with model artifact
        '''
        
        model_path = os.path.join(self.model_dir,f"best_model--fitness-{self.fitness:.2f}--{self.name.replace('=','_')}")
        
        print(f'Logging model artifact for organism {self.name} at\n{model_path}')
        
        os.makedirs(self.model_dir, exist_ok=True)
        run = run or wandb
        log_model_artifact(self.model, model_path, encoder=self.class_encoder, run=run, metadata=self.chromosome)

#     @property
#     def data(self):
#         return (self.x_train, self.y_train), (self.x_test, self.y_test)

#     def load_data(self, train, test=None, std=False, onehot=False, smoothing=0.0):
#         """ Load in memory data
#             train: expect form: (x_train, y_train)
#         """
#         self.x_train, self.y_train = train
#         if test is not None:
#             self.x_test, self.y_test   = test
#         if std:
#             self.x_train, self.x_test = self.standardization(self.x_train, self.x_test)

#         if self.y_train.ndim == 2:
#             self.n_classes = np.max(self.y_train) + 1
#         else:
#             self.n_classes = self.y_train.shape[1]
#         if onehot:
#             self.y_train = to_categorical(self.y_train, self.n_classes)
#             self.y_test  = to_categorical(self.y_test, self.n_classes)
#         if smoothing > 0.0:
#             self.y_train = self.label_smoothing(self.y_train, self.n_classes, smoothing)

#     def cifar10(self, epochs=10, decay=('cosine', 0), save: str=None):
#         """ Train on CIFAR-10
#             epochs : number of epochs for full training
#         """
#         from tensorflow.keras.datasets import cifar10
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         x_train, x_test = self.standardization(x_train, x_test)
#         y_train = to_categorical(y_train, 10)
#         y_test  = to_categorical(y_test, 10)
#         y_train = self.label_smoothing(y_train, 10, 0.1)

#         # compile the model
#         self.compile(loss='categorical_crossentropy', metrics=['acc'])

#         self.warmup(x_train, y_train, save=save)

#         lr, batch_size = self.random_search(x_train, y_train, x_test, y_test, save=save)

#         self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                       lr=lr, decay=decay, save=save)
#         self.evaluate(x_test, y_test)

#     def cifar100(self, epochs=20, decay=('cosine', 0), save: str=None):
#         """ Train on CIFAR-100
#             epochs : number of epochs for full training
#         """
#         from tensorflow.keras.datasets import cifar100
#         (x_train, y_train), (x_test, y_test) = cifar100.load_data()
#         x_train, x_test = self.normalization(x_train, x_test)
#         y_train = to_categorical(y_train, 100)
#         y_test  = to_categorical(y_test, 100)
#         y_train = self.label_smoothing(y_train, 100, 0.1)
        
        
#         self.compile(loss='categorical_crossentropy', metrics=['acc'])

#         self.warmup(x_train, y_train, save=save)

#         lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test, save=save)

#         self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                       lr=lr, decay=decay, save=save)
#         self.evaluate(x_test, y_test)

#     def coil100(self, epochs=20, decay=('cosine', 0), save: str=None):
#         """
#         Columbia University Image Library (COIL-100)
#         """
#         # Get TF.dataset generator for COIL100
#         train, info = tfds.load('coil100', split='train', shuffle_files=True, with_info=True, as_supervised=True)
#         n_classes = info.features['label'].num_classes
#         n_images = info.splits['train'].num_examples
#         input_shape = info.features['image'].shape

#         # Get the dataset into memory
#         train = train.shuffle(n_images).batch(n_images)
#         for images, labels in train.take(1):
#             pass
#         images = np.asarray(images)
#         images, _ = self.standardization(images, None)
#         labels = to_categorical(np.asarray(labels), n_classes)

#         # split the dataset into train/test
#         x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

#         self.compile(loss='categorical_crossentropy', metrics=['acc'])

#         self.warmup(x_train, y_train, save=save)

#         lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test, save=save)

#         self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                       lr=lr, decay=decay, save=save)
#         self.evaluate(x_test, y_test)

        
        
#     def plant_village(self, target_size=[256,256], epochs=20, decay=('cosine', 0), save: str=None, allow_resume: bool=False):
#         """
#         Plant Village leaf disease dataset (2016)
#         """
        
#         data_config = DictConfig({
#                             'load':{'dataset_name':'plant_village',
#                                     'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
#                                     'data_dir':'/media/data/jacob/tensorflow_datasets'},
#                             'preprocess':{'batch_size':32,
#                                           'target_size':target_size}
#                                 })
        
#         data, class_encoder = load_and_preprocess_data(data_config)

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
#         images = np.asarray(images)
#         labels = np.asarray(labels)        
#         x_train, y_train = images, labels
# #         input_shape = x_train.shape[1:]
#         print(f'Loaded {num_samples} samples into memory from plant_village train')
#         # Get the dataset into memory
#         num_samples = batch_size*validation_steps
#         val_dataset = val_dataset.unbatch().batch(num_samples)
#         for images, labels in val_dataset.take(1):
#             pass
#         images = np.asarray(images)
#         labels = np.asarray(labels)        
#         x_val, y_val = images, labels
#         print(f'Loaded {num_samples} samples into memory from plant_village val')

#         # Get the dataset into memory
#         num_samples = batch_size*test_steps
#         test_dataset = test_dataset.unbatch().batch(num_samples)
#         for images, labels in test_dataset.take(1):
#             pass
#         images = np.asarray(images)
#         labels = np.asarray(labels)        
#         x_test, y_test = images, labels
#         print(f'Loaded {num_samples} samples into memory from plant_village test')

# #         images, _ = self.standardization(images, None)
# #         labels = to_categorical(np.asarray(labels), n_classes)
#         # split the dataset into train/test
# #         x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

#         self.set_wandb_env_vars(project='ResNet50_v2', group=f'resnet50_v2-plant_village-res{target_size[0]}')

#         print('compiling')
#         self.compile(loss='categorical_crossentropy', metrics=['acc', 'recall','precision'])
        
#         with wandb.init(reinit=True, job_type='warmup', tags=['warmup']) as run:
#             print('initiating warmup')
#             self.warmup(x_train, y_train, save=save, allow_resume=allow_resume)
            
# #         with wandb.init(reinit=True, job_type='grid_search', tags=['grid_search']) as run:
#         print('initiating grid_search. dir(self):\n',dir(self))
#         lr, batch_size = self.grid_search(x_train, y_train, x_val, y_val, save=save,
#                                           batch_range=[16, 32], allow_resume=allow_resume)
        
#         with wandb.init(reinit=True, job_type='training', tags=['training']) as run:
#             print('initiating training. dir(self):\n')
#             self.training(x_train, y_train, validation_data = (x_val, y_val), epochs=epochs, batch_size=batch_size,
#                           lr=lr, decay=decay, save=save)
            
#             print('initiating evaluate. dir(self):\n',dir(self))
#             result = self.evaluate(x_test, y_test)
#             run.log({'test_results':result})
