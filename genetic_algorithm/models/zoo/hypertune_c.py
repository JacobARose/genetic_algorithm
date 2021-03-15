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

from typing import Tuple
import random
import math
import sys, os, json
import wandb


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



class HyperTune(object):
    ''' Hyperparameter tuning  base (super) class for Composable Models '''

    def __init__(self):
        """ Constructor
        """
        pass

    ###
    # Hyperparameter Tuning
    ###

    def _tune(self,
              x_train, y_train,
              x_test, y_test, 
              epochs,
              num_train_samples,
              num_test_samples,
              lr,
              batch_size,
              weights,
              loss,
              metrics,
              frozen_layers: Tuple=None,
              run_index=0):
        """ Helper function for hyperparameter tuning
            x_train   : training images
            y_train   : training labels
            x_test    : test images
            y_test    : test labels
            lr        : trial learning rate
            batch_size: the batch size (constant)
            epochs    : the number of epochs
            steps     : steps per epoch
            weights   : warmup weights
            loss      : loss function
            metrics   : metrics to report during training
        """
        try:
            steps_per_epoch = int(np.ceil(num_train_samples/batch_size))
            test_steps = int(np.ceil(num_test_samples/batch_size))
            
            if isinstance(frozen_layers, tuple):
                assert len(frozen_layers)==2
                self.freeze(self.model, frozen_layers)
            
            # Compile the model for the new learning rate
            self.compile(optimizer=Adam(lr), loss=loss, metrics=metrics)
            config = {'epochs':epochs,
                      'num_train_samples':num_train_samples,
                      'num_test_samples':num_test_samples,
                      'lr':lr,
                      'batch_size':batch_size,
                      'frozen_layers':frozen_layers, 
                      'loss':loss,
                      'run_index':run_index}
            with wandb.init(reinit=True, job_type='grid_search', config=config, tags=['grid_search', str(run_index)]) as run:
            
                callbacks = self.get_callbacks(lr_schedule=False, log_wandb=True)
            
                print("\n*** Learning Rate: ", lr, "Batch Size: ", batch_size, "frozen_layers: ", frozen_layers, "run_index", run_index)
#                 self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
#                                epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=1)
                if isinstance(x_train, tf.python.data.ops.dataset_ops.BatchDataset):
                    x_train = x_train.unbatch().batch(batch_size)
                    if y_train:
                        y_train = y_train.unbatch().batch(batch_size)
                    x_test = x_test.unbatch().batch(batch_size)
                    if y_test:
                        y_test = y_test.unbatch().batch(batch_size)
                if issubclass(type(x_train), tf.data.Dataset):
                    batch_size=None

                self.model.fit(x_train,
                               y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               callbacks=callbacks,
                               verbose=1)
                # Evaluate the model
                result = self.evaluate(x_test,
                                       y_test,
                                       steps=test_steps)
                run.log({'val_results':result})
        except Exception as e:
            print(e)
            print('Attempting to continue with next trial')
            result = [np.inf]
        finally:         
            # Reset the weights
            self.model.set_weights(weights)

        return result

        # Search learning rate
    
    def _tune_lr(self,
                 x_train=None, y_train=None, 
                 x_test=None, y_test=None,
                 epochs=3,
                 num_train_samples=None,
                 num_test_samples=None,
                 lr_range=[0.0001, 0.001, 0.01],
                 batch_size=32,
                 frozen_layers: Tuple[int]=None,
                 loss='categorical_crossentropy', 
                 metrics=['acc'], 
                 save=None,
                 allow_resume=False,
                 run_index: int=0):
        
        weights = self.model.get_weights()
        ## SEARCH INITIAL RANGE ##
        v_loss = []
        for lr in lr_range:
            result = self._tune(x_train, y_train,
                                x_test, y_test,
                                epochs,
                                num_train_samples,
                                num_test_samples,
                                lr,
                                batch_size,
                                weights, 
                                loss,
                                metrics,
                                frozen_layers=frozen_layers,
                                run_index=run_index)
            v_loss.append(result[0])
            run_index += 1
            
        ## KEEP BEST INITIAL LEARNING RATE FROM MACRO SEARCH
        best_loss = sys.float_info.max
        for _ in range(len(lr_range)):
            if v_loss[_] < best_loss:
                best_loss = v_loss[_]
                lr = lr_range[_]

        ## IF BEST LR IS THE SMALLEST, EXPAND SEARCH TO LR / 2
        if lr == lr_range[0]:
            # try 1/2 the lowest learning rate
            new_lr = (lr / 2.0)
            result = self._tune(x_train, y_train,
                                x_test, y_test, 
                                epochs,
                                num_train_samples,
                                num_test_samples,
                                new_lr,
                                batch_size,
                                weights,
                                loss,
                                metrics,
                                frozen_layers=frozen_layers, 
                                run_index=run_index)
            run_index += 1

            ## KEEP LR / 2 IF IT'S BEST
            if result[0] < best_loss:
                lr = new_lr          # lr / 2.0
            ## IF NOT, TRY THE MIDPOINT BETWEEN THE 1ST AND 2ND LOWEST LEARNING RATES IN THE SEARCH RANGE
            else:
                new_lr = (lr_range[0] + lr_range[1]) / 2.0
                result = self._tune(x_train, y_train,
                                    x_test, y_test,
                                    epochs,
                                    num_train_samples,
                                    num_test_samples,
                                    new_lr,
                                    batch_size,
                                    weights,
                                    loss,
                                    metrics,
                                    frozen_layers=frozen_layers,
                                    run_index=run_index)
                run_index += 1

                ## KEEP (LRANGE[0] + LRANGE[1]) / 2.0
                if result[0] < best_loss:
                    lr = new_lr        # lr / 2.0
                
        elif lr == lr_range[len(lr_range)-1]:
            ## IF BEST LEARNING RATE WAS THE MAX IN SEARCH RANGE, EXPAND SEARCH TO LR * 2
            new_lr = (lr * 2.0)
            result = self._tune(x_train, y_train,
                                x_test, y_test,
                                epochs,
                                num_train_samples,
                                num_test_samples,
                                new_lr,
                                batch_size,
                                weights,
                                loss,
                                metrics,
                                frozen_layers=frozen_layers,
                                run_index=run_index)
            run_index += 1
            ## KEEP LR * 2 IF IT'S BEST
            if result[0] < best_loss:
                lr = new_lr           # lr * 2.0
		
        print("*** Selected best learning rate:", lr)
        return lr, best_loss, run_index
    
    
    
    
    
    def _tune_batch_size(self, 
                         x_train=None, y_train=None,
                         x_test=None, y_test=None,
                         epochs=3,
                         num_train_samples=None,
                         num_test_samples=None,
                         lr:float=None,
                         batch_range=[32, 64],
                         frozen_layers: Tuple[int]=None,
                         loss='categorical_crossentropy', 
                         metrics=['acc'], 
                         save=None,
                         allow_resume=False,
                         run_index: int=0):
        # skip the first batch size - since we used it in searching learning rate
#         datagen = ImageDataGenerator()
        weights = self.model.get_weights()
        v_loss = []
        print(f'BEGINNING HP TUNING OF BATCH SIZE, ACROSS RANGE: {batch_range}')
        for batch_size in batch_range:
            print("*** Batch Size", batch_size)

            result = self._tune(x_train, y_train,
                                x_test, y_test,
                                epochs,
                                num_train_samples,
                                num_test_samples,
                                lr,
                                batch_size,
                                weights,
                                loss,
                                metrics, 
                                frozen_layers=frozen_layers, 
                                run_index=run_index)
            v_loss.append(result[0])            
            run_index += 1
            
        # Find the best batch size based on validation loss
        best_loss = sys.float_info.max
        batch_size = batch_range[0]
        for i in range(len(batch_range)):
            if v_loss[i] < best_loss:
                best_loss = v_loss[i]
                batch_size = batch_range[i]

        print("*** Selected best batch size:", batch_size)
        return batch_size, best_loss, run_index
    
    
    
    
    
    
    def grid_search(self, 
                    x_train=None, y_train=None,
                    x_test=None, y_test=None,
                    epochs=3,
                    num_train_samples=None, num_test_samples=None,
                    lr_range=[0.0001, 0.001, 0.01], 
                    batch_range=[32, 128],
                    frozen_layers_hp_range = [(0,-17), (0,-28), (0,-40), (0,-52), (0,-63)],
                    loss='categorical_crossentropy', 
                    metrics=['acc'], 
                    save=None,
                    allow_resume=False, 
                    run_index: int=0):
        """ Do a grid search for hyperparameters
            x_train : training images
            y_train : training labels
            epochs  : number of epochs
            steps   : number of steps per epoch
            lr_range: range for searching learning rate
            batch_range: range for searching batch size
            loss    : loss function
            metrics : metrics to report during training
            
            # TODO: Refactor for flexible rebatching of tf.data inputs
            
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train
            x_test  = self.x_test
            y_test  = self.y_test
            
        if num_train_samples is None:
            num_train_samples = x_train.shape[0]
        if num_test_samples is None:
            num_test_samples = x_test.shape[0]

        best_hp = AttrDict({'lr':lr_range[0],
                            'batch_size':batch_range[0],
                            'frozen_layers':frozen_layers_hp_range[0],
                            'loss':np.inf,
                            'run_index':run_index})

        if save is not None:
            if allow_resume:
                if os.path.exists(f'{save}/tune/chkpt.index'):
                    self.model.load_weights(save + '/warmup/chkpt')
                    with open(save + '/tune/hp.json', 'r') as f:
                        data = json.load(f)
                        best_hp.update(data)
#                         lr, bs, frozen_layers, best_loss = data['lr'], data['bs'], data['frozen_layers'], data['best_loss']                    
                    print("Loading weights and resuming from end of previous grid search")
                    return best_hp # lr, bs, frozen_layers
            
            for path in [ save, save + '/tune']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.isfile(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.isfile(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')

        print("\n*** Hyperparameter Grid Search")        
        ################################################
        ################################################
#         frozen_layers_hp_range = [(0,-17), (0,-28), (0,-40), (0,-52), (0,-63)]
        print('BEGINNING RECURSIVE HPARAM SEARCH.')
        print(f'OUTER LOOP: FROZEN_LAYERS RANGE -> {frozen_layers_hp_range}')
        print(f'INNER LOOP: LEARNING RATE RANGE -> {lr_range}')
        num_trials = len(frozen_layers_hp_range)*len(lr_range)
        print(f'{len(frozen_layers_hp_range)} * {len(lr_range)} = {num_trials} total trials, each for {epochs} epochs and {num_train_samples:,} training samples')
        ################################################
        batch_size = batch_range[0]
        for frozen_layers in frozen_layers_hp_range:
            lr, loss, run_index = self._tune_lr(x_train, y_train,
                                                x_test, y_test,
                                                epochs,
                                                num_train_samples,
                                                num_test_samples,
                                                lr_range=lr_range,
                                                batch_size=batch_size, 
                                                frozen_layers=frozen_layers,
                                                loss=loss,
                                                metrics=metrics,
                                                save=save,
                                                allow_resume=allow_resume, 
                                                run_index=run_index)
            
            if loss < best_hp.loss:
                best_hp.update(frozen_layers=frozen_layers,
                               lr=lr,
                               best_loss=loss,
                               run_index=run_index)
                
        ################################################
        ################################################
        lr = best_hp.lr
        best_batch_size, best_loss, run_index = self._tune_batch_size(x_train, y_train,
                                                                      x_test, y_test,
                                                                      epochs,
                                                                      num_train_samples,
                                                                      num_test_samples,
                                                                      lr=lr,
                                                                      batch_range=batch_range, 
                                                                      frozen_layers=frozen_layers,
                                                                      loss=loss, 
                                                                      metrics=metrics,
                                                                      save=save,
                                                                      allow_resume=allow_resume,
                                                                      run_index=run_index)
        
        best_hp.update(batch_size=best_batch_size,
                       best_loss=best_loss,
                       run_index=run_index)
        ################################################
        ################################################
        if save is not None:
            with open(save + '/tune/hp.json', 'w') as f:
                data = best_hp #{ 'lr' : best_lr, 'bs': best_batch_size , 'best_loss':best_loss}
                json.dump(data, f)
            self.model.save_weights(save + '/tune/chkpt')

        # return the best learning rate and batch size
        return best_hp ## best_lr, best_batch_size, best_loss

    def random_search(self,
                      x_train=None, y_train=None,
                      x_test=None, y_test=None,
                      epochs=3,
                      steps=250,
                      lr_range=[0.0001, 0.001, 0.01, 0.1],
                      batch_range=[32, 128],
                      loss='categorical_crossentropy', 
                      metrics=['acc'], 
                      trials=5,
                      save=None):
        """ Do a grid search for hyperparameters
            x_train : training images
            y_train : training labels
            epochs  : number of epochs
            steps   : number of steps per epoch
            lr_range: range for searching learning rate
            batch_range: range for searching batch size
            loss    : loss function
            metrics : metrics to report during training
            trials  : maximum number of trials
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train
            x_test  = self.x_test
            y_test  = self.y_test

        if save is not None:
            for path in [ save, save + '/tune']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.isfile(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.isfile(save + '/init/chkpt.index'):
#                 self.model.load_weights(save + '/warmup/chkpt')
                self.model.load_weights(save + '/init/chkpt')

        print("\n*** Hyperparameter Random Search")

        # Save the original weights
        weights = self.model.get_weights()

        # Base the number of steps on the min batch size to try
        min_bs = np.min(batch_range)
        best = (0, 0, 0)

        # lr values already tried, as not to repeat
        tried = []
        for _ in range(trials):
            print("\nTrial ", _ + 1, "of", trials)

            lr = lr_range[random.randint(0, len(lr_range)-1)]
            bs = batch_range[random.randint(0, len(batch_range)-1)]

            # Check for repeat
            if (lr, bs) in tried:
                print("Random Selection already tried", (lr, bs))
                continue
            tried.append( (lr, bs))

            # Adjust steps so each trial sees same number of examples
            trial_steps = int(min_bs / bs * steps)

            result = self._tune(x_train, y_train, x_test, y_test, epochs, trial_steps, lr, bs, weights, loss, metrics)
    
            # get the model and hyperparameters with the best validation accuracy
            # we call this a near-optima point
            val_acc = result[1]
            if val_acc > best[0]:
                best = (val_acc, lr, bs)
                print("\nCurrent Best: lr", lr, "bs", bs)

        # narrow search space to within vicinity of the best near-optima
        learning_rates = [ best[1] / 2, best[1] * 2]
        batch_sizes = [int(best[2] / 2), int(best[2] * 2)]
        for _ in range(trials):
            print("\nNarrowing, Trial", _ + 1)
            lr = learning_rates[random.randint(0, 1)]
            bs = batch_sizes[random.randint(0, 1)]

            # Check for repeat
            if (lr, bs) in tried:
                print("Random Selection already tried", (lr, bs))
                continue
            tried.append( (lr, bs))

            # Adjust steps so each trial sees same number of examples
            trial_steps = int(min_bs / bs * steps)

            result = self._tune(x_train, y_train, x_test, y_test, epochs, trial_steps, lr, bs, weights, loss, metrics)
   
            val_acc = result[1]
            if val_acc > best[0]:
                best = (val_acc, lr, bs)
                print("\nCurrent Best: lr", lr, "bs", bs)

        print("\nSelected Learning Rate", lr, "Batch Size", bs)

        if save is not None:
            with open(save + '/tune/hp.json', 'w') as f:
                data = { 'lr' : lr, 'bs': bs, 'trials': trials }
                json.dump(data, f)
            self.model.save_weights(save + '/tune/chkpt')

        return best[1], best[2]

    
    
#             # equalize the number of examples per epoch
#             steps = int(np.floor(num_train_samples / bs))
#             self.model.fit(x_train, y_train, batch_size=bs,
#                            epochs=epochs, steps_per_epoch=steps,
#                            verbose=1)

#             self.model.fit(datagen.flow(x_train, y_train, batch_size=bs),
#                                      epochs=epochs, steps_per_epoch=steps, verbose=1)
            # Evaluate the model
#             result = self.evaluate(x_test, y_test)
#             v_loss.append(result[0])
            
            # Reset the weights
#             self.model.set_weights(weights)

