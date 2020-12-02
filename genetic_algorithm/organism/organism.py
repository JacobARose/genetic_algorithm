
VERBOSE = True
import pandas as pd
import json
from box import Box
from bunch import Bunch
import copy
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

import omegaconf
from omegaconf import OmegaConf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, ELU, LeakyReLU, Flatten, Dense, Add, AveragePooling2D, GlobalAveragePooling2D
from typing import List, Tuple, Union, Dict, NamedTuple
from genetic_algorithm import stateful
from genetic_algorithm.chromosome import Chromosome
import wandb
from wandb.keras import WandbCallback


ActivationLayers = Box(ReLU=ReLU, ELU=ELU, LeakyReLU=LeakyReLU)
PoolingLayers = Box(MaxPool2D=MaxPool2D, AveragePooling2D=AveragePooling2D)
import gc


class Organism:
    def __init__(self,
                 data: Dict[str,tf.data.Dataset],
                 config=None,
                 chromosome=None,
                 phase=0,
                 generation_number=0,
                 organism_id=0,
                 best_organism=None,
                 DEBUG=False):
        '''
        
        Organism is an actor with a State that can take Action in the environment
        
        config is a . accessible dict object containing model params that will stay constant during evolution
        chromosome is a dictionary of genes
        phase is the phase that the individual belongs to
        best_organism is the best organism of the previous phase
        
        TODO:
        
        1. implement to_json and from_json methods for copies
        2. Separate out step where organism is associated with a dataset
        '''
        self.data = data
        self.train_data = data['train']
        self.val_data = data['val']
        self.test_data = data['test']
        self.config = config
        self.chromosome = chromosome
        self.phase = phase
        self.generation_number = generation_number
        self.organism_id = organism_id
        self.best_organism=best_organism

        if phase > 0:
            if best_organism is None:
                print(f'phase {phase} gen {generation} organism {organism_id}.\nNo previous best model, creating from scratch.')
            else:
                self.last_model = best_organism.model
            
        self.debug = DEBUG
    
    @property
    def name(self):
        return f'phase_{self.phase}-gen_{self.generation_number}-organism_{self.organism_id}'
    
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config=None):
        config = config or OmegaConf.create({})
        print(config)
        config.input_shape = config.input_shape or (224,224,3)
        config.output_size = config.output_size or 38
        config.epochs_per_organism = config.epochs_per_organism or 5
        self._config = config
        
    def get_metrics(self):
        return [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    
    @property
    def fitness_metric_name(self):
        return 'accuracy'
   
    @property
    def chromosome(self):
        return self._chromosome.get_chromosome(serialized=False)
    
    @chromosome.setter
    def chromosome(self, chromosome):
        self._chromosome = chromosome
        
    def build_model(self):
        '''
        This is the function to build the keras model
        '''
        K.clear_session()
        gc.collect()
        inputs = Input(shape=self.config.input_shape)
        if self.phase != 0:
            # Slice the prev best model # Use the model as a layer # Attach new layer to the sliced model
            intermediate_model = Model(inputs=self.last_model.input,
                                       outputs=self.last_model.layers[-3].output)
            for layer in intermediate_model.layers:
                # To make the iteration efficient
                layer.trainable = False
            inter_inputs = intermediate_model(inputs)
            x = Conv2D(filters=self.chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['a_filter_size'],
                       use_bias=self.chromosome['a_include_BN'])(inter_inputs)
            # This is to ensure that we do not randomly chose anothere activation
            self.chromosome['activation_type'] = self.best_organism.chromosome['activation_type']
        else:
            # For PHASE 0 only
            # input layer
            x = Conv2D(filters=self.chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['a_filter_size'],
                       use_bias=self.chromosome['a_include_BN'])(inputs)
            
        if self.chromosome['a_include_BN']:
            x = BatchNormalization()(x)
        x = self.chromosome['activation_type']()(x)
        if self.chromosome['include_pool']:
            x = self.chromosome['pool_type'](strides=(1,1),
                                             padding='same')(x)
        if self.phase != 0 and self.chromosome['b_include_layer'] == False:
            # Except for PHASE0, there is a choice for
            # the number of layers that the model wants
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['a_output_channels'],
                           kernel_size=(1,1),
                           padding='same')(inter_inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.config.output_size, activation='softmax')(x)
        else:
            # PHASE0 or no skip
            # in the tail
            x = Conv2D(filters=self.chromosome['b_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['b_filter_size'],
                       use_bias=self.chromosome['b_include_BN'])(x)
            if self.chromosome['b_include_BN']:
                x = BatchNormalization()(x)
            x = self.chromosome['activation_type']()(x)
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['b_output_channels'],
                           padding='same',
                           kernel_size=(1,1))(inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.config.output_size, activation='softmax')(x)
        self.model = Model(inputs=[inputs], outputs=[x])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=self.get_metrics())
        
    def fitnessFunction(self,
                        train_data,
                        val_data,
                        generation_number):
        '''
        This function is used to calculate the
        fitness of an individual.
        '''
        print('\n'*2,'FITNESS FUNCTION')
        print('vars():', vars())
        
        config = self.config
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            config = OmegaConf.to_container(config)
        
        
        self.run = wandb.init(**self.get_wandb_credentials(phase=self.phase,
                               generation_number=generation_number),
                               config=config,
                               reinit=True)
        self.run_id = self.run.id
        
        with self.run:
            self.model.fit(train_data,
                           steps_per_epoch=self.config.steps_per_epoch,
                           epochs=self.config.epochs_per_organism,
                           callbacks=[WandbCallback()],
                           verbose=1)
            self.results = self.model.evaluate(val_data,
                                               steps=self.config.validation_steps,
                                               return_dict=True,
                                               verbose=1)
            self.fitness = self.results[self.fitness_metric_name]
            print(self.name)
            print('fitness:', self.fitness)
            print('results:\n', len(self.results))
            print(self.results)
        
        
#     @results.setter
#     def results(self, metrics_values):
#         self._results = {name:values for name, value in zip(self.model.metrics_names, metrics_values)}
        
#     @property
#     def results(self):
#         return self._results
        
    def crossover(self,
                  partner,
                  generation_number):
        '''
        # TODO Move crossover() logic outside of Organism, or refactor to reinforce Factory pattern
        
        This function helps in making children from two
        parent individuals.
        '''
        child_chromosome = {}
        endpoint = np.random.randint(low=0, high=len(self.chromosome))
        for idx, key in enumerate(self.chromosome):
            if idx <= endpoint:
                child_chromosome[key] = self.chromosome[key]
            else:
                child_chromosome[key] = partner.chromosome[key]
                
        child_chromosome = Chromosome(child_chromosome).get_chromosome(serialized=True)
        child = Organism(chromosome=child_chromosome,
                         data=self.data,
                         config=self.config,
                         phase=self.phase,
                         generation_number=generation_number,
                         organism_id=f'{self.organism_id}+{partner.organism_id}',
                         best_organism=self.best_organism)
        
        child.build_model()
        child.fitnessFunction(self.train_data,
                              self.val_data,
                              generation_number=generation_number)
        return child
    
    def mutation(self, generation_number):
        '''
        One of the gene is to be mutated.
        '''
        index = np.random.randint(0, len(self.chromosome))
        key = list(self.chromosome.keys())[index]
        if  self.phase != 0:
            self.chromosome[key] = options[key][np.random.randint(len(options[key]))]
        else:
            self.chromosome[key] = options_phase0[key][np.random.randint(len(options_phase0[key]))]
        self.build_model()
        self.fitnessFunction(self.train_data,
                             self.val_data,
                             generation_number=generation_number)
    
    def show(self):
        '''
        Util function to show the individual's properties.
        '''
        pp.pprint(self.config)
        pp.pprint(self.chromosome)
        
        
    def get_wandb_credentials(self, phase: int=None, generation_number: int=None):
        phase = phase or self.phase
        generation_number = generation_number or self.generation_number
        if self.debug:
            return get_wandb_credentials(phase=phase,
                                          generation_number=generation_number,
                                          entity="jrose",
                                          project=f"vlga-plant_village-DEBUG")           
        return get_wandb_credentials(phase=phase,
                                      generation_number=generation_number,
                                      entity="jrose",
                                      project=f"vlga-plant_village")

        
    
def get_wandb_credentials(phase: int,
                          generation_number: int,
                          entity="jrose",
                          project=f"vlga-plant_village",
                          name=None):
    
    return dict(entity=entity,
                project=project,
                group='KAGp{}'.format(phase),
                job_type='g{}'.format(generation_number),
                name=name)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()