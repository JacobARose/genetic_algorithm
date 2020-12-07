VERBOSE = True
import os
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
from tensorflow.keras.losses import CategoricalCrossentropy
from typing import List, Tuple, Union, Dict, NamedTuple
from genetic_algorithm import stateful
from genetic_algorithm.chromosome import Chromosome
from genetic_algorithm.chromosome import sampler
from genetic_algorithm.utils.data_utils import log_model_artifact
import wandb
from wandb.keras import WandbCallback


ActivationLayers = Box(ReLU=ReLU, ELU=ELU, LeakyReLU=LeakyReLU)
PoolingLayers = Box(MaxPool2D=MaxPool2D, AveragePooling2D=AveragePooling2D)
import gc


def process_chromosome_functions(chromosome: Dict=None, **kwargs):
    '''
    Function to convert a chromosome dict containing str names of functions to the actual function. Use this when running organism.build_model()
    '''
    if chromosome:
        chromosome = copy.deepcopy(chromosome)
        chromosome['activation_type'] = ActivationLayers[chromosome['activation_type']]
        chromosome['pool_type'] = PoolingLayers[chromosome['pool_type']]
        return chromosome
    elif 'activation_type' in kwargs:
        return ActivationLayers[kwargs['activation_type']]
    elif 'pool_type' in kwargs:
        return PoolingLayers[kwargs['pool_type']]
    else:
        raise Exception(f'Invalid arguments provided to function process_chromosome_functions() in {__file__}')




class Organism:
    def __init__(self,
                 data: Dict[str,tf.data.Dataset],
                 config=None,
                 chromosome=None,
                 phase=0,
                 generation_number=0,
                 organism_id=0,
                 best_organism=None,
                 class_encoder=None,
                 verbose=True,
                 debug=False):
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
        self.class_encoder=class_encoder
        self._parent = best_organism
        self._children=[]
        
        self.model_dir = config.model_dir or '/media/data_cifs_lrs/projects/prj_fossils/users/jacob/experiments/genetic_algorithm_default'
        if phase > 0:
            if best_organism is None:
                print(f'phase {phase} gen {generation} organism {organism_id}.\nNo previous best model, creating from scratch.')
            else:
                self.last_model = best_organism.model
            
        self.debug = debug
        self.verbose = verbose
    
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
        config.steps_per_epoch = config.steps_per_epoch or len(self.data['train'])
        config.validation_steps = config.validation_steps or len(self.data['val'])
        
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
        return self._chromosome
    
    @chromosome.setter
    def chromosome(self, chromosome):
        self._chromosome = chromosome
        
    def build_model(self):
        '''
        This is the function to build the keras model
        '''
        K.clear_session()
        gc.collect()
        
        chromosome = process_chromosome_functions(chromosome=self.chromosome)
        
        inputs = Input(shape=self.config.input_shape)
        if self.phase != 0:
            # Slice the prev best model # Use the model as a layer # Attach new layer to the sliced model
            intermediate_model = Model(inputs=self.last_model.input,
                                       outputs=self.last_model.layers[-3].output)
            for layer in intermediate_model.layers:
                # To make the iteration efficient
                layer.trainable = False
            inter_inputs = intermediate_model(inputs)
            x = Conv2D(filters=chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=chromosome['a_filter_size'],
                       use_bias=chromosome['a_include_BN'])(inter_inputs)
            # This is to ensure that we do not randomly chose anothere activation
            chromosome['activation_type'] = process_chromosome_functions(activation_type=self.best_organism.chromosome['activation_type'])
        else:
            # For PHASE 0 only
            # input layer
            x = Conv2D(filters=chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=chromosome['a_filter_size'],
                       use_bias=chromosome['a_include_BN'])(inputs)
            
        if chromosome['a_include_BN']:
            x = BatchNormalization()(x)
        x = chromosome['activation_type']()(x)
        if chromosome['include_pool']:
            x = chromosome['pool_type'](strides=(1,1),
                                             padding='same')(x)
        if self.phase != 0 and chromosome['b_include_layer'] == False:
            # Except for PHASE0, there is a choice for
            # the number of layers that the model wants
            if chromosome['include_skip']:
                y = Conv2D(filters=chromosome['a_output_channels'],
                           kernel_size=(1,1),
                           padding='same')(inter_inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.config.output_size, activation='softmax')(x)
        else:
            # PHASE0 or no skip
            # in the tail
            x = Conv2D(filters=chromosome['b_output_channels'],
                       padding='same',
                       kernel_size=chromosome['b_filter_size'],
                       use_bias=chromosome['b_include_BN'])(x)
            if chromosome['b_include_BN']:
                x = BatchNormalization()(x)
            x = chromosome['activation_type']()(x)
            if chromosome['include_skip']:
                y = Conv2D(filters=chromosome['b_output_channels'],
                           padding='same',
                           kernel_size=(1,1))(inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.config.output_size, activation='softmax')(x)
        self.model = Model(inputs=[inputs], outputs=[x])
        self.model.compile(optimizer='adam',
                           loss=CategoricalCrossentropy(label_smoothing=chromosome['label_smoothing']),
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
        
        config = self.config
        if isinstance(config, omegaconf.dictconfig.DictConfig):
            config = OmegaConf.to_container(config)
        
        
        self.run = wandb.init(**self.get_wandb_credentials(phase=self.phase,
                               generation_number=generation_number),
                               config=config,
                               reinit=True,
                               tags=['fitnessFunction', self.name, self.config.experiment_uid])
        self.run_id = self.run.id
        self.run_name = self.run.name
        
        with self.run:
            self.run.log({'chromosome':self.chromosome}, commit=False)
            
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
    def add_child(self, child):
        assert isinstance(child, Organism)
        self._children.append(child)
        
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
                
#         child_chromosome = Chromosome(child_chromosome).get_chromosome(serialized=True)
        child = Organism(chromosome=child_chromosome,
                         data=self.data,
                         config=self.config,
                         phase=self.phase,
                         generation_number=generation_number,
                         organism_id=f'{self.organism_id}+{partner.organism_id}',
                         best_organism=self.best_organism,
                         class_encoder=self.class_encoder)
        
        child.build_model()
        child.fitnessFunction(self.train_data,
                              self.val_data,
                              generation_number=generation_number)
        self.add_child(child)
        partner.add_child(child)
        return child
    
    def mutation(self, generation_number):
        '''
        One of the gene is to be mutated.
        '''
        # TODO refactor this random sampling
        index = np.random.randint(0, len(self.chromosome))
        key = list(self.chromosome.keys())[index]
                
            
        ### TODO Consider whether or not I should refactor this to include mutation in zeroth phase
        if  self.phase != 0:
            options = sampler.get_options(self.phase)
            new_variant = options.sample_k_variants_from_gene(gene=key, k=1)
            if self.verbose:
                print(f'Mutating organism {self.name} with chromosome {self.chromosome}.')
                print(f'Mutation resulted in gene {key} changing from variant {self.chromosome[key]} to {new_variant}')
            self.chromosome[key] = new_variant                              #options[key][np.random.randint(len(options[key]))]
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
        
    def __repr__(self):
        return f'<Organism object[{self.name}]>'
        
        
    def get_wandb_credentials(self, phase: int=None, generation_number: int=None):
        phase = phase or self.phase
        generation_number = generation_number or self.generation_number
        if self.debug:
            return get_wandb_credentials(phase=phase,
                                          name=self.name,
                                          generation_number=generation_number,
                                          entity="jrose",
                                          project=f"vlga-plant_village-DEBUG")           
        return get_wandb_credentials(phase=phase,
                                      generation_number=generation_number,
                                      entity="jrose",
                                      name=self.name,
                                      project=f"vlga-plant_village")
    
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