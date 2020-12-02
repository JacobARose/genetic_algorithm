#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_ds = data['train'].map(lambda x,y: (resize(x),y)).shuffle(1024).cache().batch(config.batch_size).prefetch(-1)
def get_hardest_k_examples(test_dataset, model, k=32):
    class_probs = model.predict(test_dataset)
    predictions = np.argmax(class_probs, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
    argsort_loss =  np.argsort(losses)

    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
    true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

    return highest_k_losses, hardest_k_examples, true_labels, predictions
        
def log_high_loss_examples(test_dataset, model, k=32, run=None):
    print(f'logging k={k} hardest examples')
    losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
    
    run = run or wandb
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, true_labels, predictions, losses)]
        })

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
from pyleaves.utils import set_tf_config
set_tf_config(num_gpus=1)

import wandb
from wandb.keras import WandbCallback
# wandb.login()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, ELU, LeakyReLU, Flatten, Dense, Add, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding
import pprint
pp = pprint.PrettyPrinter(indent=4)

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(666)
tf.random.set_seed(666)

from typing import List, Tuple, Union, Dict, NamedTuple
import tensorflow_datasets as tfds
from omegaconf import OmegaConf

# from tfrecord_utils.img_utils import resize_repeat
# from boltons.funcutils import partial
# import logging
# logger = logging.getLogger('')

LOG_DIR = '/media/data/jacob/GitHub/evolution_logs'
import os
os.makedirs(LOG_DIR, exist_ok=True)
from paleoai_data.utils.logging_utils import get_logger
logger = get_logger(logdir=LOG_DIR, filename='generation_evolution_logs.log', append=True)

get_ipython().system('nvidia-smi')


# In[2]:


from genetic_algorithm.datasets.plant_village import ClassLabelEncoder, load_and_preprocess_data
from genetic_algorithm import stateful


# ## Data Definitions

# ## Creating and tracking label encoders

# In[3]:


# dataset_name='plant_village'
# data_dir = '/media/data/jacob/tensorflow_datasets'

exp_config = OmegaConf.create({'seed':756, #237,
                               'batch_size':24,
                               'input_shape':(224,224,3),
                               'output_size':38,
                               'epochs_per_organism':3,
                               'results_dir':'/media/data_cifs_lrs/projects/prj_fossils/users/jacob/experiments/Nov2020-Jan2021'
                              })

data_config = OmegaConf.create({'load':{},'preprocess':{}})

data_config['load'] = {'dataset_name':'plant_village',
                       'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
                       'data_dir':'/media/data/jacob/tensorflow_datasets'}

data_config['preprocess'] = {'batch_size':exp_config.batch_size,
                             'target_size':exp_config.input_shape[:2]}

generation_config = OmegaConf.create({
                                      'population_size':5,
                                      'num_generations_per_phase':3,
                                      'fitSurvivalRate': 0.5,
                                      'unfitSurvivalProb':0.2,
                                      'mutationRate':0.1,
                                      'num_phases':5
                                    })
organism_config = OmegaConf.create({'input_shape':exp_config.input_shape,
                                    'output_size':exp_config.output_size,
                                    'epochs_per_organism':exp_config.epochs_per_organism})


# In[4]:


# dataset_name='plant_village'
# data_dir = '/media/data/jacob/tensorflow_datasets'

DEBUG = False#True

if DEBUG:
    exp_config = OmegaConf.create({'seed':6227,
                                   'batch_size':16,
                                   'input_shape':(128,128,3),
                                   'output_size':38,
                                   'epochs_per_organism':1,
                                   'results_dir':'/media/data_cifs_lrs/projects/prj_fossils/users/jacob/experiments/Nov2020-Jan2021/debugging_trials'
                                  })

    data_config = OmegaConf.create({'load':{},'preprocess':{}})
    data_config['load'] = {'dataset_name':'plant_village',
                           'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
                           'data_dir':'/media/data/jacob/tensorflow_datasets'}

    data_config['preprocess'] = {'batch_size':exp_config.batch_size,
                                 'target_size':exp_config.input_shape[:2]}

    generation_config = OmegaConf.create({
                                          'population_size':1,
                                          'num_generations_per_phase':1,
                                          'fitSurvivalRate': 0.5,
                                          'unfitSurvivalProb':0.2,
                                          'mutationRate':0.1,
                                          'num_phases':3
                                        })
    organism_config = OmegaConf.create({'input_shape':exp_config.input_shape,
                                        'output_size':exp_config.output_size,
                                        'epochs_per_organism':exp_config.epochs_per_organism})


# In[5]:


config = OmegaConf.create({
                            'experiment':exp_config,
                            'data':data_config,
                            'generation':generation_config,
                            'organism':organism_config
})
print(config.pretty())


# In[6]:


data, class_encoder = load_and_preprocess_data(data_config)


# In[7]:


if DEBUG:
    config.organism.steps_per_epoch = 10
    config.organism.validation_steps = 10
else:
    config.organism.steps_per_epoch = len(data['train'])
    config.organism.validation_steps = len(data['val'])


# In[8]:


config


# # Organism
# An organism contains the following:
# 
# 1. phase - This denotes which phase does the organism belong to
# 2. chromosome - A dictionary of genes (hyperparameters)
# 3. model - The `tf.keras` model corresponding to the chromosome
# 4. best_organism - The best organism in the previous **phase**

# In[9]:


VERBOSE = True
import pandas as pd
import json
from box import Box
from bunch import Bunch
# from pprint import pprint as pp
import random

ActivationLayers = Box(ReLU=ReLU, ELU=ELU, LeakyReLU=LeakyReLU)
PoolingLayers = Box(MaxPool2D=MaxPool2D, AveragePooling2D=AveragePooling2D)



class Chromosome(stateful.Stateful):#BaseChromosome):#(NamedTuple):
    
    def __init__(self,
                 hparams: Dict=None,
                 name=''):
        super().__init__()
        self.set_state(hparams)

    def get_state(self):
        """Returns the current state of this object.
        This method is called during `save`.
        """
        return self._state
        

    def set_state(self, state):
        """Sets the current state of this object.
        This method is called during `reload`.
        # Arguments:
          state: Dict. The state to restore for this object.
        """
        self._state = state
        
    @property
    def deserialized_state(self):
        state = copy.deepcopy(self.get_state())
        state['activation_type'] = ActivationLayers[state['activation_type']]
        state['pool_type'] = PoolingLayers[state['pool_type']]

#         state['activation_type'] = [ActivationLayers[act_layer] for act_layer in state['activation_type']]
#         state['pool_type'] = [PoolingLayers[pool_layer] for pool_layer in state['pool_type']]
        return state
        


# In[10]:


import copy

class ChromosomeOptions(stateful.Stateful): #BaseOptions): #object):
    """
    Container class for encapsulating variable-length lists of potential gene variants (individual hyperparameters).
    To be used as a reservoir from which to sample a complete chromosome made up of 1 variant per gene.
    
    This should be logged for describing the scope of a given AutoML experiment's hyperparameter search space

    Gene: The unique identifier of a particular hyperparameter that may reference any of a set of possible variant values.
    Variant: The particular value of a gene. Used to refer to the 1 value for a single chromosome instance, or 1 value from a set of gene options.

    Args:
        NamedTuple ([type]): [description]
    """

    def __init__(self,
                 hparam_lists,
                 phase=0,
                 seed=None):
        
#         self.__chromosomes = {k:v for k,v in locals().items() if k not in ['self', 'kwargs'] and not k.startswith('__')}
#         print(self.__chromosomes)
        
        self.set_state(hparam_lists, phase=phase, seed=seed)

    def get_state(self):
        """Returns the current state of this object.
        This method is called during `save`.
        """
        return self.state
    
    def get_config(self):
        config = copy.deepcopy(self.state)
        return config
        

    def set_state(self, state, phase=0, seed=None):
        """Sets the current state of this object.
        This method is called during `reload`.
        # Arguments:
          state: Dict. The state to restore for this object.
        """
        self.set_seed(seed)
        self.phase = phase
        self.state = state

    def set_seed(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def sample_k_variants_from_gene(self, gene: str, k: int=1):
        '''
        Randomly sample the list of variants corresponding to the key indicated by the first arg, 'gene'. Produce a random sequence of length k, with the default==1.
        
        Note: If k==1: this automatically returns a single unit from the variants list, which may or may not be a scalar object (e.g. int, str, float)
        If k > 1: then the sampled variants will always be returned in a list.
        
        '''
        all_variants = self.chromosomes[gene]
        variant_idx = self.rng.integers(low=0, high=len(all_variants), size=k)
        sampled_variants = [all_variants[idx] for idx in variant_idx.tolist()]
        if k==1:
            sampled_variants = sampled_variants[0]
        return sampled_variants
    
    def generate_chromosome(self, phase: int=None, seed=None):
        '''
        Primary function for utilizing a ChromosomeOptions object during experimentation.
        Running this function will randomly generate a new Chromosome instance for which each genetic variant is randomly sampled from this object's contained data,
        in the form of mappings between gene names as keys, and lists of variants as values.
        '''
        return Chromosome(hparams={gene:self.sample_k_variants_from_gene(gene) for gene in self.chromosomes.keys()})
    
    def generate_k_chromosomes(self, k: int=1, seed=None):
        return [self.generate_chromosome(seed=seed) for _ in range(k)]
        
    @property
    def chromosomes(self):
        return self.state

    
    @property
    def deserialized_state(self):
        state = copy.deepcopy(self.state)
        state['activation_type'] = [ActivationLayers[act_layer] for act_layer in state['activation_type']]
        state['pool_type'] = [PoolingLayers[pool_layer] for pool_layer in state['pool_type']]
        return state
    

class ChromosomeSampler:
    
    def __call__(self, phase: int):
        
        if phase==0:
            options = ChromosomeOptions({
#                                       'b_include_layer':[True],
                                      'a_filter_size':[(1,1), (3,3), (5,5), (7,7), (9,9)],
                                      'a_include_BN':[True, False],
                                      'a_output_channels':[8, 16, 32, 64, 128, 256, 512],
                                      'activation_type':['ReLU', 'ELU', 'LeakyReLU'],
                                      'b_filter_size':[(1,1), (3,3), (5,5), (7,7), (9,9)],
                                      'b_include_BN':[True, False],
                                      'b_output_channels':[8, 16, 32, 64, 128, 256, 512],
                                      'include_pool':[True, False],
                                      'pool_type':['MaxPool2D', 'AveragePooling2D'],
                                      'include_skip':[True, False]
                                      },
                                      phase=phase)

        else:
            options = ChromosomeOptions({
                                      'b_include_layer':[True, False],
                                      'a_filter_size':[(1,1), (3,3), (5,5), (7,7), (9,9)],
                                      'a_include_BN':[True, False],
                                      'a_output_channels':[8, 16, 32, 64, 128, 256, 512],
                                      'activation_type':['ReLU', 'ELU', 'LeakyReLU'],
                                      'b_filter_size':[(1,1), (3,3), (5,5), (7,7), (9,9)],
                                      'b_include_BN':[True, False],
                                      'b_output_channels':[8, 16, 32, 64, 128, 256, 512],
                                      'include_pool':[True, False],
                                      'pool_type':['MaxPool2D', 'AveragePooling2D'],
                                      'include_skip':[True, False]
                                      },
                                      phase=phase)
        return options.generate_chromosome(phase=phase)


# In[11]:


phases = []
sampler=ChromosomeSampler()
phases.append(sampler(phase=0))
phases.append(sampler(phase=1))


# ## Schema for defining, loading, using, and logging configuration for hparam search
# 
# 
# ### 1. Begin Hooks
# 
#     a. at_search_begin
#         Store hparam search space definitions in a file called `search_space.json`
#     b. at_trial_begin
# 
#     c. at_train_begin
# 
#     d. at_epoch_begin
# 
#     e. at_batch_begin
# 
# ### 2. End Hooks
# 
#     a. at_batch_end
#     
#     b. at_epoch_end
# 
#     c. at_train_end
# 
#     d. at_trial_end
# 
#     e. at_search_end
# 
# 
# 7. 

# ## 1. INTERESTING REFACTOR IDEA:
#     TODO: Refactor chromosome structure to standardize the configuration options for repeated model structures
#     ### (3 AM 11/27/20)
# 
#     e.g. Create a separate ConvOptions(NamedTuple) class to contain all 3 options:
#         filter_size
#     include_BN
#     output_channels
# 
#     Then in each "ChromosomeOptions" (consider making each of those a chromosome, and upgrading what's now a chromosome to a full Genome)
#     store a separate ConvOptions for layer a and layer b, separately.
# 
# 
# ## 2. TODO: 
#     Consider transferring mutate() method from Organism to Chromosome, while potentially keeping crossover() method as part of organism's namespace. Purpose is to encapsulate functionality as close as possible with the data/abstractions it will operate on
# 
# 
# ## 3. To Consider:
#     How can I quantify the information coverage and computational complexity of a given set of chromosome options? 
# 
#         a. Start with the raw # of permutations of all chromosome options
#         b. Adjust by the expected coverage for each variant. E.g. How much of the hyperparameter space are we covering in our naive uniform grid search?

# In[15]:


import gc


class Organism:
    def __init__(self,
                 data: Dict[str,tf.data.Dataset],
                 config=None,
                 chromosome=None,
                 phase=0,
                 generation_number=0,
                 organism_id=0,
                 best_organism=None):
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
        return [tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')]
   
    @property
    def chromosome(self):
        return self._chromosome.deserialized_state
    
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
        print('FFITNESS FUNCTION FFS')
        print('vars():', vars())
        self.run = wandb.init(**self.get_wandb_credentials(phase=self.phase,
                                                generation_number=generation_number),
                   config=self.config)
        
        self.model.fit(train_data,
                       steps_per_epoch=self.config.steps_per_epoch,
                       epochs=self.config.epochs_per_organism,
                       callbacks=[WandbCallback()],
                       verbose=1)
        _, self.fitness = self.model.evaluate(val_data,
                                              steps=self.config.validation_steps,
                                              verbose=1)
        
        
#     def fitnessFunction(self,
#                         train_data,
#                         val_data,
#                         generation_number):
#         '''
#         This function is used to calculate the
#         fitness of an individual.
#         '''
#         print('FFITNESS FUNCTION FFS')
#         print('vars():', vars())
#         wandb.init(**self.get_wandb_credentials(phase=self.phase,
#                                                generation_number=generation_number))
        
#         self.model.fit(train_data,
#                        steps_per_epoch=self.config.steps_per_epoch,
#                        epochs=self.config.epochs_per_organism,
#                        callbacks=[WandbCallback()],
#                        verbose=1)
#         _, self.fitness = self.model.evaluate(val_data,
#                                               steps=self.config.validation_steps,
#                                               verbose=1)
    def crossover(self,
                  partner,
                  generation_number):
        '''
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
                          project=f"vlga-plant_village"):
    
    return dict(entity=entity,
                project=project,
                group='KAGp{}'.format(phase),
                job_type='g{}'.format(generation_number))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# # Generation
# This is a class that hold generations of models.
# 
# 1. fitSurvivalRate - The amount of fit individuals we want in the next generation.
# 2. unfitSurvivalProb - The probability of sending unfit individuals
# 3. mutationRate - The mutation rate to change genes in an individual.
# 4. phase - The phase that the generation belongs to.
# 5. population_size - The amount of individuals that the generation consists of.
# 6. best_organism - The best organism (individual) is the last phase

# In[16]:


class Generation:
    def __init__(self,
                 data,
                 generation_config,
                 organism_config,
                 phase,
                 previous_best_organism,
                 verbose: bool=False):
        self.data = data
        self.config = generation_config
        self.organism_config = organism_config
        self.population = []
        self.generation_number = 0
        self.phase = phase
        # creating the first population: GENERATION_0
        # can be thought of as the setup function
        self.previous_best_organism = previous_best_organism or None
        self.best = {}
        self._initialized = False
        self.initialize_population(verbose=verbose)
        self.verbose = verbose
        
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config=None):
        config = config or OmegaConf.create({})
        config.population_size = config.population_size or 5
        config.num_generations_per_phase = config.num_generations_per_phase or 3
        config.fitSurvivalRate = config.fitSurvivalRate or 0.5
        config.unfitSurvivalProb = config.unfitSurvivalProb or 0.2
        config.mutationRate = config.mutationRate or 0.1
        config.num_phases = config.num_phases or 5
        
        self._config = config
        self.__dict__.update(config)
        
        
    def initialize_population(self, verbose=True):
        '''
        1. Create self.population_size individual organisms from scratch by randomly sampling an initial set of hyperparameters (a chromosome)
        2. As each is instantiated, build its model
        3. Assess their fitness one-by-one
        4. Sort models by relative fitness so we have a (potentially) new Best Organism (best model)
        4. Increment generation number to 1
        '''

        for idx in range(self.population_size):
            if verbose:
                print('<'*10,' '*5,'>'*10)
                print(f'Creating, training then testing organism {idx} out of a maximum {self.population_size} from generation {self.generation_number} and phase {self.phase}')
            org = Organism(chromosome=sampler(self.phase), #.get_state(),
                           data=self.data,
                           config=self.organism_config,
                           phase=self.phase,
                           generation_number=self.generation_number,
                           organism_id=idx,
                           best_organism=self.previous_best_organism)
            org.build_model()
            org.fitnessFunction(org.data['train'],
                                org.data['test'],
                                generation_number=self.generation_number)
            self.population.append(org)

        self._initialized = True
        self.sortModel(verbose=verbose)
        self.generation_number += 1
        self.evaluate(run=self.population[0].run)

    def sortModel(self, verbose: bool=True):
        '''
        sort the models according to the 
        fitness in descending order.
        '''
        previous_best = self.best_fitness
        fitness = [ind.fitness for ind in self.population]
        sort_index = np.argsort(fitness)[::-1]
        self.population = [self.population[index] for index in sort_index]

        if self.best_organism_so_far.fitness > previous_best:
            self.best['organism'] = self.best_organism_so_far
            self.best['model'] = self.best_organism_so_far.model
            self.best['fitness'] = self.best_organism_so_far.fitness
            
            if verbose:
                print(f'''NEW BEST MODEL:
                Fitness = {self.best["fitness"]:.3f}
                Previous Fitness = {previous_best:.3f}
                Name = {self.best['organism'].name}
                chromosome = {self.best['organism'].chromosome}''')
        
    @property
    def best_organism_so_far(self):
        if self._initialized:
            return self.population[0]
        else:
            return self.previous_best_organism

    @property
    def best_fitness(self):
        if self._initialized:
            return self.population[0].fitness
        elif self.previous_best_organism is not None:
            return self.previous_best_organism.fitness
        else:
            return 0.0
        
        
    def generate(self):
        '''
        Generate a new generation in the same phase
        '''
        number_of_fit = int(self.population_size * self.fitSurvivalRate)
        new_pop = self.population[:number_of_fit]
        for individual in self.population[number_of_fit:]:
            if np.random.rand() <= self.unfitSurvivalProb:
                new_pop.append(individual)
        for index, individual in enumerate(new_pop):
            if np.random.rand() <= self.mutationRate:
                new_pop[index].mutation(generation_number=self.generation_number)
        fitness = [ind.fitness for ind in new_pop]
        children=[]
        for idx in range(self.population_size-len(new_pop)):
            parents = np.random.choice(new_pop, replace=False, size=(2,), p=softmax(fitness))
            A=parents[0]
            B=parents[1]
            child=A.crossover(B, generation_number=self.generation_number)
            children.append(child)
        self.population = new_pop+children
        self.sortModel()
        self.generation_number+=1

    def evaluate(self, run=None, last=False):
        '''
        Evaluate the generation
        '''
        print('EVALUATE')
        fitness = [ind.fitness for ind in self.population]

        BestOrganism = self.population[0]
        if run is None:
            run = BestOrganism.run
        run.log({'population_size':len(fitness)}, commit=False)
        run.log({'Best fitness': fitness[0]}, commit=False)
        run.log({'Average fitness': sum(fitness)/len(fitness)})
        
        self.population[0].show()
        print('BEST ORGANISM', BestOrganism.name)
        k=16
        if last:
            k=32
        model_path = f'best-model-phase_{self.phase}.png'
        tf.keras.utils.plot_model(BestOrganism.model, to_file=model_path)
        run.log({"best_model": [wandb.Image(model_path, caption=f"Best Model phase_{self.phase}")]})
        log_high_loss_examples(BestOrganism.test_dataset,
                               BestOrganism.model, 
                               k=k,
                               run=run)
            
        return BestOrganism

    def run_generation(self):
        print('RUN GENERATION')
        self.generate()
        last = False
        if self.generation_number == self.num_generations_per_phase:
            last = True
        best_organism = self.evaluate(last=last)
        return best_organism
        
    def run_phase(self):#, num_generations_per_phase: int=1):
        print('RUN PHASE')
        while self.generation_number < self.num_generations_per_phase:
            best_organism = self.run_generation()
            print(f'FINISHED GENERATION {self.generation_number}')
            print(vars())
            
            if self.verbose:
                print(f'FINISHED generation {self.generation_number}. Best fitness = {best_organism.fitness}')
            
        return self.population[0] #best_organism
        
#             return self.population[0]


# In[ ]:


best_organism = None
for phase in range(config.generation.num_phases):
    print("PHASE {}".format(phase))
    generation = Generation(data=data,
                            generation_config=config['generation'],
                            organism_config=config['organism'],
                            phase=phase,
                            previous_best_organism=best_organism,
                            verbose=VERBOSE)
    
    best_organism = generation.run_phase()


# In[ ]:


### 1. Using tfds.features.ClassLabel

# feature_labels = tfds.features.ClassLabel(names=vocab)
# data = ['Potato___healthy',
#         'Potato___Late_blight',
#         'Raspberry___healthy',
#         'Soybean___healthy',
#         'Squash___Powdery_mildew',
#         'Strawberry___healthy',
#         'Strawberry___Leaf_scorch',
#         'Tomato___Bacterial_spot',
#         'Tomato___Early_blight',
#         'Tomato___healthy']

# data += data[::-1]
# print([feature_labels.str2int(label) for label in data])
# data = train_data
# data_enc = data.map(lambda x,y: (x, feature_labels.int2str(y)))

### 2. Using StringLookup and CategoryEncoding Layers

# layer = StringLookup(vocabulary=vocab, num_oov_indices=0, mask_token=None)
# i_layer = StringLookup(vocabulary=layer.get_vocabulary(), invert=True)
# int_data = layer(data)

# print(len(layer.get_vocabulary()))
# print(len(class_encoder.class_list))
# print(set(layer.get_vocabulary())==set(class_encoder.class_list))

# i_layer = StringLookup(vocabulary=layer.get_vocabulary(), invert=True)
# int_data = layer(data)

# print(layer(data))
# print(i_layer(int_data))


# In[ ]:


# # from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding
# # data = tf.constant(["a", "b", "c", "b", "c", "a"])
# # # Use StringLookup to build an index of the feature values
# # indexer = StringLookup()
# # indexer.adapt(data)
# # # Use CategoryEncoding to encode the integer indices to a one-hot vector
# # encoder = CategoryEncoding(output_mode="binary")
# # encoder.adapt(indexer(data))
# # # Convert new test data (which includes unknown feature values)
# # test_data = tf.constant(["a", "b", "c", "d", "e", ""])
# # encoded_data = encoder(indexer(test_data))
# # print(encoded_data)

# vocab = ["a", "b", "c", "d"]
# data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
# layer = StringLookup(vocabulary=vocab)
# i_layer = StringLookup(vocabulary=layer.get_vocabulary(), invert=True)
# int_data = layer(data)

# print(layer(data))
# print(i_layer(int_data))

