
VERBOSE = True
import pandas as pd
import json
from box import Box
from bunch import Bunch
import copy
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, ELU, LeakyReLU, Flatten, Dense, Add, AveragePooling2D, GlobalAveragePooling2D
from typing import List, Tuple, Union, Dict, NamedTuple
from genetic_algorithm import stateful


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
        state = copy.deepcopy(state)
                state['activation_type'] = ActivationLayers[state['activation_type']]
        state['pool_type'] = PoolingLayers[state['pool_type']]
        if isinstance(state['activation_type'], str)
        
        self._state = state
        
    def get_chromosome(self, serialized=False):
        if serialized:
            return self.get_state()
        state = copy.deepcopy(self.get_state())
        state['activation_type'] = ActivationLayers[state['activation_type']]
        state['pool_type'] = PoolingLayers[state['pool_type']]
        return state
        
#     @property
#     def deserialized_state(self):
#         state = copy.deepcopy(self.get_state())
#         state['activation_type'] = ActivationLayers[state['activation_type']]
#         state['pool_type'] = PoolingLayers[state['pool_type']]

#         state['activation_type'] = [ActivationLayers[act_layer] for act_layer in state['activation_type']]
#         state['pool_type'] = [PoolingLayers[pool_layer] for pool_layer in state['pool_type']]
#         return state



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
    def chromosomes(self, serialized=False):
        return self.state

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
