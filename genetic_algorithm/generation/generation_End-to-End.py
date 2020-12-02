#!/usr/bin/env python
# coding: utf-8




'''


Created: 11-26-2020 by Jacob Rose


Description: End-to-End prototype command line script for Evolutionary Algorithm


python /media/data/jacob/GitHub/genetic_algorithm/Notebooks/generation/generation.py



'''

def get_hardest_k_examples(test_dataset, model, k=32):
    class_probs = model.predict(test_dataset)
    predictions = np.argmax(class_probs, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
    argsort_loss =  np.argsort(losses)

    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
    true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

    return highest_k_losses, hardest_k_examples, true_labels, predictions
        
def log_high_loss_examples(test_dataset, model, k=32):
    print(f'logging k={k} hardest examples')
    losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
    wandb.log(
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

from typing import List, Tuple, Union, Dict
import tensorflow_datasets as tfds
from omegaconf import OmegaConf

from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial

# import logging
# logger = logging.getLogger('')

LOG_DIR = '/media/data/jacob/GitHub/evolution_logs'
import os
os.makedirs(LOG_DIR, exist_ok=True)
from paleoai_data.utils.logging_utils import get_logger
logger = get_logger(logdir=LOG_DIR, filename='generation_evolution_logs.log', append=True)

TFDS_DATASETS = ['plant_village']

class ClassLabelEncoder:
    def __init__(self, ds_info: tfds.core.dataset_info.DatasetInfo):
        self.info = ds_info
        self.dataset_name = ds_info.full_name
        self.num_samples = ds_info.splits['train'].num_examples
        self.num_classes = ds_info.features['label'].num_classes
        self.class_list = ds_info.features['label'].names
        self._str2int = ds_info.features['label'].str2int
        self._int2str = ds_info.features['label'].int2str
        
    def str2int(self, labels: Union[List[str],Tuple[str]]):
        labels = _valid_eager_tensor(labels)
        if not isinstance(labels, [list, tuple]):
            assert isinstance(labels, str)
            labels = [labels]
        return [self._str2int(l) for l in labels]
    
    def int2str(self, labels: Union[List[int],Tuple[int]]):
        labels = _valid_eager_tensor(labels)
        if not isinstance(labels, [list, tuple]):
            assert isinstance(labels, (int, np.int64))
            labels = [labels]
        return [self._int2str(l) for l in labels]
    
    def one_hot(self, label: tf.int64):
        '''
        One-Hot encode integer labels
        Use tf.data.Dataset.map(lambda x,y: (x, encoder.one_hot(y))) and pass in individual labels already encoded in int64 format.
        '''
        return tf.one_hot(label, depth=self.num_classes)
    
    def __repr__(self):
        return f'''Dataset Name: {self.dataset_name}
        Num_samples: {self.num_samples}
        Num_classes: {self.num_classes}'''
    
    def _valid_eager_tensor(self, tensor, strict=False):
        '''
        If tensor IS an EagerTensor, return tensor.numpy(). 
        if strict==True, and tensor IS NOT an EagerTensor, then raise AssertionError.
        if strict==False, and tensor IS NOT an EagerTensor, then return tensor without modification 
        '''
        try:
            assert isinstance(labels, tf.python.framework.ops.EagerTensor)
        except AssertionError:
            if strict:
                raise AssertionError(f'Strict EagerTensor requirement failed assertion test in ClassLabelEncoder._valid_eager_tensor method')
        labels = labels.numpy()
        return labels

def load_plant_village_dataset(split=['train'],
                               data_dir=None,
                               batch_size=None):
    
    builder = tfds.builder('plant_village', data_dir=data_dir)
    ds_info = builder.info
    builder.download_and_prepare()

    print(f'splits: {split}')
    data = builder.as_dataset(split=list(split),
                              shuffle_files=True,
                              batch_size=batch_size,
                              as_supervised=True
                              )
    
    if not isinstance(data, (tuple, list)):
        data = {'train':data}
    elif len(data)==2:
        data = {'train':data[0], 'val':data[1]}
    elif len(data)==3:
        data = {'train':data[0], 'val':data[1], 'test':data[2]}
    
    return data, builder

def load_tfds_dataset(dataset_name='plant_village', 
                      split={'train':'train'},
                      data_dir=None,
                      batch_size=None):
    '''
    General interface function to properly route users to the correct function for loading their queried dataset from Tensorflow Datasets (TFDS) public data.
    '''
    assert dataset_name in TFDS_DATASETS
    
    print(f'Getting the TFDS dataset: {dataset_name}')
    if dataset_name == 'plant_village':
        return load_plant_village_dataset(split      =split,
                                          data_dir   =data_dir,
                                          batch_size =batch_size)
    else:
        raise Exception('Attempted to load dataset from TFDS that we have yet to build an adapter for. Consider building a minimal working prototype by using alternative datasets as a template.')

        
def rgb2gray_3channel(img):
    '''
    Convert rgb image to grayscale, but keep num_channels=3
    '''
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.grayscale_to_rgb(img)
    return img


def get_parse_example_func(target_size, num_classes, color_mode='rgb'):
    resize = resize_repeat(target_size=tuple(target_size), training=False)
    one_hot = partial(tf.one_hot, depth=num_classes)
    parse_color_channels = rgb2gray_3channel if color_mode=='grayscale' else lambda x: x
    def _parse_example(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = rgb2gray_3channel(x)
        x = resize(x)
        y = one_hot(y)
        return x,y
    return _parse_example

def preprocess_data(data: tf.data.Dataset, target_size=None, num_classes=None, batch_size=1, color_mode='rgb'):
    parse_example = get_parse_example_func(target_size=target_size, num_classes=num_classes, color_mode=color_mode)
    return data.map(lambda x,y: parse_example(x, y), num_parallel_calls=-1) \
               .shuffle(1024) \
               .batch(batch_size) \
               .prefetch(-1)

def load_and_preprocess_data(data_config):

    data, builder = load_tfds_dataset(dataset_name=data_config.load.dataset_name,
                                      split=data_config.load.split,
                                      data_dir=data_config.load.data_dir)

    data_info     = builder.info
    class_encoder = ClassLabelEncoder(data_info)
    print(class_encoder)
#     vocab = class_encoder.class_list
    preprocess = partial(preprocess_data,
                         batch_size=data_config.preprocess.batch_size,
                         target_size=data_config.preprocess.target_size,
                         num_classes=class_encoder.num_classes,
                         color_mode=data_config.preprocess.color_mode)

    data['train'] = preprocess(data=data['train']) #, batch_size=config.batch_size)
    data['val'] = preprocess(data=data['val']) #, batch_size=config.batch_size)
    data['test'] = preprocess(data=data['test']) #, batch_size=config.batch_size)
    
    return data, class_encoder




#####################################

# if __name__=='__main__':
def main():

    exp_config = OmegaConf.create({'seed':756, #237,
                                   'batch_size':16,
                                   'input_shape':(224,224,3),
                                   'output_size':38,
                                   'epochs_per_organism':3
                                  })

    data_config = OmegaConf.create({'load':{},'preprocess':{}})

    data_config['load'] = {'dataset_name':'plant_village',
                           'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
                           'data_dir':'/media/data/jacob/tensorflow_datasets'}

    data_config['preprocess'] = {'batch_size':exp_config.batch_size,
                                 'target_size':exp_config.input_shape[:2],
                                 'color_mode':'grayscale'}

    organism_config = OmegaConf.create({'input_shape':exp_config.input_shape,
                                        'output_size':38,
                                        'epochs_per_organism':exp_config.epochs_per_organism})
    generation_config = OmegaConf.create({
                                          'population_size':5,
                                          'num_generations_per_phase':3,
                                          'fitSurvivalRate': 0.5,
                                          'unfitSurvivalProb':0.2,
                                          'mutationRate':0.1,
                                          'num_phases':5
                                        })

    data, class_encoder = load_and_preprocess_data(data_config)
        
        
        
    prevBestOrganism = None
    for phase in range(generation_config.num_phases):
        print("PHASE {}".format(phase))
        generation = Generation(data=data,
                                generation_config=generation_config,
                                organism_config=organism_config,
                                phase=phase,
                                prevBestOrganism=prevBestOrganism)
    #     while generation.generation_number < num_generations_per_phase:
        generation.generate()
        if generation.generation_number == generation.num_generations_per_phase:
            # print('I AM THE BEST IN THE PHASE')
            prevBestOrganism = generation.evaluate(last=True)
        else:
            generation.evaluate()
            
            


options_phase0 = {
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'activation_type': [ReLU, ELU, LeakyReLU],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D],
    'include_skip': [True, False]
}

options = {
    'include_layer': [True, False],
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D],
    'include_skip': [True, False]
}

def random_hyper(phase):
    if phase == 0:
        return {
        'a_filter_size': options_phase0['a_filter_size'][np.random.randint(len(options_phase0['a_filter_size']))],
        'a_include_BN': options_phase0['a_include_BN'][np.random.randint(len(options_phase0['a_include_BN']))],
        'a_output_channels': options_phase0['a_output_channels'][np.random.randint(len(options_phase0['a_output_channels']))],
        'activation_type': options_phase0['activation_type'][np.random.randint(len(options_phase0['activation_type']))],
        'b_filter_size': options_phase0['b_filter_size'][np.random.randint(len(options_phase0['b_filter_size']))],
        'b_include_BN': options_phase0['b_include_BN'][np.random.randint(len(options_phase0['b_include_BN']))],
        'b_output_channels': options_phase0['b_output_channels'][np.random.randint(len(options_phase0['b_output_channels']))],
        'include_pool': options_phase0['include_pool'][np.random.randint(len(options_phase0['include_pool']))],
        'pool_type': options_phase0['pool_type'][np.random.randint(len(options_phase0['pool_type']))],
        'include_skip': options_phase0['include_skip'][np.random.randint(len(options_phase0['include_skip']))]
        }
    else:
        return {
        'a_filter_size': options['a_filter_size'][np.random.randint(len(options['a_filter_size']))],
        'a_include_BN': options['a_include_BN'][np.random.randint(len(options['a_include_BN']))],
        'a_output_channels': options['a_output_channels'][np.random.randint(len(options['a_output_channels']))],
        'b_filter_size': options['b_filter_size'][np.random.randint(len(options['b_filter_size']))],
        'b_include_BN': options['b_include_BN'][np.random.randint(len(options['b_include_BN']))],
        'b_output_channels': options['b_output_channels'][np.random.randint(len(options['b_output_channels']))],
        'include_pool': options['include_pool'][np.random.randint(len(options['include_pool']))],
        'pool_type': options['pool_type'][np.random.randint(len(options['pool_type']))],
        'include_layer': options['include_layer'][np.random.randint(len(options['include_layer']))],
        'include_skip': options['include_skip'][np.random.randint(len(options['include_skip']))]
        }

        
        

    
def get_wandb_credentials(phase: int, generation_number: int):
    return dict(entity="jrose",
                project=f"vlga-plant_village-cmd",
                group='KAGp{}'.format(phase),
                job_type='g{}'.format(generation_number))


class Organism:
    def __init__(self,
                 data: Dict[str,tf.data.Dataset],
                 config=None,
                 chromosome={},
                 phase=0,
                 prevBestOrganism=None):
        '''
        config is a . accessible dict object containing model params that will stay constant during evolution
        chromosome is a dictionary of genes
        phase is the phase that the individual belongs to
        prevBestOrganism is the best organism of the previous phase
        
        TODO:
        
        1. implement to_json and from_json methods for copies
        2. Separate out step where organism is associated with a dataset
        '''
        self.data = data
        self.train_data = data['train']
        self.val_data = data['val']
        self.test_data = data['test']
        self.config = config
        self.phase = phase
        self.chromosome = chromosome
        self.prevBestOrganism=prevBestOrganism
        if phase != 0:
            # In a later stage, the model is made by
            # attaching new layers to the prev best model
            self.last_model = prevBestOrganism.model
    
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, config=None):
        config = config or OmegaConf.create({})
        config.input_shape = config.input_shape or (224,224,3)
        config.output_size = config.output_size or 38
        config.epochs_per_organism = config.epochs_per_organism or 5
        self._config = config
    
    def build_model(self):
        '''
        This is the function to build the keras model
        '''
        K.clear_session()
        inputs = Input(shape=self.config.input_shape)
        if self.phase == 0:
            x = Conv2D(filters=self.chromosome['a_output_channels'],
                       padding='same',
                       kernel_size=self.chromosome['a_filter_size'],
                       use_bias=self.chromosome['a_include_BN'])(inputs)
        else:
            # Slice the prev best model         # Use the model as a layer # Attach new layer to the sliced model
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
            # This is to ensure that we do not randomly chose another activation
            self.chromosome['activation_type'] = self.prevBestOrganism.chromosome['activation_type']

        if self.chromosome['a_include_BN']:
            x = BatchNormalization()(x)
        x = self.chromosome['activation_type']()(x)
        if self.chromosome['include_pool']:
            x = self.chromosome['pool_type'](strides=(1,1),
                                             padding='same')(x)
        if self.phase != 0 and self.chromosome['include_layer'] == False:
            # Except for PHASE0, there is a choice for
            # the number of layers that the model wants
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['a_output_channels'],
                           kernel_size=(1,1),
                           padding='same')(inter_inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.output_shape, activation='softmax')(x)
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
                           metrics=['accuracy'])
        
    def fitnessFunction(self,
                        train_data,
                        val_data,
                        generation_number):
        '''
        This function is used to calculate the
        fitness of an individual.
        '''
        wandb.init(**get_wandb_credentials(phase=self.phase,
                                           generation_number=generation_number))
        
        self.model.fit(train_data,
                       epochs=self.config.epochs_per_organism,
                       callbacks=[WandbCallback()],
                       verbose=1)
        _, self.fitness = self.model.evaluate(val_data,
                                              verbose=1)
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
                         prevBestOrganism=self.prevBestOrganism)
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
# 6. prevBestOrganism - The best organism (individual) is the last phase

# In[10]:


class Generation:
    def __init__(self,
                 data,
                 generation_config,
                 organism_config,
                 phase,
                 prevBestOrganism):
        self.data = data
        self.config = generation_config
        self.organism_config = organism_config
        self.population = []
        self.generation_number = 0
        self.phase = phase
        # creating the first population: GENERATION_0
        # can be thought of as the setup function
        self.prevBestOrganism = prevBestOrganism or None
        self.initialize_population()
        
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
        
        
    def initialize_population(self):
        '''
        1. Create self.population_size individual organisms from scratch by randomly sampling an initial set of hyperparameters (a chromosome)
        2. As each is instantiated, build its model
        3. Assess their fitness one-by-one
        4. Sort models by relative fitness so we have a (potentially) new Best Organism (best model)
        4. Increment generation number to 1
        '''

        for idx in range(self.population_size):
            print(f'Creating, training then testing organism {idx} of generation {self.generation_number} and phase {self.phase}')
            org = Organism(chromosome=random_hyper(self.phase),
                           data=self.data,
                           config=self.organism_config,
                           phase=self.phase,
                           prevBestOrganism=self.prevBestOrganism)
            org.build_model()
            org.fitnessFunction(org.data['train'],
                                org.data['test'],
                                generation_number=self.generation_number)
            self.population.append(org)

        # sorts the population according to fitness (high to low)
        self.sortModel()
        self.generation_number += 1

    def sortModel(self):
        '''
        sort the models according to the 
        fitness in descending order.
        '''
        fitness = [ind.fitness for ind in self.population]
        sort_index = np.argsort(fitness)[::-1]
        self.population = [self.population[index] for index in sort_index]

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

    def evaluate(self, last=False):
        '''
        Evaluate the generation
        '''
        fitness = [ind.fitness for ind in self.population]
        
        wandb.log({'population_size':len(fitness)}, commit=False)
        wandb.log({'Best fitness': fitness[0]}, commit=False)
        wandb.log({'Average fitness': sum(fitness)/len(fitness)})
        
        self.population[0].show()
        if last:
            BestOrganism = self.population[0]
            model_path = f'best-model-phase_{self.phase}.png'
            tf.keras.utils.plot_model(BestOrganism.model, to_file=model_path)
            wandb.log({"best_model": [wandb.Image(model_path, caption=f"Best Model phase_{self.phase}")]})
            log_high_loss_examples(BestOrganism.test_dataset,
                                   BestOrganism.model, 
                                   k=32)
            
            return BestOrganism

        


if __name__=='__main__':
    
    main()