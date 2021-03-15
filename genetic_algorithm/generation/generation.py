#!/usr/bin/env python
# coding: utf-8




'''


Created: 12-01-2020 by Jacob Rose


Description: Segmented code for Generation class definition. To be used by main scripts and jupyter notebooks.


python /media/data/jacob/GitHub/genetic_algorithm/Notebooks/generation/generation.py



'''





# def get_hardest_k_examples(test_dataset, model, k=32):
#     class_probs = model.predict(test_dataset)
#     predictions = np.argmax(class_probs, axis=1)
#     losses = tf.keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
#     argsort_loss =  np.argsort(losses)

#     highest_k_losses = np.array(losses)[argsort_loss[-k:]]
#     hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
#     true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

#     return highest_k_losses, hardest_k_examples, true_labels, predictions
        
# def log_high_loss_examples(test_dataset, model, k=32):
#     print(f'logging k={k} hardest examples')
#     losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
#     wandb.log(
#         {"high-loss-examples":
#                             [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
#                              for hard_example, label, pred, loss in zip(hardest_k_examples, true_labels, predictions, losses)]
#         })

import cv2
import wandb
import os
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from genetic_algorithm.organism.organism import Organism
from genetic_algorithm.plotting import log_high_loss_examples, log_multiclass_metrics, log_classification_report
from genetic_algorithm.datasets.plant_village import ClassLabelEncoder, load_and_preprocess_data
from genetic_algorithm import stateful
from genetic_algorithm.chromosome import sampler #ChromosomeSampler

import logging
logger = logging.getLogger(__name__)




    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Generation:
    def __init__(self,
                 data,
                 generation_config,
                 organism_config,
                 phase,
                 previous_best_organism,
                 class_encoder=None,
                 verbose: bool=False,
                 initialize: bool=True,
                 debug=False):
        self.data = data
        self.config = generation_config
        self.organism_config = organism_config
        self.population = []
        self.generation_number = 0
        self.phase = phase
        # creating the first population: GENERATION_0
        # can be thought of as the setup function
        self.previous_best_organism = previous_best_organism or None
        self.class_encoder = class_encoder
        self.best = {}
        self._initialized = False
        self.debug = debug
        if initialize:
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
        
    @property
    def name(self):
        return f'phase_{self.phase}-gen_{self.generation_number}--contains_{self.population_size}_organisms'
    
    def __repr__(self):
        return f'<Generation object[{self.name}]>'
        
    def initialize_population(self, verbose=True):
        '''
        1. Create self.population_size individual organisms from scratch by randomly sampling an initial set of hyperparameters (a chromosome)
        2. As each is instantiated, build its model
        3. Assess their fitness one-by-one
        4. Sort models by relative fitness so we have a (potentially) new Best Organism (best model)
        4. Increment generation number to 1
        '''
        if self._initialized:
            print('Population has already been initialized, passing through without action')
            return

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
                           best_organism=self.previous_best_organism,
                           class_encoder=self.class_encoder,
                           debug=self.debug)
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
        #Ensure that at least 2 individuals survive to reproduce the next generation
        number_of_fit = max([int(self.population_size * self.fitSurvivalRate), 2])
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
            
        self.run = wandb.init(**BestOrganism.get_wandb_credentials(phase=BestOrganism.phase,
                              generation_number=BestOrganism.generation_number),
                              resume='allow',
                              tags=['evaluate', BestOrganism.name, BestOrganism.config.experiment_uid],
                              id=run.id)
        self.run_id = self.run.id
        
        with self.run:
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

        return BestOrganism

    def run_generation(self):
        print(f'RUN GENERATION {self.generation_number}')
        self.generate()
        last = False
        if self.generation_number == self.num_generations_per_phase:
            last = True
        best_organism = self.evaluate(last=last)
        
        
        return best_organism
        
    def run_phase(self):#, num_generations_per_phase: int=1):
        print('\n'*2,f'RUN PHASE {self.phase}')
        while self.generation_number < self.num_generations_per_phase:
            best_organism = self.run_generation()
            
#             best_organism.log_model()
            print(f'FINISHED GENERATION {self.generation_number}')
            print(vars())
            
            if self.verbose:
                print(f'FINISHED generation {self.generation_number}. Best fitness = {best_organism.fitness}')
            
        return self.population[0] #best_organism
    
    
    
    
    
    
    
    

# import os

# import tensorflow as tf

# from utils.logger import get_logger


# https://github.com/The-AI-Summer/Deep-Learning-In-Production/blob/master/7.How%20to%20build%20a%20custom%20production-ready%20Deep%20Learning%20Training%20loop%20in%20Tensorflow%20from%20scratch/executor/unet_trainer.py


# LOG = get_logger('trainer')


# class UnetTrainer:

#     def __init__(self, model, input, loss_fn, optimizer, metric, epoches):
#         self.model = model
#         self.input = input
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
#         self.metric = metric
#         self.epoches = epoches

#         self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#         self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=3)

#         self.train_log_dir = 'logs/gradient_tape/'
#         self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

#         self.model_save_path = 'saved_models/'

#     def train_step(self, batch):
#         trainable_variables = self.model.trainable_variables
#         inputs, labels = batch
#         with tf.GradientTape() as tape:
#             predictions = self.model(inputs)
#             step_loss = self.loss_fn(labels, predictions)

#         grads = tape.gradient(step_loss, trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, trainable_variables))
#         self.metric.update_state(labels, predictions)

#         return step_loss, predictions

#     def train(self):
#         for epoch in range(self.epoches):
#             LOG.info(f'Start epoch {epoch}')

#             step_loss = 0
#             for step, training_batch in enumerate(self.input):
#                 step_loss, predictions = self.train_step(training_batch)
#                 LOG.info("Loss at step %d: %.2f" % (step, step_loss))

#             train_acc = self.metric.result()
#             LOG.info("Training acc over epoch: %.4f" % (float(train_acc)))

#             save_path = self.checkpoint_manager.save()
#             LOG.info("Saved checkpoint: {}".format(save_path))

#             self._write_summary(step_loss, epoch)

#             self.metric.reset_states()

#         save_path = os.path.join(self.model_save_path, "unet/1/")
#         tf.saved_model.save(self.model, save_path)

#     def _write_summary(self, loss, epoch):
#         with self.train_summary_writer.as_default():
#             tf.summary.scalar('loss', loss, step=epoch)
#             tf.summary.scalar('accuracy', self.metric.result(), step=epoch)
#             # tensorboard --logdir logs/gradient_tape





# from abc import ABC, abstractmethod

# from utils.config import Config


# class BaseModel(ABC):
#     """Abstract Model class that is inherited to all models"""

#     def __init__(self, cfg):
#         self.config = Config.from_json(cfg)

#     @abstractmethod
#     def load_data(self):
#         pass

#     @abstractmethod
#     def build(self):
#         pass

#     @abstractmethod
#     def train(self):
#         pass

#     @abstractmethod
#     def evaluate(self):
#         pass








###############################################


# # -*- coding: utf-8 -*-
# """Unet model"""

# # standard library

# # external
# import tensorflow as tf
# from tensorflow_examples.models.pix2pix import pix2pix

# from dataloader.dataloader import DataLoader
# from utils.logger import get_logger
# from executor.unet_trainer import UnetTrainer

# # internal
# from .base_model import BaseModel

# LOG = get_logger('unet')


# class UNet(BaseModel):
#     """Unet Model Class"""

#     def __init__(self, config):
#         super().__init__(config)
#         self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
#         self.model = None
#         self.output_channels = self.config.model.output

#         self.dataset = None
#         self.info = None
#         self.batch_size = self.config.train.batch_size
#         self.buffer_size = self.config.train.buffer_size
#         self.epoches = self.config.train.epoches
#         self.val_subsplits = self.config.train.val_subsplits
#         self.validation_steps = 0
#         self.train_length = 0
#         self.steps_per_epoch = 0

#         self.image_size = self.config.data.image_size
#         self.train_dataset = []
#         self.test_dataset = []

#     def load_data(self):
#         """Loads and Preprocess data """
#         LOG.info(f'Loading {self.config.data.path} dataset...')
#         self.dataset, self.info = DataLoader().load_data(self.config.data)
#         self.train_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.batch_size,
#                                                                            self.buffer_size, self.image_size)
#         self._set_training_parameters()

#     def _set_training_parameters(self):
#         """Sets training parameters"""
#         self.train_length = self.info.splits['train'].num_examples
#         self.steps_per_epoch = self.train_length // self.batch_size
#         self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits

#     def build(self):
#         """ Builds the Keras model based """
#         layer_names = [
#             'block_1_expand_relu',  # 64x64
#             'block_3_expand_relu',  # 32x32
#             'block_6_expand_relu',  # 16x16
#             'block_13_expand_relu',  # 8x8
#             'block_16_project',  # 4x4
#         ]
#         layers = [self.base_model.get_layer(name).output for name in layer_names]

#         # Create the feature extraction model
#         down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

#         down_stack.trainable = False

#         up_stack = [
#             pix2pix.upsample(self.config.model.up_stack.layer_1, self.config.model.up_stack.kernels),  # 4x4 -> 8x8
#             pix2pix.upsample(self.config.model.up_stack.layer_2, self.config.model.up_stack.kernels),  # 8x8 -> 16x16
#             pix2pix.upsample(self.config.model.up_stack.layer_3, self.config.model.up_stack.kernels),  # 16x16 -> 32x32
#             pix2pix.upsample(self.config.model.up_stack.layer_4, self.config.model.up_stack.kernels),  # 32x32 -> 64x64
#         ]

#         inputs = tf.keras.layers.Input(shape=self.config.model.input)
#         x = inputs

#         # Downsampling through the model
#         skips = down_stack(x)
#         x = skips[-1]
#         skips = reversed(skips[:-1])

#         # Upsampling and establishing the skip connections
#         for up, skip in zip(up_stack, skips):
#             x = up(x)
#             concat = tf.keras.layers.Concatenate()
#             x = concat([x, skip])

#         # This is the last layer of the model
#         last = tf.keras.layers.Conv2DTranspose(
#             self.output_channels, self.config.model.up_stack.kernels, strides=2,
#             padding='same')  # 64x64 -> 128x128

#         x = last(x)

#         self.model = tf.keras.Model(inputs=inputs, outputs=x)

#         LOG.info('Keras Model was built successfully')

#     def train(self):
#         """Compiles and trains the model"""
#         LOG.info('Training started')
#         optimizer = tf.keras.optimizers.Adam()
#         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         metrics = tf.keras.metrics.SparseCategoricalAccuracy()

#         trainer = UnetTrainer(self.model, self.train_dataset, loss, optimizer, metrics, self.epoches)
#         trainer.train()

#     def evaluate(self):
#         """Predicts resuts for the test dataset"""

#         predictions = []
#         LOG.info(f'Predicting segmentation map for test dataset')

#         for im in self.test_dataset.as_numpy_iterator():
#             DataLoader().validate_schema(im[0])
#             break

#         for image, mask in self.test_dataset:
#             tf.print(image)
#             # LOG.info(f'Predicting segmentation map {image}')
#             predictions.append(self.model.predict(image))
#         return predictions
#################################################################

  
# import tensorflow as tf
# import numpy as np

# from utils.plot_image import display

# from utils.config import Config

# from configs.config import CFG


# class UnetInferrer:
#     def __init__(self):
#         self.config = Config.from_json(CFG)
#         self.image_size = self.config.data.image_size

#         self.saved_path = '/home/aisummer/src/soft_eng_for_dl/saved_models/unet'
#         self.model = tf.saved_model.load(self.saved_path)

#         # print(list(    self.model.signatures.keys()))

#         self.predict = self.model.signatures["serving_default"]
#         # print(self.predict.structured_outputs)

#     def preprocess(self, image):
#         image = tf.image.resize(image, (self.image_size, self.image_size))
#         return tf.cast(image, tf.float32) / 255.0

#     def infer(self, image=None):
#         tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
#         tensor_image = self.preprocess(tensor_image)
#         shape= tensor_image.shape
#         tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
#         print(tensor_image.shape)
#         pred = self.predict(tensor_image)['conv2d_transpose_4']
#         display([tensor_image[0], pred[0]])
#         pred = pred.numpy().tolist()
#         return {'segmentation_output':pred}




# ##################################################################

# """Config class"""

# import json


# class Config:
#     """Config class which contains data, train and model hyperparameters"""

#     def __init__(self, data, train, model):
#         self.data = data
#         self.train = train
#         self.model = model

#     @classmethod
#     def from_json(cls, cfg):
#         """Creates config from json"""
#         params = json.loads(json.dumps(cfg), object_hook=HelperObject)
#         return cls(params.data, params.train, params.model)


# class HelperObject(object):
#     """Helper class to convert json into Python object"""
#     def __init__(self, dict_):
#         self.__dict__.update(dict_)
        
        
        
#         """ main.py """

# from configs.config import CFG
# from model.unet import UNet


# def run():
#     """Builds model, loads data, trains and evaluates"""
#     model = UNet(CFG)
#     model.load_data()
#     model.build()
#     # model.train()
#     model.evaluate()


# if __name__ == '__main__':
#     run()

    
    
    
    
    
    
# import matplotlib.pyplot as plt
# import tensorflow as tf


# def display(display_list):
#     plt.figure(figsize=(15, 15))

#     title = ['Input Image', 'Predicted Mask']

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         plt.axis('off')
#     plt.show()
    
    
    
    
    
    
  
# from unittest.mock import patch

# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds

# from configs.config import CFG
# from model.unet import UNet


# def dummy_load_data(*args, **kwargs):
#     with tfds.testing.mock_data(num_examples=1):
#         return tfds.load(CFG['data']['path'], with_info=True)


# class UnetTest(tf.test.TestCase):

#     def setUp(self):
#         super(UnetTest, self).setUp()
#         self.unet = UNet(CFG)

#     def tearDown(self):
#         pass

#     def test_normalize(self):
#         input_image = np.array([[1., 1.], [1., 1.]])
#         input_mask = 1
#         expected_image = np.array([[0.00392157, 0.00392157], [0.00392157, 0.00392157]])

#         result = self.unet._normalize(input_image, input_mask)
#         self.assertAllClose(expected_image, result[0])

#     def test_ouput_size(self):
#         shape = (1, self.unet.image_size, self.unet.image_size, 3)
#         image = tf.ones(shape)
#         self.unet.build()
#         self.assertEqual(self.unet.model.predict(image).shape, shape)

#     @patch('model.unet.DataLoader.load_data')
#     def test_load_data(self, mock_data_loader):
#         mock_data_loader.side_effect = dummy_load_data
#         shape = tf.TensorShape([None, self.unet.image_size, self.unet.image_size, 3])

#         self.unet.load_data()
#         mock_data_loader.assert_called()

#         self.assertItemsEqual(self.unet.train_dataset.element_spec[0].shape, shape)
#         self.assertItemsEqual(self.unet.test_dataset.element_spec[0].shape, shape)


# if __name__ == '__main__':
#     tf.test.main()

# # coverage run -m unittest /home/aisummer/PycharmProjects/Deep-Learning-Production-Course/model/tests/unet_test.py
# # coverage report -m  /home/aisummer/PycharmProjects/Deep-Learning-Production-Course/model/tests/unet_test.py
