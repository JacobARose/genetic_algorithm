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

    
import wandb
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from genetic_algorithm.organism.organism import Organism
from genetic_algorithm.plotting import log_high_loss_examples
from genetic_algorithm.datasets.plant_village import ClassLabelEncoder, load_and_preprocess_data
from genetic_algorithm import stateful
from genetic_algorithm.chromosome import sampler #ChromosomeSampler





    
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
            
        self.run = wandb.init(**BestOrganism.get_wandb_credentials(phase=BestOrganism.phase,
                              generation_number=BestOrganism.generation_number),
                              resume='allow',
                              tags=['evaluate'],
                              id=run.id)
        self.run_id = self.run.id
        
        with self.run:

            self.run.log({'population_size':len(fitness)}, commit=False)
            self.run.log({'Best fitness': fitness[0]}, commit=False)
            self.run.log({'Average fitness': sum(fitness)/len(fitness)}, commit=False)

            self.population[0].show()
            print('BEST ORGANISM', BestOrganism.name)
    #         k=16
            if last:
#                 import ipdb;ipdb.set_trace()
                k=64
                model_path = f'best-model-phase_{self.phase}.png'
                tf.keras.utils.plot_model(BestOrganism.model, to_file=model_path)
                model_structure_image = [wandb.Image(model_path, caption=f"Best Model phase_{self.phase}")]
                run.log({"best_model": model_structure_image}, commit=False)
                log_high_loss_examples(BestOrganism.test_data,
                                       BestOrganism.model, 
                                       k=k,
                                       run=self.run)

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
            print(f'FINISHED GENERATION {self.generation_number}')
            print(vars())
            
            if self.verbose:
                print(f'FINISHED generation {self.generation_number}. Best fitness = {best_organism.fitness}')
            
        return self.population[0] #best_organism