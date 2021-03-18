import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, ELU, LeakyReLU, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D, Add

import numpy as np
import os
from pprint import pprint
import wandb
from wandb.keras import WandbCallback
np.random.seed(666)
tf.random.set_seed(666)

options_phase0 = {
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'activation_type': [ReLU, ELU, LeakyReLU],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D]
    }

options = {
    'include_layer': [True, False],
    'a_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'a_include_BN': [True, False],
    'a_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'activation_type': [ReLU, ELU, LeakyReLU],
    'b_filter_size': [(1,1), (3,3), (5,5), (7,7), (9,9)],
    'b_include_BN': [True, False],
    'b_output_channels': [8, 16, 32, 64, 128, 256, 512],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D]
    }


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32')
    X_train = X_train/255.

    X_test = X_test.astype('float32')
    X_test = X_test/255.

    y_train = tf.reshape(tf.one_hot(y_train, 10), shape=(-1, 10))
    y_test = tf.reshape(tf.one_hot(y_test, 10), shape=(-1, 10))

    # Create TensorFlow dataset
    BATCH_SIZE = 256
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1024).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, test_ds

train_ds, test_ds = load_cifar10()


class _BaseOrganism:
    def __init__(self,
                 chromosome={},
                 phase=0,
                 prevBestOrganism=None):
        '''
        chromosome is a dictionary of genes
        phase is the phase that the individual belongs to
        prevBestOrganism is the best organism of the previous phase
        '''
        self.phase = phase
        self.chromosome = chromosome
        self.prevBestOrganism = prevBestOrganism
        self.model = None
        if phase != 0:
            # In a later stage, the model is made by
            # attaching new layers to the prev best model
            self.last_model = prevBestOrganism.model
    
    def build_model(self):
        '''
        This is the function to build the keras model
        '''
        raise NotImplementedError()

    def fitnessFunction(self,
                        train_ds,
                        test_ds,
                        generation_number):
        '''
        This function is used to calculate the
        fitness of an individual.
        '''
        wandb.init(entity="authors",
                   project="vlga",
                   group='KAGp{}'.format(self.phase),
                   job_type='g{}'.format(generation_number))
        self.model.fit(train_ds,
                       epochs=3,
                       callbacks=[WandbCallback()],
                       verbose=0)
        _, self.fitness = self.model.evaluate(test_ds,
                                              verbose=0)
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
        child = Organism(chromosome= child_chromosome, phase=self.phase, prevBestOrganism=self.prevBestOrganism)
        child.build_model()
        child.fitnessFunction(train_ds,
                              test_ds,
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
        self.fitnessFunction(train_ds,
                             test_ds,
                             generation_number=generation_number)
    
    def show(self):
        '''
        Util function to show the individual's properties.
        '''
        pprint(self.chromosome)


class Organism(_BaseOrganism):
    def __init__(self,
                 chromosome={},
                 phase=0,
                 prevBestOrganism=None):
        '''
        chromosome is a dictionary of genes
        phase is the phase that the individual belongs to
        prevBestOrganism is the best organism of the previous phase
        '''
        self.phase = phase
        self.chromosome = chromosome
        self.prevBestOrganism=prevBestOrganism
        if phase != 0:
            # In a later stage, the model is made by
            # attaching new layers to the prev best model
            self.last_model = prevBestOrganism.model
    
    def build_model(self,
                    target_size=(224,224,3)):
        '''
        This is the function to build the keras model
        '''
        keras.backend.clear_session()
        inputs = Input(shape=target_size)
        if self.phase != 0:
            # Slice the prev best model
            # Use the model as a layer
            # Attach new layer to the sliced model
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
            self.chromosome['activation_type'] = self.prevBestOrganism.chromosome['activation_type']
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
        if self.phase != 0 and self.chromosome['include_layer'] == False:
            # Except for PHASE0, there is a choice for
            # the number of layers that the model wants
            if self.chromosome['include_skip']:
                y = Conv2D(filters=self.chromosome['a_output_channels'],
                           kernel_size=(1,1),
                           padding='same')(inter_inputs)
                x = Add()([y,x])
            x = GlobalAveragePooling2D()(x)
            x = Dense(10, activation='softmax')(x)
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
            x = Dense(10, activation='softmax')(x)
        self.model = Model(inputs=[inputs], outputs=[x])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        return self.model

    def build_model_and_log(self, config):
        with wandb.init(project="artifacts-example", job_type="initialize", config=config) as run:
            config = wandb.config
            
            self.model = self.build_model(**config)

            model_artifact = wandb.Artifact(
                "convnet", type="model",
                description="Simple AlexNet style CNN",
                metadata=dict(config))

            self.model.save("initialized_model.keras")
            # ➕ another way to add a file to an Artifact
            model_artifact.add_file("initialized_model.keras")
            wandb.save("initialized_model.keras")

            run.log_artifact(model_artifact)

    def train_and_log(self, config):

        with wandb.init(project="artifacts-example", job_type="train", config=config) as run:
            config = wandb.config

            data = run.use_artifact('mnist-preprocess:latest')
            data_dir = data.download()
            training_dataset =  read(data_dir, "training")
            validation_dataset = read(data_dir, "validation")
            
            model_artifact = run.use_artifact("convnet:latest")
            model_dir = model_artifact.download()
            model_path = os.path.join(model_dir, "initialized_model.keras")
            model = keras.models.load_model(model_path)

            model_config = model_artifact.metadata

            config.update(model_config)
    
            self.train(model, training_dataset, validation_dataset, config)

            model_artifact = wandb.Artifact(
                "trained-model", type="model",
                description="NN model trained with model.fit",
                metadata=dict(config))

            model.save("trained_model.keras")
            model_artifact.add_file("trained_model.keras")
            wandb.save("trained_model.keras")

            run.log_artifact(model_artifact)

        return model

    def evaluate_and_log(self, config=None):
        
        with wandb.init(project="artifacts-example", job_type="report", config=config) as run:
            data = run.use_artifact('mnist-preprocess:latest')
            data_dir = data.download()
            test_dataset = read(data_dir, "test")

            model_artifact = run.use_artifact("trained-model:latest")
            model_dir = model_artifact.download()
            model_path = os.path.join(model_dir, "trained_model.keras")
            model = keras.models.load_model(model_path)

            loss, accuracy, highest_losses, hardest_examples, true_labels, preds = self.evaluate(model, test_dataset)

            run.summary.update({"loss": loss, "accuracy": accuracy})

            wandb.log({"high-loss-examples":
                [wandb.Image(hard_example, caption=str(pred) + "," +  str(label))
                for hard_example, pred, label in zip(hardest_examples, preds, true_labels)]})

    def train_eval_run(self):
        callback_config = {"log_weights": True,
                        "save_model": False,
                        "log_batch_frequency": 10}

        train_config = {"batch_size": 128,
                        "epochs": 5,
                        "optimizer": "adam",
                        "callback_config": callback_config}

        self.train_and_log(train_config)
        self.evaluate_and_log()


    def evaluate(self, model, test_dataset):
        """
        ## Evaluate the trained model
        """

        loss, accuracy = model.evaluate(test_dataset.x, test_dataset.y, verbose=1)
        highest_losses, hardest_examples, true_labels, predictions = self.get_hardest_k_examples(test_dataset, model)

        return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions


    def get_hardest_k_examples(self, test_dataset, model, k=32):
        class_probs = model(test_dataset.x)
        predictions = np.argmax(class_probs, axis=1)
        losses = keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
        argsort_loss =  np.argsort(losses)

        highest_k_losses = np.array(losses)[argsort_loss[-k:]]
        hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
        true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

        return highest_k_losses, hardest_k_examples, true_labels, predictions

    def train(self, model, training, validation, config):
        """
        ## Train the model
        """
        model.compile(loss="categorical_crossentropy",
                    optimizer=config.optimizer, metrics=["accuracy"])

        callback = wandb.keras.WandbCallback(
            validation_data=(validation.x[:32], validation.y[:32]),
            input_type="images", labels=[str(i) for i in range(10)],
            **config["callback_config"])

        model.fit(training.x, training.y,
                validation_data=(validation.x, validation.y),
                batch_size=config.batch_size, epochs=config.epochs,
                callbacks=[callback])    
#region
    # def fitnessFunction(self,
    #                     train_ds,
    #                     test_ds,
    #                     generation_number):
    #     '''
    #     This function is used to calculate the
    #     fitness of an individual.
    #     '''
    #     wandb.init(entity="authors",
    #                project="vlga",
    #                group='KAGp{}'.format(self.phase),
    #                job_type='g{}'.format(generation_number))
    #     self.model.fit(train_ds,
    #                    epochs=3,
    #                    callbacks=[WandbCallback()],
    #                    verbose=0)
    #     _, self.fitness = self.model.evaluate(test_ds,
    #                                           verbose=0)
    # def crossover(self,
    #               partner,
    #               generation_number):
    #     '''
    #     This function helps in making children from two
    #     parent individuals.
    #     '''
    #     child_chromosome = {}
    #     endpoint = np.random.randint(low=0, high=len(self.chromosome))
    #     for idx, key in enumerate(self.chromosome):
    #         if idx <= endpoint:
    #             child_chromosome[key] = self.chromosome[key]
    #         else:
    #             child_chromosome[key] = partner.chromosome[key]
    #     child = Organism(chromosome= child_chromosome, phase=self.phase, prevBestOrganism=self.prevBestOrganism)
    #     child.build_model()
    #     child.fitnessFunction(train_ds,
    #                           test_ds,
    #                           generation_number=generation_number)
    #     return child
    
    # def mutation(self, generation_number):
    #     '''
    #     One of the gene is to be mutated.
    #     '''
    #     index = np.random.randint(0, len(self.chromosome))
    #     key = list(self.chromosome.keys())[index]
    #     if  self.phase != 0:
    #         self.chromosome[key] = options[key][np.random.randint(len(options[key]))]
    #     else:
    #         self.chromosome[key] = options_phase0[key][np.random.randint(len(options_phase0[key]))]
    #     self.build_model()
    #     self.fitnessFunction(train_ds,
    #                          test_ds,
    #                          generation_number=generation_number)
    
    # def show(self):
    #     '''
    #     Util function to show the individual's properties.
    #     '''
    #     pprint(self.chromosome)
#endregion

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


if __name__=="__main__":

    org1 = Organism(chromosome=random_hyper(phase=0), phase=0)
    org1.build_model()
    tf.keras.utils.plot_model(org1.model, show_shapes=True)

    org2 = Organism(chromosome=random_hyper(phase=1), phase=1, last_model=org1)
    org2.build_model()
    tf.keras.utils.plot_model(org2.model, show_shapes=True)


