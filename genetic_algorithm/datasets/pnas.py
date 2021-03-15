import os
os.environ['TFDS_DATA_DIR'] = '/media/data/jacob/tensorflow_datasets'
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial
from typing import List, Tuple, Union, Dict, NamedTuple

from genetic_algorithm import stateful
from genetic_algorithm.utils.data_utils import class_counts

from pyleaves.utils.WandB_artifact_utils import load_Leaves_Minus_PNAS_dataset, load_dataset_from_artifact

PALEOAI_DATASETS = ['PNAS']

class ClassLabelEncoder(stateful.Stateful):
    def __init__(self, true_labels: np.ndarray, num_classes: int=None, name: str=''):
        self.dataset_name = name
        self.class_names = class_counts(true_labels)
        
        self.num_samples = true_labels.shape[0]
        self.num_classes = num_classes or len(self.class_names)
        self._str2int = {name:num for num, name in enumerate(self.class_names)}
        self._int2str = {num:name for num, name in enumerate(self.class_names)}
        
        
    def __getstate__(self):
        return {'dataset_name':self.dataset_name,
                'num_samples':self.num_samples,
                'num_classes':self.num_classes,
                'class_names':self.class_names}.copy()

    def __setstate__(self, state):
        self.__dict__.update({'dataset_name':state['dataset_name'],
                              'num_samples':state['num_samples'],
                              'num_classes':state['num_classes'],
                              'class_names':state['class_names']})

        self._str2int = {name:num for num, name in enumerate(state['class_names'])}
        self._int2str = {num:name for num, name in enumerate(state['class_names'])}
        self.info = None
        
    def get_state(self):
        return self.__getstate__()
    
    def set_state(self, state):
        self.__setstate__(state)

    def decode_predictions(self, preds, top=5):
        """Decodes the prediction of an PlantVillage model.
        Arguments:
            preds: Numpy array encoding a batch of predictions.
            top: Integer, how many top-guesses to return. Defaults to 5.
        Returns:
            A list of lists of top class prediction tuples
            `(class_name, class_description, score)`.
            One list of tuples per sample in batch input.
        Raises:
            ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
        """
        if preds.ndim != 2 or preds.shape[1] != self.num_classes:
            raise ValueError(f'`decode_predictions` expects '
                             'a batch of predictions '
                            f'(i.e. a 2D array of shape (samples, {self.num_classes})). '
                             'Found array with shape: ' + str(preds.shape))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(self.class_names[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
        return results


    def str2int(self, labels: Union[List[str],Tuple[str]]):
        labels = self._valid_eager_tensor(labels)
        if not isinstance(labels, (list, tuple)):
            if isinstance(labels, pd.Series):
                labels = labels.values
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            else:
                assert isinstance(labels, str)
                labels = [labels]
        output = []
        keep_labels = self._str2int
        for l in labels:
            if l in keep_labels:
                output.append(keep_labels[l])
        return output
#         return [self._str2int(l) for l in labels]

    def int2str(self, labels: Union[List[int],Tuple[int]]):
        labels = self._valid_eager_tensor(labels)
        if not isinstance(labels, (list, tuple)):
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            else:
                assert isinstance(labels, (int, np.int64))
                labels = [labels]
        output = []
        keep_labels = self._int2str
        for l in labels:
            if l in keep_labels:
                output.append(keep_labels[l])
        return output
#         return [self._int2str(l) for l in labels]

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
            assert isinstance(tensor, tf.python.framework.ops.EagerTensor)
            tensor = tensor.numpy()
        except AssertionError:
            if strict:
                raise AssertionError(f'Strict EagerTensor requirement failed assertion test in ClassLabelEncoder._valid_eager_tensor method')
#         np_tensor = tensor.numpy()
        return tensor




####################################################

def load_data_from_tensor_slices(data: pd.DataFrame,
                                 cache_paths: Union[bool,str]=True,
                                 training=False,
                                 seed=None,
                                 x_col='path',
                                 y_col='label',
                                 id_col='catalog_number',
                                 dtype=None):
    import tensorflow as tf
    dtype = dtype or tf.uint8
    num_samples = data.shape[0]

    def load_img(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    index_data = tf.data.Dataset.from_tensor_slices(data[id_col].values.tolist())
    x_data = tf.data.Dataset.from_tensor_slices(data[x_col].values.tolist())
    y_data = tf.data.Dataset.from_tensor_slices(data[y_col].values.tolist())
    
    data = tf.data.Dataset.zip((index_data, x_data, y_data))
    data = data.map(lambda idx, x, y: {'index':idx,'x':x,'y':y})
    data = data.take(num_samples).cache()
    
    # TODO TEST performance and randomness of the order of shuffle and cache when shuffling full dataset each iteration, but only filepaths and not full images.
    if training:
        data = data.shuffle(num_samples,seed=seed, reshuffle_each_iteration=True)

    data = data.map(lambda example: {'index':example['index'],
                                     'x':tf.image.convert_image_dtype(load_img(example['x'])*255.0,dtype=dtype),
                                     'y':example['y']}, num_parallel_calls=-1)
    return data








def load_pnas_dataset(threshold=100,
                      validation_split=0.1,
                      seed=None,
                      y='family'):

    train_df, test_df = load_dataset_from_artifact(dataset_name='PNAS', threshold=threshold, test_size=0.5, version='latest')
    train_df, val_df  = train_test_split(train_df, test_size=validation_split, random_state=seed, shuffle=True, stratify=train_df[y])
    
    return {'train':train_df,
            'val':val_df,
            'test':test_df}

def extract_data(data: Dict[str,pd.DataFrame],
                 x='path',
                 y='family',
                 uuid='catalog_number',
                 shuffle_first=True,
                 data_cifs_repair=True,
                 seed=None):
    
    subset_keys = list(data.keys())
    
    class_encoder = ClassLabelEncoder(true_labels=data['train'][y], name='PNAS')
    
    
    extracted_data = {}
    for subset in subset_keys:
        if shuffle_first:
            data[subset] = data[subset].sample(frac=1)
            
            
        if data_cifs_repair:
            data[subset] = data[subset].assign(raw_path=data[subset].apply(lambda x: x.raw_path.replace('data_cifs_lrs','data_cifs'), axis=1),
                                               path=data[subset].apply(lambda x: x.path.replace('data_cifs_lrs','data_cifs'), axis=1))
            
            
            
        
        paths = data[subset][x]
        text_labels = data[subset][y]
        labels = class_encoder.str2int(text_labels)
        uuids = data[subset][uuid]
        
        extracted_data[subset] = pd.DataFrame.from_records([{'index':catalog_number, 'path':path, 'label':label, 'text_label':text_label} for catalog_number, path, label, text_label in zip(uuids, paths, labels, text_labels)])
        
        training = (subset=='train')
        extracted_data[subset] = load_data_from_tensor_slices(data=extracted_data[subset], training=training, seed=seed, id_col='index', x_col='path', y_col='label', dtype=tf.float32)
    
    return extracted_data, class_encoder




def load_and_extract_pnas(threshold=100,
                          validation_split=0.1,
                          seed=None,
                          x_col='path',
                          y_col='family',
                          uuid_col='catalog_number'):
    

    data = load_pnas_dataset(threshold=threshold,
                      validation_split=validation_split,
                      seed=seed,
                      y=y_col)

    data, class_encoder = extract_data(data=data,
                                     x=x_col,
                                     y=y_col,
                                     uuid=uuid_col,
                                     shuffle_first=True,
                                     seed=seed)
    
    return data, class_encoder


def get_parse_example_func(target_size, num_classes):
    resize = resize_repeat(target_size=tuple(target_size), training=False)
    one_hot = partial(tf.one_hot, depth=num_classes)
    def _parse_example(x, y):
#         x = tf.image.convert_image_dtype(x, tf.float32)
        x = resize(x)
        y = one_hot(y)
        return x,y
    return _parse_example







from pyleaves.utils.img_aug_utils import apply_cutmixup #, display_batch_augmentation, transform
# from functools import partial


# train_data_xy = train_iter.unbatch().map(lambda x,y,_: (x,y))
# val_data = val_iter.map(lambda x,y,_: (x,y))#.repeat()

# train_data = apply_cutmixup(dataset=train_data_xy, aug_batch_size=config.batch_size, num_classes=config.num_classes, target_size=config.target_size, batch_size=config.batch_size)



# _transform = partial(transform, aug_batch_size=config.batch_size, num_classes=config.num_classes, target_size=config.target_size)
# display_batch_augmentation(data_iter=train_iter, augmentation_function=_transform, label_names=label_names, aug_batch_size=config.batch_size, top_k=2, row=6, col=4)


def preprocess_data(data: tf.data.Dataset,
                    target_size=None,
                    num_classes=None,
                    batch_size=1,
                    augmix_batch_size=2,
                    augment=False, 
                    training=False,
                    buffer_size=1024,
                    remove_uuid=False): #class_encoder=None):
    parse_example = get_parse_example_func(target_size=target_size, num_classes=num_classes) #class_encoder=class_encoder)
    if remove_uuid:
        data = data.map(lambda row: parse_example(row['x'], row['y']), num_parallel_calls=-1)
    else:
        data = data.map(lambda row: (row['index'], *parse_example(row['x'], row['y'])), num_parallel_calls=-1)
    
    if training:
        data = data.shuffle(buffer_size)
    
#     if augment:
#         data = apply_cutmixup(dataset=data, aug_batch_size=augmix_batch_size, num_classes=num_classes, target_size=target_size, batch_size=batch_size)
#     else:
    data = data.batch(batch_size)
    

    return data.prefetch(-1)
        
    
        
#                     .shuffle(buffer_size) \
#                     .batch(batch_size) \
#                     .prefetch(-1)



def load_and_preprocess_data(data_config):
    """ Load a dataset, resize images, one_hot encode labels, optionally apply any augmentations. 
    Returns a dict of {split_name:tf.data.Dataset} key:value pairs
    Args:
        data_config (DictConfig):
            data_config must have the following structure:
                {
                'load':{
                        'dataset_name':'plant_village',
                        'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
                        'data_dir':'/media/data/jacob/tensorflow_datasets'
                        },
                'preprocess' = {
                                'batch_size':32,
                                'target_size':[256,256],
                                'threshold':100
                                }
                 }
    """
    
    data, class_encoder = load_and_extract_pnas(threshold=data_config.preprocess.threshold,
                              validation_split=0.1,
                              seed=data_config.seed,
                              x_col='path',
                              y_col='family',
                              uuid_col='catalog_number')
    
    augmix_batch_size = None
    if 'augment' in data_config:
        augmix_batch_size  = data_config.augment.augmix_batch_size
    
    preprocess = partial(preprocess_data,
                         batch_size=data_config.preprocess.batch_size,
                         target_size=data_config.preprocess.target_size,
                         num_classes=class_encoder.num_classes,
                         augmix_batch_size=augmix_batch_size,
                         augment=('augment' in data_config))

    data['train'] = preprocess(data=data['train'],training=True) 
    data['val'] = preprocess(data=data['val'])
    data['test'] = preprocess(data=data['test'])
    
    return data, class_encoder, preprocess





#     load_pnas_dataset(threshold=100,
#                       validation_split=0.1,
#                       seed=None,
#                       y='family')
    
#     from pyleaves.utils.pipeline_utils import flip, rotate, rgb2gray_1channel, rgb2gray_3channel, sat_bright_con, _cond_apply, load_data_from_tfrecords, smart_resize_image
#     from pyleaves.utils.img_aug_utils import resize_repeat
#     import tensorflow as tf
#     from tensorflow.keras import backend as K

# ###
# # TODO Try to migrate below functions to a datasets/common.py or similar module



# def load_tfds_dataset(dataset_name='PNAS',
#                       batch_size=None):
#     '''
#     General interface function to properly route users to the correct function for loading their queried dataset from Tensorflow Datasets (TFDS) public data.
#     '''
#     assert dataset_name in TFDS_DATASETS
    
#     print(f'Getting the PaleoAI dataset: {dataset_name}')
#     if dataset_name == 'plant_village':
#         return load_plant_village_dataset(split      =split,
#                                           data_dir   =data_dir,
#                                           batch_size =batch_size)
#     else:
#         raise Exception('Attempted to load dataset from TFDS that we have yet to build an adapter for. Consider building a minimal working prototype by using alternative datasets as a template.')
    

# def get_parse_example_func(target_size, num_classes):
#     resize = resize_repeat(target_size=tuple(target_size), training=False)
#     one_hot = partial(tf.one_hot, depth=num_classes)
#     def _parse_example(x, y):
#         x = tf.image.convert_image_dtype(x, tf.float32)
#         x = resize(x)
#         y = one_hot(y)
#         return x,y
#     return _parse_example

# def preprocess_data(data: tf.data.Dataset, target_size=None, num_classes=None, batch_size=1, buffer_size=1024): #class_encoder=None):
#     parse_example = get_parse_example_func(target_size=target_size, num_classes=num_classes) #class_encoder=class_encoder)
#     return data.map(lambda x,y: parse_example(x, y), num_parallel_calls=-1) \
#                 .shuffle(buffer_size) \
#                 .batch(batch_size) \
#                 .prefetch(-1)



# def load_and_preprocess_data(data_config):
#     """ Load a dataset, resize images, one_hot encode labels, optionally apply any augmentations. 
#     Returns a dict of {split_name:tf.data.Dataset} key:value pairs
#     Args:
#         data_config (DictConfig):
#             data_config must have the following structure:
#                 {
#                 'load':{
#                         'dataset_name':'plant_village',
#                         'split':['train[0%:60%]','train[60%:70%]','train[70%:100%]'],
#                         'data_dir':'/media/data/jacob/tensorflow_datasets'
#                         },
#                           'preprocess' = {
#                                           'batch_size':32,
#                                           'target_size':[256,256]
#                                           }
#                          }
#                     }
                
                
#                 }
#     """

#     data, builder = load_tfds_dataset(dataset_name=data_config.load.dataset_name,
#                                       split=data_config.load.split,
#                                       data_dir=data_config.load.data_dir)

#     data_info     = builder.info
#     class_encoder = ClassLabelEncoder(data_info)
#     print(class_encoder)
# #     vocab = class_encoder.class_list
#     preprocess = partial(preprocess_data,
#                          batch_size=data_config.preprocess.batch_size,
#                          target_size=data_config.preprocess.target_size,
#                          num_classes=class_encoder.num_classes)

#     data['train'] = preprocess(data=data['train']) #, batch_size=config.batch_size)
#     data['val'] = preprocess(data=data['val']) #, batch_size=config.batch_size)
#     data['test'] = preprocess(data=data['test']) #, batch_size=config.batch_size)
    
#     return data, class_encoder









# def decode_predictions(preds, top=5, model_json=""):


#     global CLASS_INDEX

#     if CLASS_INDEX is None:
#         CLASS_INDEX = json.load(open(model_json))
#     results = []
#     for pred in preds:
#         top_indices = pred.argsort()[-top:][::-1]
#         for i in top_indices:
#             each_result = []
#             each_result.append(CLASS_INDEX[str(i)])
#             each_result.append(pred[i])
#             results.append(each_result)

#     return results