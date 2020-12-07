import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial
from typing import List, Tuple, Union, Dict, NamedTuple

from genetic_algorithm import stateful

TFDS_DATASETS = ['plant_village']

class ClassLabelEncoder(stateful.Stateful):
    def __init__(self, ds_info: tfds.core.dataset_info.DatasetInfo):
        self.info = ds_info
        self.dataset_name = ds_info.full_name
        self.num_samples = ds_info.splits['train'].num_examples
        self.num_classes = ds_info.features['label'].num_classes
        self.class_names = ds_info.features['label'].names
#         self._str2int = ds_info.features['label'].str2int
#         self._int2str = ds_info.features['label'].int2str
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
        CLASS_INDEX = self.class_names
        if preds.ndim != 2 or preds.shape[1] != self.num_classes:
            raise ValueError('`decode_predictions` expects '
                 'a batch of predictions '
                 '(i.e. a 2D array of shape (samples, 1000)). '
                 'Found array with shape: ' + str(preds.shape))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
        return results


    def str2int(self, labels: Union[List[str],Tuple[str]]):
        labels = self._valid_eager_tensor(labels)
        if not isinstance(labels, (list, tuple)):
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





###
# TODO Try to migrate below functions to a datasets/common.py or similar module



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
    

def get_parse_example_func(target_size, num_classes):
    resize = resize_repeat(target_size=tuple(target_size), training=False)
    one_hot = partial(tf.one_hot, depth=num_classes)
    def _parse_example(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = resize(x)
        y = one_hot(y)
        return x,y
    return _parse_example

def preprocess_data(data: tf.data.Dataset, target_size=None, num_classes=None, batch_size=1): #class_encoder=None):
    parse_example = get_parse_example_func(target_size=target_size, num_classes=num_classes) #class_encoder=class_encoder)
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
                         num_classes=class_encoder.num_classes)

    data['train'] = preprocess(data=data['train']) #, batch_size=config.batch_size)
    data['val'] = preprocess(data=data['val']) #, batch_size=config.batch_size)
    data['test'] = preprocess(data=data['test']) #, batch_size=config.batch_size)
    
    return data, class_encoder