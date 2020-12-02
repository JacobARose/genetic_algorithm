
import tensorflow as tf
import tensorflow_datasets as tfds

from tfrecord_utils.img_utils import resize_repeat
from boltons.funcutils import partial
from typing import List, Tuple, Union, Dict, NamedTuple

TFDS_DATASETS = ['plant_village']

class ClassLabelEncoder:
    def __init__(self, ds_info: tfds.core.dataset_info.DatasetInfo):
        self.info = ds_info
        self.dataset_name = ds_info.full_name
        self.num_samples = ds_info.splits['train'].num_examples
        self.num_classes = ds_info.features['label'].num_classes
        self.class_names = ds_info.features['label'].names
        self._str2int = ds_info.features['label'].str2int
        self._int2str = ds_info.features['label'].int2str
        
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
#     if CLASS_INDEX is None:
#         fpath = data_utils.get_file(
#         'imagenet_class_index.json',
#         CLASS_INDEX_PATH,
#         cache_subdir='models',
#         file_hash='c2c37ea517e94d9795004a39431a14cb')
#         with open(fpath) as f:
#             CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

        
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
    return data.map(lambda x,y: parse_example(x, y)) \
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