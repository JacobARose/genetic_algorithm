# Created 12/10/2020 by Jacob A Rose


from tqdm.notebook import trange, tqdm
from box import Box
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from pprint import pprint as pp

# tf.reset_default_graph()

def count_model_params(model, verbose=True):
    param_counts = {'trainable_params':
                          np.sum([K.count_params(w) for w in model.trainable_weights]),
                    'non_trainable_params':
                          np.sum([K.count_params(w) for w in model.non_trainable_weights])
                   }
    param_counts['total_params'] = param_counts['trainable_params'] + param_counts['non_trainable_params']
             
    if verbose:
        pp({k:f'{v:,}' for k,v in param_counts.items()})
    return param_counts




def calculate_tf_data_image_stats(dataset: tf.data.Dataset, summary=False):

    stats = Box({'min':[],
                 'max':[],
                 'mean':[],
                 'std':[],
                 'sum':[],
                 'count':[]})

    for x,y in tqdm(dataset):
        stats.min.extend([tf.reduce_min(x).numpy()])
        stats.max.extend([tf.reduce_max(x).numpy()])
        stats.mean.extend([tf.reduce_mean(x).numpy()])
        stats.std.extend([tf.math.reduce_std(x).numpy()])
        stats.sum.extend([tf.reduce_sum(x).numpy()])
        stats.count.extend([len(x)])

    if summary:
        summary_stats = Box({'min':np.min(stats.min),
                            'max':np.max(stats.max),
                            'mean':np.mean(stats.mean),
                            'std':np.std(stats.std),
                            'sum':np.sum(stats.sum),
                            'count':np.sum(stats.count)})
        return summary_stats
    return stats

######################################
# print(f'Running calculate_tf_data_image_stats() on data subsets: {list(data.keys())}\n')
# summary_stats = {subset:calculate_tf_data_image_stats(dataset=subset_data, summary=True) for subset, subset_data in data.items()}

# for k, subset_stats in summary_stats.items():
#     print(k)
#     pp(subset_stats.to_dict())


#######################################

def load_sample_image():
    image_path = tf.keras.utils.get_file('Birds_sample_image.jpg',
                                         'https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg')

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def load_sample_image_test():
    img = load_sample_image()
    assert img.shape.as_list() == [2951, 1814,3]
    print('Success')
    
# load_sample_image_test()
# img = load_sample_image()



from collections import Mapping, Container
from sys import getsizeof
 
def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
    
    Example:
        >> deep_getsizeof(prediction_results.get_state(), set())
 
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str):# or isinstance(0, unicode):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r