

import numpy as np
import pandas as pd
from typing import Dict, Union

def class_counts(y: np.ndarray, as_dataframe: bool=False) -> Union[Dict[Union[str,int],int],pd.DataFrame]:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    if as_dataframe:
        counts = pd.DataFrame([(k,v) for k,v in counts.items()]).rename(columns={0:'label', 1:'label_count'})
    return counts


from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
import wandb

from genetic_algorithm.datasets.plant_village import ClassLabelEncoder

def save_class_labels(class_labels: OneToOne, label_path: str):
    '''
    Save dictionary of str:int class labels as a csv file containing just the str labels in the order they're provided. Use load_class_labels() with the same filepath to load them back.
    
    '''
    
    if isinstance(class_labels, ClassLabelEncoder):
        label_path+='.json'
        class_labels.save(label_path)
    
    elif isinstance(class_labels, (dict,OneToOne)):
        data = pd.DataFrame(list(class_labels.keys()))
        label_path++'.csv'
        data.to_csv(label_path, index=None, header=False)
    else:
        raise Exception(f'unsupported label object of type {type(class_labels)}')
    
    return label_path


def load_class_labels(label_path: str):
    data = pd.read_csv(label_path, header=None, squeeze=True).values.tolist()
    loaded = OneToOne({label:i for i, label in enumerate(data)})
    return loaded    



def log_model_artifact(model, model_path, encoder, run=None, metadata=None):
    # TODO: link the logged model artifact to a logged classification report
    
    model.save(model_path)
    model_name = str(Path(model_path).name).replace('+','.')
    
    run = run or wandb
    artifact = wandb.Artifact(type='model', name=model_name)
    if os.path.isfile(model_path):
        artifact.add_file(model_path, name=model_name)
        class_label_path = os.path.join(os.path.dirname(model_path), 'labels')
    elif os.path.isdir(model_path):
        artifact.add_dir(model_path, name=model_name)
        class_label_path = os.path.join(model_path, 'labels')

#     class_label_path = os.path.join(os.path.dirname(model_path), 'labels')
    class_label_path = save_class_labels(class_labels=encoder, label_path=class_label_path)
    artifact.add_file(class_label_path, name=str(Path(class_label_path).name))
    
    run.log_artifact(artifact)
    
    
    
    
    
    
def log_csv_artifact(model, model_path, encoder, run=None, metadata=None):
    # TODO: link the logged model artifact to a logged classification report
    
    model.save(model_path)
    model_name = str(Path(model_path).name)
    
    run = run or wandb
    artifact = wandb.Artifact(type='model', name=model_name)
    if os.path.isfile(model_path):
        artifact.add_file(model_path, name=model_name)
        class_label_path = os.path.join(os.path.dirname(model_path), 'labels')
    elif os.path.isdir(model_path):
        artifact.add_dir(model_path, name=model_name)
        class_label_path = os.path.join(model_path, 'labels')

#     class_label_path = os.path.join(os.path.dirname(model_path), 'labels')
    save_class_labels(class_labels=encoder, label_path=class_label_path)
    artifact.add_file(class_label_path, name=str(Path(class_label_path).name))
    
    run.log_artifact(artifact)
    
    
    
    
# from collections import Mapping, Container
# from sys import getsizeof
 
# def deep_getsizeof(o, ids):
#     """Find the memory footprint of a Python object
 
#     This is a recursive function that drills down a Python object graph
#     like a dictionary holding nested dictionaries with lists of lists
#     and tuples and sets.
 
#     The sys.getsizeof function does a shallow size of only. It counts each
#     object inside a container as pointer only regardless of how big it
#     really is.
    
#     Example:
#         >> deep_getsizeof(prediction_results.get_state(), set())
 
#     :param o: the object
#     :param ids:
#     :return:
#     """
#     d = deep_getsizeof
#     if id(o) in ids:
#         return 0
 
#     r = getsizeof(o)
#     ids.add(id(o))
 
#     if isinstance(o, str):# or isinstance(0, unicode):
#         return r
 
#     if isinstance(o, Mapping):
#         return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
 
#     if isinstance(o, Container):
#         return r + sum(d(x, ids) for x in o)
 
#     return r