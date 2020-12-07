

import numpy as np
from typing import Dict

def class_counts(y: np.ndarray) -> Dict[int,int]:
    return dict(zip(*np.unique(y, return_counts=True)))


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