import os
from pathlib import Path
import wandb
import json
import tensorflow as tf

class Stateful(object):

    def get_state(self):
        """Returns the current state of this object.
        This method is called during `save`.
        """
        raise NotImplementedError

    def set_state(self, state):
        """Sets the current state of this object.
        This method is called during `reload`.
        # Arguments:
          state: Dict. The state to restore for this object.
        """
        raise NotImplementedError

    def save(self, fname):
        """Saves this object using `get_state`.
        # Arguments:
          fname: The file name to save to.
        """
        state = self.get_state()
        state_json = json.dumps(state)
        with tf.io.gfile.GFile(fname, 'w') as f:
            f.write(state_json)
        return str(fname)

    def reload(self, fname):
        """Reloads this object using `set_state`.
        # Arguments:
          fname: The file name to restore from.
        """
        with tf.io.gfile.GFile(fname, 'r') as f:
            state_data = f.read()
        state = json.loads(state_data)
        self.set_state(state)
        
    def __repr__(self):
        return '\n'.join([f'{k}:\n\t{v}' for k,v in self.get_state().items()])
        
    def __hash__(self):
        return hash(repr(self))
        
    def __eq__(self, other):
        return hash(self)==hash(other)
    
    def log_json_artifact(self, path: str, run=None, artifact_type: str=None):
        '''
        Saves to disk then logs to wandb a state dict as a json file artifact.
        
        '''        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'Logging json artifact for object {self.name} at\n{path}')

        run = run or wandb
        artifact_type = artifact_type or str(type(self))
        self.save(path)
        state = self.get_state()
        metadata = None
        if 'meta' in state.keys():
            metadata = state['meta']
        log_json_artifact(path, run=run, artifact_type=artifact_type, metadata=metadata)
    
    @classmethod
    def log_model_artifact(cls, model, model_path: str, class_encoder=None, run=None, metadata=None, name: str=None):
        '''
        Logs a
        
        # TODO log chromosome along with model artifact
        '''
        metadata = metadata or {}
        name = name or ''
        print(f'Logging model artifact for Object {name} at\n{model_path}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        run = run or wandb
        log_model_artifact(model, model_path, encoder=class_encoder, run=run, metadata=metadata)
    
    
    
def log_json_artifact(path, run=None, artifact_type='json_artifact', metadata=None):
    # TODO: link the logged model artifact to a logged classification report
    file_name = str(Path(path).name)

    run = run or wandb
    artifact = wandb.Artifact(type=artifact_type, name=file_name)
    if os.path.isfile(path):
        artifact.add_file(path, name=file_name)
    elif os.path.isdir(model_path):
        artifact.add_dir(path, name=file_name)
    run.log_artifact(artifact)
    
    
    

    
    
    
    
    
    
    
    
# import jsonpickle
# def test_weights_encode_decode():

#     from pyleaves.utils import set_tf_config
#     set_tf_config(num_gpus=1)
    
#     import tensorflow as tf
    
#     best_model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)

#     best_weights = best_model.get_weights()
    
#     frozen = jsonpickle.encode(best_weights)
#     thawed = jsonpickle.decode(frozen)
    
#     layer_num = 0
#     for x, x_thawed in zip(best_weights, thawed):
#         if np.all(x==x_thawed):
#             layer_num += 1
#     assert layer_num == len(best_weights)
    
    