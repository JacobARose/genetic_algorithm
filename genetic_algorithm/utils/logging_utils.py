
import numpy as np
from typing import Dict

# def class_counts(y: np.ndarray) -> Dict[int,int]:
#     return dict(zip(*np.unique(y, return_counts=True)))

from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
import wandb

from genetic_algorithm.datasets.plant_village import ClassLabelEncoder
from genetic_algorithm.plotting import display_classification_report


from typing import List, Any, Dict
from genetic_algorithm import stateful


import datetime
def get_datetime_str(datetime_obj: datetime.datetime=None):
    '''Helper function to get formatted date and time as a str. Defaults to current time if none is passed.
    
    Example:
        >> get_datetime_str()
        'Thu Dec 10 04:10:31 2020'
    '''
    
    if datetime_obj is None:
        datetime_obj = datetime.datetime.utcnow()
        
    return datetime_obj.strftime('%c')

def np_onehot(y: np.ndarray, depth: int):
    '''Stand in replacement for tf.onehot().numpy()    
    '''
    return np.identity(depth)[y].astype(np.uint8)

def test_np_onehot():##y=None, depth=None):
    y = np.array([0,4,2,1,3])
    y_onehot_ground = np.array([[1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1.],
                                [0., 0., 1., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0.]])
    y_one_hot = np_onehot(y, depth=5)
    
    assert np.all(y_one_hot == y_onehot_ground)
    assert np.all(np.argmax(y_one_hot, axis=1) == y)






class PredictionResults(stateful.Stateful):
    ''' Container for validating, saving, and loading data and metadata associated with predictions made by a tensorflow image classification model
    
    
    # TODO:
        1. Implement __getitem__, __setitem__ methods to allow selecting individual examples with the [] overloaded operator,
            e.g.
            Integer slicing:
                >>prediction_results[2]
                (y_true[2], y_pred[2])
            Query sample IDs by str:
                >>prediction_results['Wing_234']
                (y_true[i], y_pred[i])
                where i == argwhere(sample ID=='Wing_234')
            
        2. Implement  __contains__ method to enable the ability to check if a particular sample is contained by searching for a unique sample ID
            e.g.
            >>'Wing_234' in prediction_results
            False
    
    '''
    
    name: str = 'predictions'
    y_prob: np.ndarray = None
    y_true: np.ndarray = None
    class_names: List[str] = None
    extra_metadata: Dict[str,Any] = None
        
    def __init__(self, y_prob, y_true, class_names=None, name='predictions', enforce_schema=True, **extra_metadata):
        self._assign_values(y_prob=y_prob, y_true=y_true, class_names=class_names, name=name, enforce_schema=enforce_schema, **extra_metadata)
        
    def _assign_values(self, y_prob, y_true, class_names=None, name='predictions', enforce_schema=True, **extra_metadata):
        self.y_prob = y_prob # (N,M)
        self.y_true = y_true # (N,M) one_hot encoded
        self.class_names = class_names #list of len == M
        self.name = name
        self.extra_metadata = extra_metadata
        if enforce_schema:
            self._enforce_schema()
        
    def _enforce_schema(self):
        assert self.y_prob is not None
        assert self.y_true is not None
        assert isinstance(self.y_prob, np.ndarray)
        assert isinstance(self.y_true, np.ndarray)
        assert self.y_prob.ndim == self.y_true.ndim == 2
        assert self.y_prob.shape == self.y_true.shape
        assert self.y_prob.shape[0] == self.num_samples
        if self.class_names:
            assert len(self.class_names) == self.num_classes
            
        if len(self.extra_metadata) > 0:
            assert isinstance(self.extra_metadata, dict)
            for key, item in self.extra_metadata.items():
                if isinstance(item, np.integer):
                    self.extra_metadata[key] = int(item)
                elif isinstance(item, np.floating):
                    self.extra_metadata[key] = float(item)
                elif isinstance(item, np.ndarray):
                    self.extra_metadata[key] = item.tolist()

    def decode_names(self, y: np.ndarray, as_array: bool=False):
        ''' int -> str data labels/predictions
        Input an array of integer labels to get their corresponding str names as either a list (default) or a np.ndarray.
        
        '''
#         y_true = self.get_y_true(one_hot=False)
        names = [self.class_names[i] for i in y]
        if as_array:
            return np.asarray(names)
        return names

    
    def get_y_pred(self, one_hot=False):
        if one_hot:
            return np_onehot(self.y_pred, depth=self.num_classes)
        return self.y_pred
                    
    def get_y_true(self, one_hot=True):
        if one_hot:
            return self.y_true
        return np.argmax(self.y_true, axis=1)
        
    @property
    def y_pred(self):
        return np.argmax(self.y_prob, axis=1)
    
    @property
    def y_true(self):
        return self._y_true
    
    @y_true.setter
    def y_true(self, y_true_new):
        ''' Constrains any new data assigned to self.y_true is formatted as ONE-HOT-ENCODED with shape (N,M)
        Keep default format of instance's y_true as one_hot encoding while in memory, convert to sparse int encoding for serialization to disk.
        
        '''
        if y_true_new.ndim == 1:
            if not issubclass(type(y_true_new[0]), np.integer):
                y_true_new = y_true_new.astype(np.uint8)
            y_true_new = tf.one_hot(y_true_new, depth=self.num_classes).numpy()
        self._y_true = y_true_new

    
    @property
    def num_classes(self):
        return self.y_prob.shape[1]

    @property
    def num_samples(self):
        return self.y_prob.shape[0]
    
    
    def get_state(self):
        y_true = self.get_y_true(one_hot=False)
        y_prob = self.y_prob
        state = {'meta':{
                         'name':self.name,
                         'num_classes':self.num_classes,
                         'num_samples':self.num_samples,
                         **{k:v for k,v in self.extra_metadata.items()}
                 },
                 'data':{
                         'class_names':self.class_names,
                         'y_true':y_true.tolist(), #Store more memory efficient representation of y_true
                         'y_prob':y_prob.tolist()
                 }}
        assert np.allclose(y_true, np.asarray(state['data']['y_true']))
        assert np.allclose(y_prob, np.asarray(state['data']['y_prob']))
        
        return state
    
    def set_state(self, state):        
        for key in ['class_names','y_true','y_prob']:
            assert key in state['data'].keys()
            
        y_prob = np.asarray(state['data']['y_prob'])
        y_true = np.asarray(state['data']['y_true'])
        class_names =  state['data']['class_names']
        name = state['meta']['name']
        extra_metadata = {k:v for k,v in state['meta'].items() if k not in ['name','num_classes','num_samples']}
        self._assign_values(y_prob=y_prob, y_true=y_true, class_names=class_names, name=name, **extra_metadata)
        
        assert self.num_classes == state['meta']['num_classes']
        assert self.num_samples == state['meta']['num_samples']
        assert self.name == state['meta']['name']
        
    def __repr__(self):
        return '\n'.join([f'{k}:\n\t{v}' for k,v in self.get_state()['meta'].items()])
        
    @classmethod
    def log_model_artifact(cls, model, model_path: str, class_encoder=None, run=None, metadata=None, name: str=None):
        '''
        Logs a
        
        '''
        metadata = metadata or {}
        name = name or ''
        print(f'Logging model artifact for Object {name} at\n{model_path}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        run = run or wandb
        log_model_artifact(model, model_path, encoder=class_encoder, run=run, metadata=metadata)

        

        
        
        
        
        
class PredictionMetrics(stateful.Stateful):
    
    name: str = 'metrics'
    _results: PredictionResults = None
    tp: np.ndarray = None
    tn: np.ndarray = None
    fp: np.ndarray = None
    fn: np.ndarray = None
    class_names: List[str] = None
    
    _metric_names = ['tp','tn','fp','fn']
    _metric_agg_funcs = ['sum','sum','sum','sum']
    _agg_func = {'sum':np.sum, 'mean':np.mean, 'std':np.std}
    

    def __init__(self, results: PredictionResults, name='metrics'):
        self.results = results
        self.class_names = results.class_names
        
    def get_values(self, agg_mode: str=None):
        '''
        if agg_mode in ['sample',0]: Return per-sample metrics
            out: np.ndarray with shape == (num_samples,)
        elif agg_mode in ['class',1]: Return local per-class metrics
            out: np.ndarray with shape == (num_classes,)
        elif agg_mode in ['macro',2]: Return global scalar metrics produced by first calculating per-class values with agg_mode 1, then performing the same aggregation over all classes
            out: np.ndarray with shape == (1,)
        elif agg_mode in ['micro',3]: Return global scalar metrics produced by first calculating per-sample values with agg_mode 0, then performing the same aggregation over all samples equally
            out: np.ndarray with shape == (1,)
        else: Return raw onehot metrics without aggregation
            out: np.ndarray with shape (num_samples, num_classes )
        
        '''
        values = {'tp':self.tp,
                  'tn':self.tn,
                  'fp':self.fp,
                  'fn':self.fn}
#         mean_values = {'recall':self.recall(),
#                        'precision':self.precision(),
#                        'accuracy':self.accuracy()}
        
        if agg_mode in ['sample',0]:
            return {k:np.sum(v, axis=1) for k,v in values.items()}
        
        elif agg_mode in ['class',1]:
            return {k:np.sum(v, axis=0) for k,v in values.items()}
            
        elif agg_mode in ['macro',2]:
            return {k:np.sum(v) for k,v in self.get_values(agg_mode='class').items()}

        elif agg_mode in ['micro',3]:
            return {k:np.sum(v) for k,v in values.items()}
        
        else:
            return values
        
    def classification_report(self, agg_funcs=['micro', 'macro'], display_widget=False):
        if agg_funcs == 'class':
            agg = agg_funcs
            classification_report = pd.DataFrame({'recall':self.recall(agg),
                                          'precision':self.precision(agg),
                                          'accuracy':self.accuracy(agg),
                                          'f1-Score':self.f1(agg)},
                                          index=self.class_names)
        else:
            classification_report = []
            for agg in agg_funcs:
                classification_report.append({'recall':self.recall(agg),
                                              'precision':self.precision(agg),
                                              'accuracy':self.accuracy(agg),
                                              'f1-Score':self.f1(agg)})
            classification_report = pd.DataFrame.from_records(classification_report)
            classification_report.index = agg_funcs
            
#         if display_widget:
        try:
            classification_report = display_classification_report(classification_report)
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            print('Failed to display HTML widget. Returning classification report dataframe')
            
        return classification_report
        
    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results: PredictionResults):
        assert isinstance(results, PredictionResults)
        self._results = results
        self.y_pred_onehot = results.get_y_pred(one_hot=True)
        self.y_true_onehot = results.get_y_true(one_hot=True)

    @property
    def tp(self): # -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 1, self.y_true_onehot == 1)
    @property
    def tn(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 0, self.y_true_onehot == 0)
    @property
    def fp(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 1, self.y_true_onehot == 0)
    @property
    def fn(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 0, self.y_true_onehot == 1)
    
    def recall(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        recall = tp / (tp + fn)
        
        if agg_mode in ['macro',2]:
            recall = np.mean(recall)
        return recall #{k:np.mean(v) for k,v in values.items()}

    def precision(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        precision = tp / (tp + fp)
        if agg_mode in ['macro',2]:
            precision = np.mean(precision)
        return precision #{k:np.mean(v) for k,v in values.items()}

    def accuracy(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        accuracy = (tp + tn) / (tp + tn + tp + fp)
        if agg_mode in ['macro',2]:
            accuracy = np.mean(accuracy)
        return accuracy #{k:np.mean(v) for k,v in values.items()}

    
    def f1(self, agg_mode: str=None):
        
        recall = self.recall(agg_mode)
        precision = self.precision(agg_mode)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    
    

    
# prediction_results = PredictionResults(y_prob,
#                                        y_true,
#                                        class_names=class_names,
#                                        name='test_predictions',
#                                        dataset_name=dataset_name,
#                                        creation_date=get_datetime_str(), 
#                                        model_name = resnet.model_name,
#                                        groups=resnet.groups)

# prediction_results.save(fname=os.path.join(save_dir,'prediction_results.json'))
# prediction_results.reload(fname=os.path.join(save_dir,'prediction_results.json'))

# import wandb
# run = wandb.init()
# prediction_results.log_json_artifact(path=os.path.join(save_dir,'prediction_results.json'), run=run) #, artifact_type: str=None)

# artifact = run.use_artifact('jrose/genetic_algorithm-Notebooks/prediction_results.json:v0', type="<class '__main__.PredictionResults'>")
# artifact_dir = artifact.download()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#         if agg_mode in ['class',1]:
#             recall = 
#             return {k:np.mean(v, axis=0) for k,v in values.items()}
#         elif agg_mode in ['macro',2]:
#             return {k:np.mean(v) for k,v in self.recall(agg_mode='class').items()}
        
    
#     def agg_by_sample(self, metric)

#     assert np.sum([tp,tn,fp,fn]) == np.prod(self.y_pred_onehot.shape)

        
        
        
#         # per-sample metrics
#         self.tp = np.sum(np.logical_and(y_pred_onehot == 1, y_true_onehot == 1), axis=1)
#         # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
#         self.tn = np.sum(np.logical_and(y_pred_onehot == 0, y_true_onehot == 0), axis=1)
#         # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
#         self.fp = np.sum(np.logical_and(y_pred_onehot == 1, y_true_onehot == 0), axis=1)
#         # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
#         self.fn = np.sum(np.logical_and(y_pred_onehot == 0, y_true_onehot == 1), axis=1)

#         assert np.sum([tp,tn,fp,fn]) == np.prod(y_pred_onehot.shape)

    @property
    def num_classes(self):
        return self.results.num_classes
        
    @property
    def positives(self):
        positives = self.tp + self.fn
        assert positives.shape[0] == positives.sum()
        return positives

    @property
    def negatives(self):
        negatives = self.tn + self.fp
        assert (negatives.shape[0]*(self.num_classes-1)) == negatives.sum()
        return negatives

        


















        
        
        
        
        
        
        
        
        
def get_hardest_k_examples(test_dataset, model, k=32):
    class_probs = model.predict(test_dataset)
    predictions = np.argmax(class_probs, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
    argsort_loss =  np.argsort(losses)

    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
    true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

    return highest_k_losses, hardest_k_examples, true_labels, predictions
        
def log_high_loss_examples(test_dataset, model, k=32, run=None):
    print(f'logging k={k} hardest examples')
    losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
    
    run = run or wandb
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, true_labels, predictions, losses)]
        })