
from faker import Faker
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from pycm import ConfusionMatrix
from typing import List, Optional, Union, Tuple

import logging
logger = logging.getLogger(__name__)



# rc('text',usetex=False)
# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.family']='sans-serif'

plt.style.use(['science','no-latex'])

# plt.style.use('Solarize_Light2')
# faker = Faker()
def shuffle_sequence(seq, seed=None):
    rng = np.random.default_rng(seed=seed)
    if isinstance(seq, (np.ndarray,list)):
        rng.shuffle(seq)

        
        
def test_shuffle_sequence(sequence_length=100):
    # TODO: Set up some kind of regression test to quantify random shuffling
    # Plot a correlation plot of sequences before and after to qualify randomness
    
    seq = np.arange(sequence_length)
    old_seq = copy.copy(seq)
    rng.shuffle(seq)
    
    logger.INFO(f'test_shuffle_sequence:\n' + \
               f'sum(old_sequence==shuffled_sequence)={sum(old_seq==seq)}')
    
    
    
# arr = np.arange(10)
# rng.shuffle(arr)


def generate_mock_names(num_names: int=1) -> List[str]:
    faker = Faker()
    return [faker.name() for _ in range(num_names)]


#####################################################################
def generate_mock_labels(num_classes: int=1, num_samples: int=1, num_correct: int=1, label_type: str='name', as_tuples=False, seed=None):
    assert num_correct <= num_samples
    assert label_type == 'name'
    random.seed(seed)
    
    classes = generate_mock_names(num_names=num_classes)
    
    y_true = [random.choice(classes) for i in range(num_samples)]
    
#     y_true = generate_mock_names(num_names=num_samples)
    
    y_correct = y_true[:num_correct]
    y_incorrect = y_true[num_correct:][::-1]
    
    y_pred = [*y_correct, *y_incorrect]
#     y_pred = [*y_true[:num_correct], *generate_mock_names(num_names=num_samples-num_correct)]
    
    if as_tuples:
        return list(zip(y_true, y_pred))
    else:
        return y_true, y_pred




def generate_mock_predictions(num_classes: int=1, num_samples: int=1, num_correct: int=1, label_type: str='name', as_tuples=False, seed=None):
    assert num_correct <= num_samples
    assert label_type == 'name'
    random.seed(seed)
    
    classes = generate_mock_names(num_names=num_classes)
    
    y_true = [random.choice(classes) for i in range(num_samples)]
    
#     y_true = generate_mock_names(num_names=num_samples)
    
    y_correct = y_true[:num_correct]
    y_incorrect = y_true[num_correct:][::-1]
    
    y_pred = [*y_correct, *y_incorrect]
#     y_pred = [*y_true[:num_correct], *generate_mock_names(num_names=num_samples-num_correct)]
    
    if as_tuples:
        return list(zip(y_true, y_pred))
    else:
        return y_true, y_pred
    
    
    
    

########################################################



def generate_mock_labels_with_predictions(num_classes: int=1, num_samples: int=1, num_correct: int=1, label_type: str='name', as_tuples=False, seed=None):
    assert num_correct <= num_samples
    assert label_type == 'name'
    random.seed(seed)
    
    classes = generate_mock_names(num_names=num_classes)
    
    y_true = [random.choice(classes) for i in range(num_samples)]
    
#     y_true = generate_mock_names(num_names=num_samples)
    
    y_correct = y_true[:num_correct]
    y_incorrect = y_true[num_correct:][::-1]
    
    y_pred = [*y_correct, *y_incorrect]
#     y_pred = [*y_true[:num_correct], *generate_mock_names(num_names=num_samples-num_correct)]
    
    if as_tuples:
        return list(zip(y_true, y_pred))
    else:
        return y_true, y_pred


def generate_mock_confusion_matrix(num_classes: int=1, num_samples: int=1, num_correct: int=1, seed=None):
    y_true, y_pred = generate_mock_labels_with_predictions(num_classes=num_classes, num_samples=num_samples, num_correct=num_correct, label_type='name', as_tuples=False, seed=seed)
    cm = ConfusionMatrix(y_true, y_pred)
    return cm









def get_color(white_to_black: float) -> Tuple[int, int, int]:
    """
    Get grayscale color.
    Parameters
    ----------
    white_to_black : float
    Returns
    -------
    color : Tuple
    Examples
    --------
    >>> get_color(0)
    (255, 255, 255)
    >>> get_color(0.5)
    (128, 128, 128)
    >>> get_color(1)
    (0, 0, 0)
    """
    if not (0 <= white_to_black <= 1):
        raise ValueError(
            f"white_to_black={white_to_black} is not in the interval [0, 1]"
        )

    index = 255 - int(255 * white_to_black)
    r, g, b = index, index, index
    return int(r), int(g), int(b)


def get_color_code(val: float, max_val: float) -> str:
    """
    Get a HTML color code which is between 0 and max_val.
    Parameters
    ----------
    val : number
    max_val : number
    Returns
    -------
    color_code : str
    Examples
    --------
    >>> get_color_code(0, 100)
    '#ffffff'
    >>> get_color_code(100, 100)
    '#000000'
    >>> get_color_code(50, 100)
    '#808080'
    """
    value = min(1.0, float(val) / max_val)
    r, g, b = get_color(value)
    return f"#{r:02x}{g:02x}{b:02x}"


# def plot_confusion_matrix(cm: Union[np.ndarray, ConfusionMatrix],
#                           labels: Optional[List[str]] = None,
#                           norm: str="LogNorm",
#                           output: str=None,
#                           title: str=None,
#                           cmap: str=None,
#                           xlabels_rotation=90,
#                           ylabels_rotation=0 #'vertical'
#                           ) -> None:
    
#     from matplotlib.colors import LogNorm
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
#     if isinstance(cm, ConfusionMatrix):
#         cm = cm.to_array()
    
#     n = len(cm)
#     if n > 20:
#         size = int(n / 4.0)
#     else:
#         size = 5
#     fig = plt.figure(figsize=(size, size), dpi=80)
#     plt.clf()
#     ax = fig.add_subplot(111)
    
#     ax.set_aspect(1)
#     if labels is None:
#         labels = [str(i) for i in range(len(cm))]
#     x = list(range(len(cm)))
#     plt.xticks(x, labels, rotation=xlabels_rotation, fontsize=12) #, fontname='sans-serif')
#     y = list(range(len(cm)))
#     plt.yticks(y, labels, rotation=ylabels_rotation, fontsize=12) #, fontname='sans-serif')
    
#     if isinstance(title, str):
#         ax.set_title(title, fontsize=24)#, fontname="Liberation Serif")
    
#     if norm == "LogNorm":
#         norm = LogNorm(vmin=max(1, np.min(cm)), vmax=np.max(cm))
#     elif norm is None:
#         norm = None
        
#     res = plt.imshow(cm,
#                      cmap=cmap,#                    interpolation=interpolation,
#                      norm=norm
#                      )
#     width, height = cm.shape

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="4%", pad=0.1)
#     plt.colorbar(res, cax=cax, extend='both')
    
#     plt.tight_layout()

#     if output:
#         logger.info(f"Save figure at '{output}'")
#         plt.savefig(output)
        
        
#     return fig, ax


        
import genetic_algorithm as ga
import os
        

def test_plot_confusion_matrix_(cm: Union[np.ndarray, ConfusionMatrix],
                                labels: List[str]=None,
                                output_dir: str=ga.__PACKAGE_RESOURCES__,
                                seed: int=None):

    labels = labels or cm.classes
    
    num_classes = len(labels)
    num_samples = len(cm.actual_vector)
    num_correct = sum(np.array(cm.actual_vector)==np.array(cm.predict_vector))
    num_incorrect = sum(np.array(cm.actual_vector)!=np.array(cm.predict_vector))
    
    fig, ax = plot_confusion_matrix(cm=cm,
                          labels=labels
                          norm=None, #"LogNorm",
                          output=os.path.join(output_dir,'test_confusion_matrix.png'),
                          title=f'Test_confusion_matrix-num_classes:{num_classes},num_samples:{num_samples},num_correct:{num_correct}',
                          xlabels_rotation=90,
                          ylabels_rotation=0)


    return fig, ax







    
    
    
        
def test_generate_and_plot_confusion_matrix(num_classes=38,
                                            num_samples=10000,
                                            num_correct=500,
                                            labels: List[str]=None,
                                            title: str=None,
                                            output_dir: str=ga.__PACKAGE_RESOURCES__,
                                            seed: int=None):
    # TODO Create dummy plant name data from World Flora Online database
    labels = labels or cm.classes
    if title is None:
        title=f'Test_confusion_matrix-num_classes:{num_classes},num_samples:{num_samples},num_correct:{num_correct}'
    
    cm = generate_mock_confusion_matrix(num_classes=num_classes, num_samples=num_samples, num_correct=num_correct, seed=seed)

    fig, ax = plot_confusion_matrix(cm=cm,
                          labels=labels,
                          norm=None, #"LogNorm",
                          output=os.path.join(output_dir,'test_confusion_matrix.png'),
                          title=title,
                          xlabels_rotation=90,
                          ylabels_rotation=0)


    return fig, ax






from paleoai_data.utils import database_utils as db_utils


def load_paleoai_table(dataset_name: str="Leaves", x_col='path', y_col='family'):

    data = db_utils.load_full_db(db_path=None, version='v0.2')
    data = data[data.dataset==dataset_name].astype({y_col:'category'})
    labels = data[y_col].cat.categories
    
#     data = data.assign(target = data[y_col].cat.categories,
#                        input_path = data[x_col])#.to_list()
    return data, labels




def test_generate_and_plot_confusion_matrix_Leaves(num_correct=2000):
    
    
    data, labels = load_paleoai_table(dataset_name: str="Leaves", x_col='path', y_col='family')
    def generate_mock_confusion_matrix(num_classes: int=1, num_samples: int=1, num_correct: int=1, seed=None):
    y_true, y_pred = generate_mock_labels_with_predictions(num_classes=num_classes, num_samples=num_samples, num_correct=num_correct, label_type='name', as_tuples=False, seed=seed)
    cm = ConfusionMatrix(y_true, y_pred)
    return cm


    test_generate_and_plot_confusion_matrix(num_classes=358,
                                            num_samples=26080,
                                            num_correct=num_correct,
                                            labels=data.family.cat
                                            output_dir=ga.__PACKAGE_RESOURCES__,
                                            title='mock Leaves confusion matrix'
                                            seed=5)

    

def test_generate_and_plot_confusion_matrix_Fossils(num_correct=2000):
    
    
    test_generate_and_plot_confusion_matrix(num_classes=358,
                                            num_samples=26080,
                                            num_correct=num_correct,
                                            labels=data.family.cat
                                            output_dir=ga.__PACKAGE_RESOURCES__,
                                            title='mock Leaves confusion matrix'
                                            seed=5)

def test_generate_and_plot_confusion_matrix_PNAS(num_correct=2000):
    
    
    test_generate_and_plot_confusion_matrix(num_classes=358,
                                            num_samples=26080,
                                            num_correct=num_correct,
                                            labels=data.family.cat
                                            output_dir=ga.__PACKAGE_RESOURCES__,
                                            title='mock Leaves confusion matrix'
                                            seed=5)

def test_generate_and_plot_confusion_matrix_plant_village(num_correct=2000):
    
    
    test_generate_and_plot_confusion_matrix(num_classes=358,
                                            num_samples=26080,
                                            num_correct=num_correct,
                                            labels=data.family.cat
                                            output_dir=ga.__PACKAGE_RESOURCES__,
                                            title='mock Leaves confusion matrix'
                                            seed=5)

    
    
    
    test_generate_and_plot_confusion_matrix_plant_village(num_classes=,
                                                          num_samples=31713,
                                                          num_correct=1500,
                                                          output_dir=ga.__PACKAGE_RESOURCES__,
                                                          seed=5)





















        
        
        
        
        
        
        
if __name__=='__main__':
    
    # Mock Leaves dataset
    from paleoai_data.utils import database_utils as db_utils

    data = db_utils.load_full_db(db_path=None, version='v0.2')
    leaves = data[data.dataset=='Leaves']
    df = leaves.family.astype('category')
    leaves_labels = df.cat.categories.to_list()
    
    test_generate_and_plot_confusion_matrix_Leaves(num_classes=358,
                                                          num_samples=26080,
                                                          num_correct=2000,
                                                          labels=data.family.cat
                                                          output_dir=ga.__PACKAGE_RESOURCES__,
                                                          title='mock Leaves confusion matrix'
                                                          seed=5)
    
    test_generate_and_plot_confusion_matrix_Fossils(num_classes=,
                                                          num_samples=31713,
                                                          num_correct=1500,
                                                          output_dir=ga.__PACKAGE_RESOURCES__,
                                                          seed=5)
        
        
    test_generate_and_plot_confusion_matrix_PNAS(num_classes=,
                                                          num_samples=31713,
                                                          num_correct=1500,
                                                          output_dir=ga.__PACKAGE_RESOURCES__,
                                                          seed=5)

    test_generate_and_plot_confusion_matrix_plant_village(num_classes=,
                                                          num_samples=31713,
                                                          num_correct=1500,
                                                          output_dir=ga.__PACKAGE_RESOURCES__,
                                                          seed=5)

    
    
    
    
    cm = generate_mock_confusion_matrix(num_classes=253, num_samples=30000, num_correct=500, seed=8)
    
    cm = generate_mock_confusion_matrix(num_classes=358, num_samples=26080, num_correct=1500, seed=8)
    
    
    cm = generate_mock_confusion_matrix(num_classes=38, num_samples=100, num_correct=50)

    plot_confusion_matrix(cm=cm,
                          labels=None,
                          norm="LogNorm",
                          xlabels_rotation=0,
                          ylabels_rotation=90)