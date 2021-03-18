import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
import os
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
import seaborn as sns
from pathlib import Path
from genetic_algorithm.utils.data_utils import class_counts
from typing import Union, Optional, List, Dict

import logging
logger = logging.getLogger(__name__)

try:
    plt.style.use(['science','ieee'])
except OSError:
    plt.style.use(['seaborn-colorblind'])

plt.rcParams.update({"text.usetex": False})

# train_ds = data['train'].map(lambda x,y: (resize(x),y)).shuffle(1024).cache().batch(config.batch_size).prefetch(-1)
def get_hardest_k_examples(x: np.ndarray, 
                           y: np.ndarray,
                           y_pred: np.ndarray,
                           losses: np.ndarray,
                           model: tf.keras.models.Model,
                           k: int=32) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        y_pred (np.ndarray): [description]
        losses (np.ndarray): [description]
        model (tf.keras.models.Model): [description]
        k (int, optional): [description]. Defaults to 32.

    Returns:
        highest_k_losses: np.ndarray
        hardest_k_examples: np.ndarray
        hardest_k_true_labels: np.ndarray
        hardest_k_predictions: np.ndarray
    """
#     class_probs = model.predict(x)
#     predictions = np.argmax(class_probs, axis=1)
#     losses = tf.keras.losses.categorical_crossentropy(y, class_probs)

#     argsort_loss =  np.argsort(losses)
#     highest_k_losses = np.array(losses)[argsort_loss[-k:]]
#     hardest_k_examples = np.stack([x[i,...] for i in argsort_loss[-k:]])
#     hardest_k_true_labels = np.argmax(np.stack([y[i] for i in argsort_loss[-k:]]), axis=1)
#     hardest_k_predictions = np.stack([y_pred[i] for i in argsort_loss[-k:]])

    argsort_loss =  np.argsort(losses)[::-1]
    highest_k_losses = np.array(losses)[argsort_loss[:k]]
    hardest_k_examples = np.stack([x[i,...] for i in argsort_loss[:k]])
    hardest_k_true_labels = np.argmax(np.stack([y[i] for i in argsort_loss[:k]]), axis=1)
    hardest_k_predictions = np.stack([y_pred[i] for i in argsort_loss[:k]])

    return highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions

def get_1_epoch_from_tf_data(dataset, max_rows=np.inf):
    bsz = next(iter(dataset))[0].shape[0]
    steps = len(dataset)*bsz
    steps = min([steps, max_rows])
    dataset = dataset.unbatch()
    print(steps)
    x, y = next(iter(dataset.batch(steps).take(1)))
    return x,y
    
def get_predictions(x: np.ndarray,
                    y: np.ndarray,
                    model: tf.keras.models.Model
                    ) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Input the input images x, true labels y, and compiled model with self.predict() method
    Return the logits, predicted class labels as ints, and the per sample losses

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        model (tf.data.Dataset): [description]

    Returns:
        [np.ndarray, np.ndarray, np.ndarray]: [description]
    """    
    y_prob = model.predict(x)
    y_pred = np.argmax(y_prob, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(y, y_prob)
    
    return y_prob, y_pred, losses

#     y_true = tf.cast(y, dtype=tf.int64) #, depth=y_prob.shape[1])
#     y_true = tf.one_hot(y, depth=y_prob.shape[1])

##################################################################
##################################################################
    
def log_high_loss_examples(test_dataset, model, k=32, log_predictions=True, max_rows=10000, run=None, commit=False):
    print(f'logging k={k} hardest examples')
    x, y_true = get_1_epoch_from_tf_data(test_dataset, max_rows=max_rows)
    y_prob, y_pred, losses = get_predictions(x, y_true, model)
    highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions = get_hardest_k_examples(x, y_true, y_pred, losses, model, k=k)
    
    run = run or wandb
    if log_predictions:
        max_rows = min([int(max_rows), len(y_pred)])
        print(f'logging {max_rows} true & predicted integer labels')
        y_true, y_pred = y_true[:max_rows,...], y_pred[:max_rows] #wandb rate limits allow a max of 10,000 rows
        data_table = pd.DataFrame({'y':np.argmax(y_true,axis=1),'y_pred':y_pred})
        table = wandb.Table(dataframe=data_table)
        run.log({"test_data" : table}, commit=False) #wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")
        
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, hardest_k_true_labels, hardest_k_predictions, highest_k_losses)]
        }, commit=commit)
    
##################################################################
##################################################################

# def plot_images_with_top_k_probs():
#     fig = plot.figure(figsize=(30, 30))
#     outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

#     for i in range(25):
#         inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
#         rnd_number = randint(0,len(pred_images))
#         pred_image = np.array([pred_images[rnd_number]])
#         pred_class = get_classlabel(model.predict_classes(pred_image)[0])
#         pred_prob = model.predict(x_true).reshape(6)
#         for j in range(2):
#             if (j%2) == 0:
#                 ax = plot.Subplot(fig, inner[j])
#                 ax.imshow(x_true[0])
#                 ax.set_title(y_pred)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 fig.add_subplot(ax)
#             else:
#                 ax = plot.Subplot(fig, inner[j])
#                 ax.bar([0,1,2,3,4,5],y_prob)
#                 fig.add_subplot(ax)


##################################################################
##################################################################
    
    
def plot_confusion_matrix(cm: Union[np.ndarray, ConfusionMatrix],
                          labels: Optional[List[str]] = None,
                          norm: str="LogNorm",
                          output: str=None,
                          title: str=None,
                          cmap: str=None,
                          xlabels_rotation=45,
                          ylabels_rotation=45 #'vertical'
                          ) -> None:
    
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
    if isinstance(cm, ConfusionMatrix):
        cm = cm.to_array()
    
    n = len(cm)
    if n > 20:
        size = int(n / 4.0)
    else:
        size = 5
    fig = plt.figure(figsize=(size, size), dpi=80)
    plt.clf()
    ax = fig.add_subplot(111)
    
    ax.set_aspect(1)
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
        
    fontsize = max([8,int(n/10.0)])
    x = list(range(len(cm)))
    plt.xticks(x[::2], labels[::2], rotation=xlabels_rotation, fontsize=fontsize) #, fontname='sans-serif')
    y = list(range(len(cm)))
    plt.yticks(y[::2], labels[::2], rotation=ylabels_rotation, fontsize=fontsize) #, fontname='sans-serif')
    
    if isinstance(title, str):
        ax.set_title(title, fontsize=24)#, fontname="Liberation Serif")
    
    if norm == "LogNorm":
        norm = LogNorm(vmin=max(1, np.min(cm)), vmax=np.max(cm))
    elif norm is None:
        norm = None
        
    res = plt.imshow(cm,
                     cmap=cmap,#                    interpolation=interpolation,
                     norm=norm
                     )
    width, height = cm.shape

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(res, cax=cax, extend='both')
    
    plt.tight_layout()

    if output:
        logger.info(f"Save figure at '{output}'")
        plt.savefig(output)
        
    return fig, ax
    
    
    
    
    
    
from IPython.html import widgets
import pandas as pd
import seaborn as sns

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "16pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "18pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '500px'),
                        ('font-size', '16pt')])
]

def display_classification_report(report: pd.DataFrame, display_widget=False):
    h_neg=(0, 359, 1)
    h_pos=(0, 359)
    s=(0., 99.9)
    l=(0., 99.9)
    
    if display_widget:
        @widgets.interact
#         def f(h_neg=(0, 359, 1), h_pos=(0, 359), s=(0., 99.9), l=(0., 99.9)):
        def f(h_neg=h_neg, h_pos= h_pos, s=s, l=l):
            return report.style.background_gradient(
                 cmap=sns.palettes.diverging_palette(h_neg=h_neg, h_pos=h_pos, s=s, l=l,
                                                       as_cmap=True))\
                         .set_precision(2)\
                         .set_caption('Global summary metrics')\
                         .set_table_styles(magnify())
    else:
        return report.style.set_precision(2)\
                           .set_caption('Global summary metrics')
#     background_gradient(
#                                                 cmap=sns.palettes.diverging_palette(h_neg=h_neg, h_pos=h_pos, s=s, l=l,as_cmap=True))\

    
    
    
    

    
    


# def plot_confusion_matrix(cm,normalize=True, title='Confusion matrix', annot=False, cmap="YlGnBu"):
#     if normalize == True:
#         df = pd.DataFrame(cm.normalized_matrix).T.fillna(0)
#         fmt='%'
#     else:
#         df = pd.DataFrame(cm.matrix).T.fillna(0)
#         fmt=','
        
#     num_classes = df.shape[0]
#     fig, ax = plt.subplots(1,1, figsize=(num_classes//2,num_classes//2))
#     ax = sns.heatmap(df,annot=annot,cmap=cmap, robust=True, fmt=fmt)#, ax=ax)
#     ax.set_title(title)
#     ax.set(xlabel='Predict', ylabel='Actual')
#     ax.set_xticks([])
#     ax.set_yticks([])
    
# #     wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
# #                         predictions, ground_truth, class_names})
#     return fig, ax









def plot_per_class_metrics(cm, class_populations=None):
#     acc = pd.DataFrame([{'class_label':k,'score':v} for k,v in cm.ACC.items()])
#     f1 = pd.DataFrame([{'class_label':k,'score':v} for k,v in cm.F1.items()])
    tpr = pd.DataFrame([{'class_label':k,'score':v} for k,v in cm.TPR.items()]).fillna(0).replace('None',0) #fillna(0) # dropna()#
    ppv = pd.DataFrame([{'class_label':k,'score':v} for k,v in cm.PPV.items()]).fillna(0).replace('None',0) #.dropna()#fillna(0)

    num_plots=2
    if class_populations is not None:
        num_plots=3

    num_classes = len(class_populations)
        
    fig, axes = plt.subplots(num_plots,1,figsize=(num_classes//4+1,num_classes//4))
#     sns.barplot(x=acc['class_label'],y=acc['score'], ax=axes[0])
#     axes[0].set_title("Per-class accuracy")
#     sns.barplot(x=f1['class_label'],y=f1['score'], ax=axes[1])
#     axes[1].set_title("Per-class F1-score")
    print('PPV:\n',ppv.describe())
    print(ppv.head())
    print('TPR:\n',tpr.describe())
    print(tpr.head())
    sns.barplot(x=tpr['class_label'],y=tpr['score'], ax=axes[1]) #, cmap="YlGnBu")
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])
    axes[0].set_title("Per-class Recall (TPR)")
    sns.barplot(x=ppv['class_label'],y=ppv['score'], ax=axes[1])#, cmap="YlGnBu")
    axes[1].set_xticks([])
    axes[1].set_xticklabels([])
    axes[1].set_title("Per-class Precision (PPV)")
    
    
    for ax in axes[:2]:
        ax.set_ylim(0.0,1.0)
    if class_populations is not None:
        sns.barplot(x=class_populations['label'],y=class_populations['population'], ax=axes[2])
        axes[2].set_title("Per-class sample count")
        axes[2].set_ylim(0.0,np.max(class_populations['population']))
        axes[2].set_xticks([])
        axes[2].set_xticklabels([])

#     plt.tight_layout()
    
    return fig, ax

##################################################################
##################################################################

def log_multiclass_metrics(dataset, model, data_split_name='test', class_encoder=None, log_predictions=True, max_rows=10000, run=None, commit=False, output_path: str='.', metadata: Dict=None):
    
    x, y_true = get_1_epoch_from_tf_data(dataset, max_rows=max_rows)
    y_true = y_true.numpy()
    
    y_prob, y_pred, losses = get_predictions(x, y_true, model)
    y_true = np.argmax(y_true, axis=1)
    print(y_true.shape, y_pred.shape)
#     cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
    
    
#     import ipdb;ipdb.set_trace()
    
    if class_encoder:
        if y_true.dtype in (np.int64, int):
            y_true = class_encoder.int2str(y_true)
            y_pred = class_encoder.int2str(y_pred)
        labels = list(set(y_true))
        name = class_encoder.dataset_name
#     else:
    cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)

    if not class_encoder:
        labels = cm.classes
        name = ''
        
        
    title='_'.join([data_split_name,name])
    import os
    run = run or wandb
    fig_cm, ax = plot_confusion_matrix(cm=cm,
                                       labels=labels,
                                       norm=None, #: str="LogNorm",
                                       output=os.path.join(output_path,'confusion_matrix'),
                                       title=title,
                                       cmap="YlGnBu")
#     fig_cm, ax = plot_confusion_matrix(cm,normalize=True,title=f'{data_split_name}_confusion_matrix',annot=False,cmap="YlGnBu")
#     run.log(wandb.Image({f'{data_split_name}_cm':fig}), commit=False)
#     run.log({f'{data_split_name}_cm':wandb.Image(fig)})#, step=run.step, commit=False)
    
    report_path=os.path.join(output_path,'classification_report-'+title)
#     cm.save_csv(output=report_output)
    log_classification_report_artifact(cm, report_path, run=run, metadata=metadata)
    

#     if class_encoder is not None:
# #         y_true = np.argmax(y_true, axis=1)
#         y_true = class_encoder.int2str(y_true)

#     class_populations = pd.DataFrame([{'label':k,'population':v} for k,v in class_counts(y_true).items()])
#     fig_met, ax = plot_per_class_metrics(cm, class_populations=class_populations)
#     run.log(wandb.Image({f'{data_split_name}_per_class_metrics':fig}), commit=commit)
#     run.log({f'{data_split_name}_per_class_metrics':wandb.Image(fig_met),
    run.log({f'{data_split_name}_cm':wandb.Image(fig_cm)})
    

##################################################################
##################################################################



from sklearn.metrics import classification_report
import seaborn as sns

def log_classification_report_artifact(cm, report_path, run=None, metadata=None):
    # TODO: link the logged model artifact to a logged classification report
    
    cm.save_csv(report_path)
    report_name = str(Path(report_path).name)
    
    run = run or wandb
    artifact = wandb.Artifact(type='classification_report', name=report_name, metadata=metadata)
    if os.path.isfile(report_path):
        artifact.add_file(report_path, name=report_name)
    elif os.path.isdir(report_path):
        artifact.add_dir(report_path, name=report_name)
    
    run.log_artifact(artifact)
    
    return report_path




def plot_classification_report(y_true, y_pred, target_names=None):
    report = classification_report(y_true, y_pred, target_names=None, output_dict=True)

    per_class_metrics = pd.DataFrame(report).T.iloc[:-3,:-1]
    class_support = pd.DataFrame(report).T.iloc[:-3,-1:]
    mean_metrics = pd.DataFrame(report).T.iloc[-3:,:-1]


    fig, ax = plt.subplots(1,3, figsize=(17,20))
    # sns.heatmap(pd.DataFrame(report).iloc[:-1,:].T, annot=True)
    # sns.heatmap(pd.DataFrame(report).T, annot=True, ax=ax[0])
#     cmap = sns.diverging_palette(230, 20, as_cmap=True)
    palette = sns.crayon_palette(list(sns.crayons.keys()))

    sns.set_palette(palette)

    sns.heatmap(mean_metrics, annot=True, ax=ax[0])#, cmap="Dark2")
    sns.heatmap(per_class_metrics, annot=True, ax=ax[1])#,cmap=cmap)#"YlOrBr_r")
    sns.heatmap(class_support, fmt=',', ax=ax[2])

    return fig, ax


def log_classification_report(dataset, model, data_split_name='test', class_encoder=None, run=None): #, commit=False):
    
    
    x, y_true = get_1_epoch_from_tf_data(dataset)
    y_true = y_true.numpy()
    
    _, y_pred, _ = get_predictions(x, y_true, model)
    y_true = np.argmax(y_true, axis=1)
    
    fig, ax = plot_classification_report(y_true, y_pred, target_names=None)
    
    run = run or wandb
    run.log({f'{data_split_name}_classification_report':wandb.Image(fig)})