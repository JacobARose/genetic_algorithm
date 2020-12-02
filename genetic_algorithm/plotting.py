import numpy as np
import pandas as pd
import tensorflow as tf
import wandb


# train_ds = data['train'].map(lambda x,y: (resize(x),y)).shuffle(1024).cache().batch(config.batch_size).prefetch(-1)
def get_hardest_k_examples(x: np.ndarray, 
                           y: np.ndarray,
                           y_pred: np.ndarray,
                           losses: np.ndarray,
                           model, 
                           k=32):

#     class_probs = model.predict(x)
#     predictions = np.argmax(class_probs, axis=1)
#     losses = tf.keras.losses.categorical_crossentropy(y, class_probs)

    argsort_loss =  np.argsort(losses)
    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = np.stack([x[i,...] for i in argsort_loss[-k:]])
    hardest_k_true_labels = np.argmax(np.stack([y[i] for i in argsort_loss[-k:]]), axis=1)
    hardest_k_predictions = np.stack([y_pred[i] for i in argsort_loss[-k:]])

    return highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions
        
def get_1_epoch_from_tf_data(dataset):
    bsz = next(iter(dataset))[0].shape[0]
    steps = len(dataset)*bsz
    dataset = dataset.unbatch()
    print(steps)
    x, y = next(iter(dataset.batch(steps).take(1)))
    return x,y
    
def get_predictions(x, y, model):
    y_prob = model.predict(x)
    y_pred = np.argmax(y_prob, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(y, y_prob)
    
    return y_prob, y_pred, losses
    
    
def log_high_loss_examples(test_dataset, model, k=32, log_predictions=True, run=None):
    print(f'logging k={k} hardest examples')
    
    x, y_true = get_1_epoch_from_tf_data(test_dataset)
    
    
    y_prob, y_pred, losses = get_predictions(x, y_true, model)
    
    highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions = get_hardest_k_examples(x, y_true, y_pred, losses, model, k=k)
    
    run = run or wandb
    if log_predictions:
        y_true, y_pred = y_true[:10000,...], y_pred[:10000] #wandb rate limits allow a max of 10,000 rows
        data_table = pd.DataFrame({'y':np.argmax(y_true,axis=1),'y_pred':y_pred})
        table = wandb.Table(dataframe=data_table)
        wandb.log({"test_data" : table}, commit=False) #wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")
        
        
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, hardest_k_true_labels, hardest_k_predictions, highest_k_losses)]
        })