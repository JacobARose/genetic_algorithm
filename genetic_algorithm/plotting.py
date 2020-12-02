import numpy as np
import pandas as pd
import tensorflow as tf
import wandb


# train_ds = data['train'].map(lambda x,y: (resize(x),y)).shuffle(1024).cache().batch(config.batch_size).prefetch(-1)
def get_hardest_k_examples(test_data, model, k=32):


    bsz = next(iter(test_data))[0].shape[0]
    steps = len(test_data)*bsz
    test_data = test_data.unbatch()
    print(steps)
    x, y = next(iter(test_data.batch(steps).take(1)))
    
    class_probs = model.predict(x)
    predictions = np.argmax(class_probs, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(y, class_probs)
    argsort_loss =  np.argsort(losses)

    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = np.stack([x[i,...] for i in argsort_loss[-k:]])
    true_labels = np.argmax(np.stack([y[i,...] for i in argsort_loss[-k:]]), axis=1)

    return highest_k_losses, hardest_k_examples, true_labels, predictions
        
def log_high_loss_examples(test_dataset, model, k=32, log_predictions=True, run=None):
    print(f'logging k={k} hardest examples')
    losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
    
    if log_predictions:
        data_table = pd.DataFrame({'y':true_labels,'y_pred':predictions})
        table = wandb.Table(dataframe=data_table)
        wandb.log({"test_data_with_probabilities" : table}, commit=False) #wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")
        
        
        
    run = run or wandb
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, true_labels, predictions, losses)]
        })