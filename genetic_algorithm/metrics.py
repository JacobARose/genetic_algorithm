

import tensorflow as tf

def mean_per_class_acc(y_true, y_pred):
    score, up_opt = tf.metrics.mean_per_class_accuracy(y_true, y_pred, num_classes=4)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)       
    return score