
import tensorflow as tf
from typing import Tuple




def build_base_model(model_name="mobile_net_v2", 
                     weights="imagenet",
                     input_shape=(224,224,3),
                     frozen_layers: Tuple[int]=None,
                     freeze_batchnorm=False):

    if model_name=="mobile_net_v2":
        model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="resnet50_v2":
        model = tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="inception_resnet_v2":
        model = tf.keras.applications.InceptionResNetV2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="inception_v3":
        model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="xception":
        model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B0":
        model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B1":
        model = tf.keras.applications.EfficientNetB1(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B2":
        model = tf.keras.applications.EfficientNetB2(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B3":
        model = tf.keras.applications.EfficientNetB3(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B4":
        model = tf.keras.applications.EfficientNetB4(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B5":
        model = tf.keras.applications.EfficientNetB5(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B6":
        model = tf.keras.applications.EfficientNetB6(input_shape=input_shape, include_top=False, weights=weights)
    elif model_name=="efficient_net_B7":
        model = tf.keras.applications.EfficientNetB7(input_shape=input_shape, include_top=False, weights=weights)
        
    model.trainable = True
    if type(frozen_layers) is tuple:
        for layer in model.layers[frozen_layers[0]:frozen_layers[1]]:
            layer.trainable = False
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            if freeze_batchnorm:
                layer.trainable = False
            else:
                layer.trainable = True

    return model

def get_preprocessing_func(model_name="mobile_net_v2", 
                           weights="imagenet"):

    if model_name=="mobile_net_v2":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name=="inception_v3":
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    elif model_name=="inception_resnet_v2":
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    elif model_name=="xception":
        preprocess_input = tf.keras.applications.xception.preprocess_input
    elif model_name=="efficient_net_B0":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B1":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B2":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B3":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B4":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B5":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B6":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="efficient_net_B7":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_name=="resnet50_v2":
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    else:
#         logger.info('Passing input without preprocessing')
        print('Passing input without preprocessing')
        preprocess_input = lambda x: x

    preprocess_input(tf.zeros([4, 224, 224, 3]))

    return preprocess_input