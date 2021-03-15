# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ResNet (50, 101, 152 + composable) version 2
# Trainable params: 31,479,336
# Paper: https://arxiv.org/pdf/1603.05027.pdf
# In this version, the BatchNormalization and ReLU activation are moved to be before the convolution in the bottleneck/projection blocks.
# In v1 and v1.5 they were after. 
# Note, this means that the ReLU that appeared after the add operation is now replaced as the ReLU proceeding the ending 1x1
# convolution in the block.

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add, Activation
from tensorflow.keras.regularizers import l2
import copy
import sys
sys.path.append('../')
from ..models_c import Composable


from tensorflow.keras.applications import resnet, resnet_v2
tf_pretrained_models = {
                          'resnet_50':resnet.ResNet50,
                          'resnet_101':resnet.ResNet101,
                          'resnet_152':resnet.ResNet152,
                          'resnet_50_v2':resnet_v2.ResNet50V2,
                          'resnet_101_v2':resnet_v2.ResNet101V2,
                          'resnet_152_v2':resnet_v2.ResNet152V2
                         }






class ResNetV2(Composable):
    """ Construct a Residual Convolution Network Network V2 """
    # Meta-parameter: list of groups: number of filters and number of blocks
    groups = { 50 : [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 6 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet50
               101: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 23 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet101
               152: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 8 },
                      { 'n_filters': 256, 'n_blocks': 36 },
                      { 'n_filters': 512, 'n_blocks': 3 } ]             # ResNet152
             }

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, 
                 num_layers: int=None,
                 input_shape=(224, 224, 3),
                 num_classes=1000,
                 include_top=True,
                 n_layers=None,
                 **hyperparameters):
        """ Construct a Residual Convolutional Neural Network V2
            n_layers    : number of layers
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regularizer : kernel regularizer
            initializer : kernel initializer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to include a bias with batchnorm
        """
        if n_layers is not None:
            print("DEPRECATION WARNING. Pass 'num_layers' instead of 'n_layers'")
            num_layers=n_layers
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # predefined
        if isinstance(num_layers, int):
            if num_layers not in [50, 101, 152]:
                raise Exception("ResNet: Invalid value for n_layers")
            groups = self.groups[num_layers]
            self.model_name = f'ResNet{num_layers}_v2'
        # user defined
        else:
            groups = num_layers
            self.model_name = 'ResNet_v2'
            
        if groups is None:
            return
        self.groups = copy.deepcopy(groups)

        # The input tensor
        inputs = Input(input_shape)
        # The stem convolutional group
        x = self.stem(inputs)
        # The learner
        outputs = self.learner(x, groups=groups)
        # The classifier 
        if include_top:
            # Add hidden dropout for training-time regularization
            outputs = self.classifier(outputs, num_classes, dropout=0.0)
        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer uses large (coarse) filter
        x = self.Conv2D(x, 64, (7, 7), strides=(2, 2), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
        """
        groups = metaparameters['groups']

        # First Residual Block Group (not strided)
        x = self.group(x, strides=(1, 1), **groups.pop(0))

        # Remaining Residual Block Groups (strided)
        for group in groups:
            x = self.group(x, **group)
        return x
    
    def group(self, x, strides=(2, 2), **metaparameters):
        """ Construct a Residual Group
            x         : input into the group
            strides   : whether the projection block is a strided convolution
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # Double the size of filters to fit the first Residual Block
        x = self.projection_block(x, strides=strides, **metaparameters)

        # Identity residual blocks
        for _ in range(n_blocks):
            x = self.identity_block(x, **metaparameters)
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 convolution block
    
        # Dimensionality reduction
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1), **metaparameters)

        # Bottleneck layer
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same", **metaparameters)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters * 4, (1, 1), strides=(1, 1), **metaparameters)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

    def projection_block(self, x, strides=(2,2), **metaparameters):
        """ Construct a Bottleneck Residual Block of Convolutions with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            strides  : whether the first convolution is strided
            n_filters: number of filters
            reg      : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # Construct the projection shortcut
        # Increase filters by 4X to match shape when added to output of block
        shortcut = self.BatchNormalization(x)
        shortcut = self.Conv2D(shortcut, 4 * n_filters, (1, 1), strides=strides, **metaparameters)

        ## Construct the 1x1, 3x3, 1x1 convolution block
    
        # Dimensionality reduction
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1,1), **metaparameters)

        # Bottleneck layer
        # Feature pooling when strides=(2, 2)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (3, 3), strides=strides, padding='same', **metaparameters)

        # Dimensionality restoration - increase the number of filters by 4X
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, 4 * n_filters, (1, 1), strides=(1, 1), **metaparameters)

        # Add the projection shortcut to the output of the residual block
        x = Add()([x, shortcut])
        return x
    
    
    @classmethod
    def from_tf_pretrained(cls, 
                           model_name: str,
                           input_shape=(224,224,3),
                           include_top=False,
                           add_classifier=True,
                           num_classes=1000,
                           pooling='max',
                           frozen_layers=None,
                           **hyperparameters):
        new = cls(input_shape=input_shape, include_top=include_top, **hyperparameters)
        
        inputs = Input(input_shape)
        _model = tf_pretrained_models[model_name](input_tensor=inputs, include_top=include_top, pooling=None, **hyperparameters)
        
        if isinstance(frozen_layers, tuple):
            assert len(frozen_layers)==2
            new.freeze(_model, frozen_layers)
        
        outputs = _model.output
        if add_classifier:
            # Add hidden dropout for training-time regularization
            outputs = new.classifier(outputs, num_classes, pooling=pooling, dropout=0.0)
            
        new._model = Model(_model.inputs, outputs)
            
        new.model_name = model_name
        return new

# Example
# resnet = ResNetV2(50)

def example():
    ''' Example for constructing/training a ResNet V2 model on CIFAR-10
    '''
    # Example of constructing a mini-ResNet
    groups = [ { 'n_filters' : 64, 'n_blocks': 1 },
               { 'n_filters': 128, 'n_blocks': 2 },
               { 'n_filters': 256, 'n_blocks': 2 }]
    resnet = ResNetV2(groups, input_shape=(32, 32, 3), n_classes=10)
    resnet.model.summary()
    resnet.cifar10()

# example()
