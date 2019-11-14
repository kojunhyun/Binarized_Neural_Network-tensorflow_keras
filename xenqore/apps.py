# Copyright (C) 2018-2019 by nepes Corp. All Rights Reserved
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Designed Neural Networks structure for the Xenqore API library package for Python.

Copyright (C) 2018-2019 by nepes Corp. All Rights Reserved

To use, simply 'import xenqore'
"""


import xenqore

import numpy as np

import tensorflow as tf


def VGGNet7(mode=0,
            network_config=xenqore.utils.NetworkConfig(),
            layer_config=xenqore.utils.layers_config(), 
            act_config=xenqore.utils.activations_config(), 
            saved_model='', 
            classes=None,
            input_shape=None):
    """
    VGGNet7

    mode == 0 : 'new' mode of VGGNet7
    mode == 1 : 'load' mode of VGGNet7
    mode == 2 : 'transfer learning' mode of VGGNet7
    Freeze the layers except the last 4 layers
    """

    #
    # Define the layers name
    #
    CONV2D_NAME = 'Qconv2d_'
    DENSE_NAME = 'Qdense_'
    BATCH_NORM_NAME = 'BN_'
    ACT_NAME = 'STE_Sign_'
    SOFTMAX_NAME = 'Softmax_'
    MAXPOOL2D_NAME = 'MaxPool_'
    GLOBAL_AVERAGE_POOLING2D_NAME = 'G_A_Pool_'
    
    if not layer_config['kernel_quantizer'] == 'ste_sign':
        CONV2D_NAME = 'Conv2d_'
        DENSE_NAME = 'Dense_'

    if not act_config == 'ste_sign':
        ACT_NAME = 'RELU_'
    
    if mode is 0:
        model = tf.keras.models.Sequential()
        
        # Layer 0 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(64, 3, **layer_config, input_shape=input_shape, name=CONV2D_NAME+str(0)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(0)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(0)))
        
        # Layer 1 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(64, 3, **layer_config, name=CONV2D_NAME+str(1)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(1)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(1)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(1)))
        
        # Layer 2 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(128, 3, **layer_config, name=CONV2D_NAME+str(2)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(2)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(2)))
        
        # Layer 3 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(128, 3, **layer_config, name=CONV2D_NAME+str(3)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(3)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(3)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(3)))
        
        # Layer 4 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(4)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(4)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(4)))
        
        # Layer 5 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(5)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(5)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(5)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(5)))
        
        # Layer 6 : Quantized Dense
        model.add(xenqore.layers.GlobalAveragePooling2D(name=GLOBAL_AVERAGE_POOLING2D_NAME+str(6)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(6)))
        
        # Drop out
        model.add(xenqore.layers.Dropout(0.1))
        
        model.add(xenqore.layers.QuantizedDense(classes, **layer_config, name=DENSE_NAME+str(6)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(6)))
        model.add(xenqore.layers.Activation("softmax", name=SOFTMAX_NAME+str(6)))
        
        return model
        
    elif mode is 1:
        model = tf.keras.models.load_model(saved_model)

        xenqore.device.transform_model_to_device(model, user_defined_name=network_config.user_defined_name)
        
        return model
    
    elif mode is 2:
        model = tf.keras.models.load_model(saved_model)
        
        for layer in model.layers:
            layer.trainable = False   
        
        trans_model = tf.keras.models.Sequential()
        
        for layer in model.layers[:-4]:        
            trans_model.add(layer)
        
        return trans_model
    
    else:
        print('\nThis mode is not exist')
        print('Check the app mode ')
        raise NotImplementedError


def VGGNet13(mode=0, 
             network_config=xenqore.utils.NetworkConfig(),
             layer_config=xenqore.utils.layers_config(), 
             act_config=xenqore.utils.activations_config(), 
             saved_model='', 
             classes=None,
             input_shape=None):
    """
    VGGNet13

    mode == 0 : 'new' mode of VGGNet7
    mode == 1 : 'load' mode of VGGNet7
    mode == 2 : 'transfer learning' mode of VGGNet7
    Freeze the layers except the last 4 layers
    """    

    #
    # Define the layers name
    #
    CONV2D_NAME = 'Qconv2d_'
    DENSE_NAME = 'Qdense_'
    BATCH_NORM_NAME = 'BN_'
    ACT_NAME = 'STE_Sign_'
    SOFTMAX_NAME = 'Softmax_'
    MAXPOOL2D_NAME = 'MaxPool_'
    GLOBAL_AVERAGE_POOLING2D_NAME = 'G_A_Pool_'
    
    if not layer_config['kernel_quantizer'] == 'ste_sign':
        CONV2D_NAME = 'Conv2d_'
        DENSE_NAME = 'Dense_'

    if not act_config == 'ste_sign':
        ACT_NAME = 'RELU_'
    
    if mode is 0:
        model = tf.keras.models.Sequential()
        
        # Layer 0 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(128, 3, **layer_config, input_shape=input_shape, name=CONV2D_NAME+str(0)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(0)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(0)))
        
        # Layer 1 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(128, 3, **layer_config, name=CONV2D_NAME+str(1)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(1)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(1)))
        
        # Layer 2 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(128, 3, **layer_config, name=CONV2D_NAME+str(2)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(2)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(2)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(2)))
        
        # Layer 3 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(192, 3, **layer_config, name=CONV2D_NAME+str(3)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(3)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(3)))
        
        # Layer 4 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(192, 3, **layer_config, name=CONV2D_NAME+str(4)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(4)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(4)))
        
        # Layer 5 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(192, 3, **layer_config, name=CONV2D_NAME+str(5)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(5)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(5)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(5)))
        
        # Layer 6 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(6)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(6)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(6)))
        
        # Layer 7 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(7)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(7)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(7)))
        
        # Layer 8 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(8)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(8)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(8)))
        model.add(xenqore.layers.MaxPool2D(name=MAXPOOL2D_NAME+str(8)))
        
        # Layer 9 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(9)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(9)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(9)))
        
        # Layer 10 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(10)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(10)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(10)))
        
        # Layer 11 : Quantized Convolution 
        model.add(xenqore.layers.QuantizedConv2D(256, 3, **layer_config, name=CONV2D_NAME+str(11)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(11)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(11)))
        
        # Layer 12 : Quantized Dense 
        model.add(xenqore.layers.GlobalAveragePooling2D(name=GLOBAL_AVERAGE_POOLING2D_NAME+str(12)))
        model.add(xenqore.layers.Activation(act_config, name=ACT_NAME+str(12)))
        
        # Drop out
        model.add(xenqore.layers.Dropout(0.1))
        
        model.add(xenqore.layers.QuantizedDense(classes, **layer_config, name=DENSE_NAME+str(12)))
        model.add(xenqore.layers.BatchNormalization(gamma_constraint=tf.keras.constraints.NonNeg(), name=BATCH_NORM_NAME+str(12)))
        model.add(xenqore.layers.Activation("softmax", name=SOFTMAX_NAME+str(12)))
        
        return model        
        
    elif mode is 1:        
        model = tf.keras.models.load_model(saved_model)

        xenqore.device.transform_model_to_device(model, user_defined_name=network_config.user_defined_name)
        
        return model        
        
    elif mode is 2:        
        model = tf.keras.models.load_model(saved_model)
        model.summary()
        for layer in model.layers:
            layer.trainable = False
            
        trans_model = tf.keras.models.Sequential()
        
        for layer in model.layers[:-4]:        
            trans_model.add(layer)
            
        return trans_model
    
    else:
        print('\nThis mode is not exist')
        print('Check the app mode ')
        raise NotImplementedError