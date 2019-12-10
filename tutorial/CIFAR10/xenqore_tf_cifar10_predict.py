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

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import xenqore

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


### networks parameter setting
network_config = xenqore.utils.NetworkConfig()
network_config.user_defined_name = 'CIFAR10'

### layers parameter setting
### setting arguments default : quantized_weight=True, weight_clip=True, use_bias=True
layers_config = xenqore.utils.layers_config()

### activations parameter setting
### setting arguments defalt : binary_activation=True
activations_config = xenqore.utils.activations_config()


### dataset 
(train_x, train_y), (valid_x, valid_y) = tf.keras.datasets.cifar10.load_data()
train_x = train_x.astype('float')
train_y = np.squeeze(train_y)
valid_x = valid_x.astype('float')
valid_y = np.squeeze(valid_y)

print('train_x shape : ', train_x.shape)
print('train_y shape : ', train_y.shape)
print('valid_x shape : ', valid_x.shape)
print('valid_y shape : ', valid_y.shape)


valid_datagen = ImageDataGenerator()

saved_model_path = 'model_1.h5'

model = tf.keras.models.load_model(saved_model_path)

model.summary()

count = 0
test_step = int(valid_x.shape[0] / network_config.batch_size)

for i in range(test_step):

    batch_output = model.predict_on_batch(valid_x[i*network_config.batch_size:(i+1)*network_config.batch_size])
    output = np.argmax(batch_output, axis=1)
    
    target = valid_y[i*network_config.batch_size:(i+1)*network_config.batch_size]

    for j in range(network_config.batch_size):
        if output[j] == target[j]:
            count += 1
evaluate_acc = count / valid_x.shape[0]
print('evaluate_acc : ', evaluate_acc)
