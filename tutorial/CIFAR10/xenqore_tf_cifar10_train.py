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


train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_datagen.fit(train_x)

valid_datagen = ImageDataGenerator()

model = xenqore.apps.VGGNet13(mode=0, 
                              network_config=network_config,
                              layer_config=layers_config, 
                              act_config=activations_config, 
                              saved_model='', 
                              classes=network_config.classes,
                              input_shape=train_x.shape[1:])

model.summary()


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=network_config.initial_lr),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)


def lr_schedule(epoch, lr):
    if epoch < 100:
        return network_config.initial_lr * 0.1 ** (epoch // 50)
    else:
        if epoch % 20 == 0:
            lr = lr * 0.5
            return lr
        return lr


callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath='tf_' + network_config.user_defined_name + '_result/model_{epoch}.h5',
    # Path where to save the model
    # The two parameters below mean that we will overwrite
    # the current checkpoint if and only if
    # the `val_loss` score has improved.
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1)


tensorboard_cbk = tf.keras.callbacks.TensorBoard(
    log_dir='tf_' + network_config.user_defined_name + '_result',
    histogram_freq=1,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq='epoch')  # How often to write logs (default: once per epoch)


trained_model = model.fit_generator(
    train_datagen.flow(train_x, train_y, batch_size=network_config.batch_size),
    epochs=network_config.epochs,
    steps_per_epoch=train_y.shape[0] // network_config.batch_size,
    validation_data=valid_datagen.flow(valid_x, valid_y, batch_size=network_config.batch_size),
    validation_steps=valid_y.shape[0] // network_config.batch_size,
    verbose=1,
    callbacks=[callbacks, tensorboard_cbk, tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)]
)
