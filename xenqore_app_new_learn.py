
import xenqore
import numpy as np
import tensorflow as tf
import vggface2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


### networks parameter setting
network_config = xenqore.utils.NetworkConfig()
network_config.user_defined_name = 'vggface2'
network_config.classes = 150

### layers parameter setting
### setting arguments default : quantized_weight=True, weight_clip=True, use_bias=True
layers_config = xenqore.utils.layers_config()
activations_config = xenqore.utils.activations_config()
#kwargs = xenqore.utils.set_layer_config(quantized_weight=False, weight_clip=False, use_bias=True)


### dataset 
#train_data, valid_data = tf.keras.datasets.cifar100.load_data()
train_data, valid_data = vggface2.vgg_load()
print('train_x shape : ', train_data[0].shape)
print('train_y shape', train_data[1].shape)



train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

valid_datagen = ImageDataGenerator()


### VGGNet7 use 3 mode
### 0 : 'new' mode, default
### 1 : 'load' mode
### 2 : 'transfer learning' mode
model = xenqore.apps.VGGNet13(mode=0, layer_config=layers_config,
                              act_config=activations_config,
                              classes=network_config.classes)
#model = xenqore.apps.VGGNet7(mode=1)
#transfer_model = xenqore.apps.VGGNet7(mode=2, saved_weights='mymodel_423.h5')

model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(lr=network_config.initial_lr),
    loss = 'sparse_categorical_crossentropy',
    metrics=["accuracy"]
)


def lr_schedule(epoch, lr):
    if epoch < 150:
        return network_config.initial_lr * 0.1 ** (epoch // 50)
    else:
        if epoch % 5 == 0:
            lr = lr * 0.1
            return lr
        return lr


callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath='vgg13_result/mymodel_{epoch}.h5',
    # Path where to save the model
    # The two parameters below mean that we will overwrite
    # the current checkpoint if and only if
    # the `val_loss` score has improved.
    save_best_only=True,
    #save_weights_only=True,
    monitor='val_accuracy',
    #monitor='val_sparse_top_k_categorical_accuracy',
    #mode='max',  
    verbose=1)

tensorboard_cbk = tf.keras.callbacks.TensorBoard(
    log_dir='vgg13_result',
    histogram_freq=1,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq='epoch')  # How often to write logs (default: once per epoch)

"""
trained_model = model.fit(
    train_dataset,
    epochs=n_config.epochs,
    steps_per_epoch=train_data[1].shape[0] // n_config.batch_size,
    validation_data=valid_dataset,
    validation_steps=valid_data[1].shape[0] // n_config.batch_size,
    verbose=1,
    #callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
    #callbacks=[callbacks, tensorboard_cbk]
    callbacks=[callbacks, tensorboard_cbk, tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)]
)
"""
############################################################################################################



trained_model = model.fit_generator(
    train_datagen.flow(train_data[0], train_data[1], batch_size=network_config.batch_size),
    epochs=network_config.epochs,
    steps_per_epoch=train_data[1].shape[0] // network_config.batch_size,
    validation_data=valid_datagen.flow(valid_data[0], valid_data[1], batch_size=network_config.batch_size),
    validation_steps=valid_data[1].shape[0] // network_config.batch_size,
    verbose=1,
    callbacks=[callbacks, tensorboard_cbk, tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)]
)






