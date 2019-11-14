
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



### VGGNet7 use 3 mode
### 0 : 'new' mode, default
### 1 : 'load' mode
### 2 : 'transfer learning' mode
#model = xenqore.apps.VGGNet13(mode=0, layer_config=layers_config,
#                              act_config=activations_config,
#                              classes=network_config.classes)
#model = xenqore.apps.VGGNet7(mode=1)
model = xenqore.apps.VGGNet13(mode=1,
                              network_config=network_config,
                              layer_config=layers_config,
                              act_config=activations_config,
                              classes=network_config.classes,
                              saved_model='mymodel_9.h5')

model.summary()

