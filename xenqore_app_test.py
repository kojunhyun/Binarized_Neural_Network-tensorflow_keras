
import xenqore
import numpy as np
import tensorflow as tf
import vggface2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image


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
print(train_data[0].dtype)
print('train_y shape', train_data[1].shape)


valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data[0], valid_data[1]))
valid_dataset = valid_dataset.batch(network_config.batch_size).repeat()

print(valid_data[0][0].shape)

data = valid_data[0][0]
data = data.astype('uint8')
data_img = Image.fromarray(data, 'RGB')
data_img = data_img.resize((100,100))
data_img.save('1_data.png')
data_img.show(title=str(valid_data[1][0]))


model = xenqore.apps.VGGNet13(mode=1,
                              network_config=network_config,
                              layer_config=layers_config,
                              act_config=activations_config,
                              classes=network_config.classes,
                              saved_model='vggface2_308.h5')


model.evaluate(valid_dataset,
               steps=valid_data[1].shape[0] // network_config.batch_size,
               verbose=1)


test_x = []
test_y = []
for i in range(10):
    test_x.append(valid_data[0][i*50])
    test_y.append(valid_data[1][i*50])

test_x = np.array(test_x)
test_y = np.array(test_y)
print('test x : ', test_x.shape)
print('test y : ', test_y.shape)

result = model.predict_classes(test_x, verbose=1)
test_label = test_y.astype('int')
test_label = test_y.flatten()
print('test label : ', test_label)
print('predict label : ', result)
print(result.shape)

#test_label = valid_data[1][:100].astype('int')
#print(test_label)