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
Setting network and layers config for the Xenqore API library package for Python.

Copyright (C) 2018-2019 by nepes Corp. All Rights Reserved

To use, simply 'import xenqore'
"""


from tensorflow.keras.utils import get_custom_objects
#from keras.utils import get_custom_objects


def register_keras_custom_object(cls):
    """Register custom object based on tensorflow_keras class"""
    
    get_custom_objects()[cls.__name__] = cls
    
    return cls


class NetworkConfig():
    """Definition of network's hyper parameter for xenqore(trained) model

    This class sets model_info structure
    batch_size: number of batch data 
    initial_lr: initial learning rate 
    var_decay: learning rate decay
    epocsh: number of iteration for training total train data
	"""
    user_defined_name = ''
    batch_size = 80
    initial_lr = 0.0095
    var_decay = 1e-5
    epochs = 500
    classes = 10


def layers_config(quantized_weight=True, weight_clip=True, use_bias=True):
    """Set layer's config"""
    
    kwargs = dict()

    if quantized_weight is True:
        kwargs["kernel_quantizer"] = "ste_sign"
    else:
        kwargs["kernel_quantizer"] = None
    

    if weight_clip is True:
        kwargs["kernel_constraint"] = "weight_clip"
    else:
        kwargs["kernel_constraint"] = None


    if use_bias is True:
        kwargs["use_bias"] = True
    else:
        kwargs["use_bias"] = False    

        
    return kwargs
    

def activations_config(binary_activation=True):
    """Set activation's config"""    

    if binary_activation is True:
        kwargs = "ste_sign"
    else:
        kwargs = "relu"
    
    return kwargs

