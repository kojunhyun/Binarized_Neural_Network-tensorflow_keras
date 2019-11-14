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


import numpy as np

import tensorflow as tf

from xenqore.utils import register_keras_custom_object

from xenqore.math import sign

from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, Flatten


@tf.custom_gradient
def _binarize_with_identity_grad(x):
    """
    Gradient calculation of binarized weight
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/custom_gradient
    """

    def grad(dy):
        return dy

    return sign(x), grad


@tf.custom_gradient
def _binarize_with_weighted_grad(x):
    """
    Regulate Gradient calculation of binarized weight
    """

    def grad(dy):
        return (1 - tf.abs(x)) * 2 * dy

    return sign(x), grad


@register_keras_custom_object
class WeightClip(tf.keras.constraints.Constraint):
    """
    Weight Clip constraint
    Constrains the weights incident to each hidden unit
    to be between `[-clip_value, clip_value]`.


    # Arguments
    clip_value: The value to clip incoming weights.
    """

    def __init__(self, clip_value=1):
        self.clip_value = clip_value

    def __call__(self, x):
        return tf.clip_by_value(x, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}


@register_keras_custom_object
class weight_clip(WeightClip):
    """Add Aliases by using WeiightClip"""
    pass

    
@register_keras_custom_object
def ste_sign(x):
    """
    Straight-Through Estimator by using sign binarization function.
    forward : sign
    backward : Straight-Through Estimator
    if   : x >= 0 , then q(x) = 1
    else : q(x) = -1

    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    

    # Arguments
    x: Input tensor.


    # Returns
    Binarized tensor.


    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    """

    x = tf.clip_by_value(x, -1, 1)

    return _binarize_with_identity_grad(x)


def serialize(initializer):
    """Serialize object to string"""
    
    return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(name, custom_objects=None):
    """Deserialize string to object"""

    return tf.keras.utils.deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="quantization function",
    )


def get(identifier):
    """Get the config of input_quantizer or kernel_quantizer"""
    
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(str(identifier))
    if callable(identifier):
        return identifier
    raise ValueError(
        f"Could not interpret quantization function identifier: {identifier}"
    )


class QuantizedLayerBase(tf.keras.layers.Layer):
    """
    Base class for defining QuantizedDense or QuantizedConv2D

    # Arguments
    input_quantizer : Input data is transfomed to 1 or -1
    kernel_quantizer : Weight is transfomed to 1 or -1
    
    if   : x >= 0 , then q(x) = 1
    else : q(x) = -1
    
    The gradient is estimated using the Straight-Through Estimator
    (essentially the binarization is replaced by a clipped identity on the
    backward pass).
    

    # References
    - [Binarized Neural Networks: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1](http://arxiv.org/abs/1602.02830)
    
    
    
    """

    def __init__(self, *args, input_quantizer=None, kernel_quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_quantizer = get(input_quantizer)
        self.kernel_quantizer = get(kernel_quantizer)

        if kernel_quantizer and not self.kernel_constraint:
            print("Using a weight quantizer without setting `kernel_constraint` ")
            print("may result in starved weights (where the gradient is always zero).")    
                        

    @property
    def quantized_weights(self):
        if self.kernel_quantizer and self.kernel is not None:
            return [self.kernel_quantizer(self.kernel)]
        return []

    @property
    def quantized_latent_weights(self):
        if self.kernel_quantizer and self.kernel is not None:
            return [self.kernel]
        return []

    def call(self, inputs):
        if self.input_quantizer:
            inputs = self.input_quantizer(inputs)
        if self.kernel_quantizer:
            full_precision_kernel = self.kernel
            self.kernel = self.kernel_quantizer(self.kernel)

        output = super().call(inputs)
        if self.kernel_quantizer:
            # Reset the full precision kernel to make keras eager tests pass.
            # Is this a problem with our unit tests or a real bug?
            self.kernel = full_precision_kernel
        return output

    def get_config(self):
        config = {
            "input_quantizer": serialize(self.input_quantizer),
            "kernel_quantizer": serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


@register_keras_custom_object
class QuantizedDense(QuantizedLayerBase, tf.keras.layers.Dense):
    """
    Quantized-Dense layer class 
    
    If the input to the layer has a rank greater than 2, then it is flattened
    prior to the initial dot product with `kernel`.
    
    
    # Arguments
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. If you don't specify anything,
        no activation is applied (`a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    input_quantizer: Quantization function applied to the input of the layer.
    kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    # Input shape
    N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common situation
    would be a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
    N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input with
    shape `(batch_size, input_dim)`, the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


@register_keras_custom_object
class QuantizedConv2D(QuantizedLayerBase, tf.keras.layers.Conv2D):
    """
    Quantized-Conv2D layer class 

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    `input_quantizer` and `kernel_quantizer` are the element-wise quantization
    functions to use. If both quantization functions are `None` this layer is
    equivalent to `Conv2D`. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
    

    # Arguments
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Specifying any stride
        value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs. `channels_last` corresponds to
        inputs with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It defaults
        to the `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate
        to use for dilated convolution. Can be a single integer to specify the same
        value for all spatial dimensions. Currently, specifying any `dilation_rate`
        value != 1 is incompatible with specifying any stride value != 1.
    activation: Activation function to use. If you don't specify anything,
        no activation is applied (`a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    input_quantizer: Quantization function applied to the input of the layer.
    kernel_quantizer: Quantization function applied to the `kernel` weights matrix.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
    
    # Input shape
    4D tensor with shape:
    `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
    4D tensor with shape:
    `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    """
    
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


