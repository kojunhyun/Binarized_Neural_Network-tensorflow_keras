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

from tensorflow.keras.layers import Activation, Dropout, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, Flatten


@tf.custom_gradient
def _binarize_with_identity_grad(x):
    '''
    Gradient calculation of binarized weight
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/custom_gradient
    '''

    def grad(dy):
        return dy

    return sign(x), grad


@register_keras_custom_object
class WeightClip(tf.keras.constraints.Constraint):
    '''
    Weight Clip constraint
    Constrains the weights incident to each hidden unit
    to be between `[-clip_value, clip_value]`.


    # Arguments
    clip_value: The value to clip incoming weights.
    '''

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
    '''
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
    '''

    x = tf.clip_by_value(x, -1, 1)

    return _binarize_with_identity_grad(x)


def serialize(initializer):
    '''Serialize object to string'''
    
    return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(name, custom_objects=None):
    '''Deserialize string to object'''

    return tf.keras.utils.deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='quantization function',
    )


def get(identifier):
    '''Get the config of input_quantizer or kernel_quantizer'''
    
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(str(identifier))
    if callable(identifier):
        return identifier
    raise ValueError(
        f'Could not interpret quantization function identifier: {identifier}'
    )


class QuantizedLayerBase(tf.keras.layers.Layer):
    '''
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
    
    
    
    '''

    def __init__(self, *args, input_quantizer=None, kernel_quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_quantizer = get(input_quantizer)
        self.kernel_quantizer = get(kernel_quantizer)

        if kernel_quantizer and not self.kernel_constraint:
            print('Using a weight quantizer without setting `kernel_constraint` ')
            print('may result in starved weights (where the gradient is always zero).')    
                        
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
            'input_quantizer': serialize(self.input_quantizer),
            'kernel_quantizer': serialize(self.kernel_quantizer),
        }
        return {**super().get_config(), **config}


@register_keras_custom_object
class QuantizedDense(QuantizedLayerBase, tf.keras.layers.Dense):
    '''
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
    '''

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
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
    '''
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
    '''
    
    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
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


@register_keras_custom_object
class BatchNorm(tf.keras.layers.BatchNormalization):
    '''Base class of Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    
    # Arguments
    axis: Integer, the axis that should be normalized(typically the features axis).
        For instance, after a `Conv2D` layer with `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used. When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during training. 
        The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
        if the fused implementation cannot be used. If `None`, use the faster
        implementation if possible. If False, do not used the fused
        implementation.
    trainable: Boolean, if `True` the variables will be marked as trainable.
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
        which means batch normalization is performed across the whole batch. When
        `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        Normalization", which creates virtual sub-batches which are each
        normalized separately (with shared gamma, beta, and moving statistics).
        Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
        the input tensor and returning a pair (scale, bias) to apply to the
        normalized values (before gamma and beta), only during training. For
        example, if axis==-1,
        `adjustment = lambda shape: (tf.random.uniform(shape[-1:], 0.93, 1.07),
                                    tf.random.uniform(shape[-1:], -0.1, 0.1))`
        will scale the normalized value by up to 7% up or down, then shift the
        result by up to 0.1 (with independent scaling and bias for each feature
        but shared across all examples), and finally apply gamma and/or beta. If
        `None`, no adjustment is applied. 
        Cannot be specified if virtual_batch_size is specified.
    
    # Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        - `training=True`: The layer will normalize its inputs using the
            mean and variance of the current batch of inputs.
        - `training=False`: The layer will normalize its inputs using the
            mean and variance of its moving statistics, learned during training.
    
    # Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
    
    # Output shape:
    Same shape as input.
    References:
    - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
        {{TRAINABLE_ATTRIBUTE_NOTE}}
    '''
    
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=tf.keras.constraints.NonNeg(),        
        **kwargs
        ):
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,            
            **kwargs
        )