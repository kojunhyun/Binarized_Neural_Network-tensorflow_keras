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


import os

import json

import ctypes

import struct

import xenqore

import platform

import numpy as np

import tensorflow as tf

from xenqore.constants import *

from xenqore.datatypes import *

from collections import OrderedDict


def trans_weight_ordering(weight):
    '''The weight ordering transform tensorflow to device'''

    tf_weight = weight.numpy()
    tf_weight = ((np.sign(np.sign(tf_weight) + 1e-8) + 1 ) / 2).astype(np.int)
    
    device_weight = []        

    if len(tf_weight.shape) == 2:
        
        H = tf_weight.shape[0]
        W = tf_weight.shape[1]        
        for i in range(W):
            for j in range(H):
                device_weight.append(tf_weight[j][i])            

    elif len(tf_weight.shape) == 4:
        
        H = tf_weight.shape[0]
        W = tf_weight.shape[1]
        ic = tf_weight.shape[2]
        oc = tf_weight.shape[3]        
        for i in range(oc):
            for j in range(ic):
                for k in range(H):
                    for n in range(W):
                        device_weight.append(tf_weight[k][n][j][i])

    device_weight = np.array(device_weight)
    
    return device_weight


def save_info_to_json(model, user_defined_name, save_path):

    depth = 0
    node = OrderedDict()
    for var in model.layers:
        node_info = OrderedDict()

        if 'Qconv' in var.name or 'quantized_conv2d' in var.name:
            if depth == 0:
                node_info['op_type'] = 1
            else:
                node_info['op_type'] = 2

            # input_size
            node_info['input_size'] = var.input_shape[1]

            # input, output, channel_size
            node_channel = OrderedDict()
            node_channel['input'] = var.input_shape[-1]
            node_channel['output'] = var.output_shape[-1]
            node_info['channel_size'] = node_channel

            # file name info
            file_info = OrderedDict()
            file_info['0'] = 'weight_' + str(depth) + '.bin'
            file_info['1'] = 'bias_' + str(depth) + '.bin'
            node_info['initializer'] = file_info

            node[str(depth)] = node_info

            depth += 1

        elif 'MaxPool' in var.name or 'max_pooling2d' in var.name:
            node_info['op_type'] = 4

            # input_size
            node_info['input_size'] = var.input_shape[1]

            # input, output, channel_size
            node_channel = OrderedDict()
            node_channel['input'] = var.input_shape[-1]
            node_channel['output'] = var.output_shape[-1]
            node_info['channel_size'] = node_channel

            node[str(depth)] = node_info

            depth += 1

        elif 'G_A_Pool' in var.name or 'global_average_pooling2d' in var.name:
            node_info['op_type'] = 8

            # input_size
            node_info['input_size'] = var.input_shape[1]

            # input, output, channel_size
            node_channel = OrderedDict()
            node_channel['input'] = var.input_shape[-1]
            node_channel['output'] = var.output_shape[-1]
            node_info['channel_size'] = node_channel

            node[str(depth)] = node_info
            depth += 1

        elif 'Qdense' in var.name or 'quantized_dense' in var.name:
            node_info['op_type'] = 16

            # input_size
            node_info['input_size'] = 1

            # input, output, channel_size
            node_channel = OrderedDict()
            node_channel['input'] = var.input_shape[-1]
            node_channel['output'] = var.output_shape[-1]
            node_info['channel_size'] = node_channel

            # file name info
            file_info = OrderedDict()
            file_info['0'] = 'weight_' + str(depth) + '.bin'
            file_info['1'] = 'bias_' + str(depth) + '.bin'
            node_info['initializer'] = file_info

            node[str(depth)] = node_info

            depth += 1

    file_data = OrderedDict()
    file_data['name'] = user_defined_name
    file_data['domain'] = 'xenqore_h01x'
    file_data['version'] = '1.0'
    file_data['model'] = {'depth' : depth, 'node' : node}

    with open(os.path.join(save_path, 'network.json'), 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

    return file_data
    

def transform_model_to_device(model, user_defined_name='UserDefined', save_path='DeviceSavePath'):
    '''
    integer bias is made by using batch_normalization's beta, gamma, mean and variance.  
    Each layer's weight and integer bias based on tensorflow are transformed to use in device.
    '''
    
    # make the save dir
    save_path = user_defined_name + '_' + save_path

    try:
        if not(os.path.isdir(save_path)):
            os.makedirs(os.path.join(save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Failed to create directory!!!!')
            raise

    json_info = save_info_to_json(model, user_defined_name, save_path)

    # tensorflow model extraction

    weight =[]
    bias = []
    bn_beta = []
    bn_gamma = []
    bn_mean = []
    bn_variance = []

    for var in model.variables:
        if 'kernel' in var.name:
            weight.append(var)
        elif 'bias' in var.name:
            bias.append(var)
        elif 'beta' in var.name:
            bn_beta.append(var)
        elif 'gamma' in var.name:
            bn_gamma.append(var)
        elif 'mean' in var.name:
            bn_mean.append(var)
        elif 'variance' in var.name:
            bn_variance.append(var)

    
    variables_count = 0
    for i in range(len(json_info['model']['node'])):
        
        if 'initializer' in json_info['model']['node'][str(i)]:
            # name setting
            weight_file_name = os.path.join(save_path, json_info['model']['node'][str(i)]['initializer']['0']) 
            integer_bias_file_name = os.path.join(save_path, json_info['model']['node'][str(i)]['initializer']['1']) 

            trans_weight = trans_weight_ordering(weight[variables_count])

            # integer bias calculation
            integer_bias = bias[variables_count].numpy() \
                + ((np.sqrt(bn_variance[variables_count].numpy() + 0.001) * bn_beta[variables_count].numpy()) / (bn_gamma[variables_count].numpy()) ) \
                - bn_mean[variables_count].numpy()
            Scale_sign = np.sign(np.sign(bn_gamma[variables_count].numpy()) + 1e-8)
            integer_bias = Scale_sign * integer_bias
            integer_bias = np.floor(integer_bias)

            integer_bias = np.where(integer_bias == np.inf, 9999, integer_bias)
            integer_bias = integer_bias.astype(np.int16)

            # convert tensorflow model to device bin file
            fw_b = open(integer_bias_file_name, 'wb')
            for j in range(len(integer_bias)):
                bias_int = struct.pack('i', int(integer_bias[j]))
                fw_b.write(bias_int)
            fw_b.close()


            if json_info['model']['node'][str(i)]['op_type'] == 16:
                fw_f = open(weight_file_name, 'wb')

                for ww in trans_weight:
                    if int(ww) == 1:
                        fw_f.write(b'\x01')
                    else:
                        fw_f.write(b'\x00')

                fw_f.close()
            else:
                fw_b = open(weight_file_name, 'wb')
                output_channel = weight[variables_count].numpy().shape[-1]
                input_channel = weight[variables_count].numpy().shape[-2]

                conv_reorder_int = [[[[0 for i1 in range(256)] for i2 in range(3)] for i3 in range(3)] for i4 in range(output_channel)]
                conv_reorder_str = [[['' for i2 in range(3)] for i3 in range(3)] for i4 in range(output_channel)]

                file_idx = 0
                for of in range(output_channel):
                    for inf in range(256):
                        for y in range(3):
                            for x in range(3):
                                if(inf < input_channel):
                                    conv_reorder_int[of][y][x][inf] = int(trans_weight[file_idx])
                                    file_idx += 1
                                else:
                                    conv_reorder_int[of][y][x][inf] = 0

                for of in range(output_channel):
                    for y in range(3):
                        for x in range(3):
                            for inf in range(255,-1,-1):
                                conv_reorder_str[of][y][x] += str(conv_reorder_int[of][y][x][inf])

                            fw_b.write(struct.pack('IIIIIIII',  \
                                                    int(conv_reorder_str[of][y][x][224:256], base=2),\
                                                    int(conv_reorder_str[of][y][x][192:224], base=2),\
                                                    int(conv_reorder_str[of][y][x][160:192], base=2),\
                                                    int(conv_reorder_str[of][y][x][128:160], base=2),\
                                                    int(conv_reorder_str[of][y][x][96 :128], base=2),\
                                                    int(conv_reorder_str[of][y][x][64 :96 ], base=2),\
                                                    int(conv_reorder_str[of][y][x][32 :64 ], base=2),\
                                                    int(conv_reorder_str[of][y][x][0  :32 ], base=2)))

                fw_b.close()

            variables_count += 1


#
# Global variable for nmengine library
#
_lib = None
_lib_path = ''


def set_lib_path(lib_path):
    """Set library path directly"""
    global _lib_path
    _lib_path = lib_path


def get_devices():
    """Gets a list of the XenQore devices that is detected by driver

    And it initializes the nmengine library
    """
    global _lib

    result = SUCCESS
    devices = []
    
    if _lib == None:
        
        # print('platform',platform.architecture(), 'uname', platform.uname()[0])
        # Load appropriate library for the operating platform 
        if platform.uname()[0] == 'Windows':
            name = _lib_path + 'xenqore.dll'
            
        elif platform.uname()[0] == 'Linux':
            name = _lib_path + 'libxenqore.so'
            
        elif platform.uname()[0] == 'Darwin':
            name = _lib_path + 'libxenqore.dylib'

        else:
            name = _lib_path + 'libxenqore.so'

        #print('dll name', name)
        _lib = ctypes.cdll.LoadLibrary(name)
    

    # Up to five device information is inquired. 
    detect_count = ctypes.c_ubyte(5)
    devices_tmp = (Device * detect_count.value)()
    result = _lib.xq_get_devices(ctypes.byref(devices_tmp),ctypes.byref(detect_count))

    for i in range(detect_count.value):
        devices.append(devices_tmp[i])

    return result, devices


def connect(device):
    """Opens selected device and performs the initialization of the XenQore."""
    return _lib.xq_connect(ctypes.byref(device))


def close(device):
    """Close selected device"""
    return _lib.xq_close(ctypes.byref(device))


def set_network_mode(device, nwtype):
    """Sets the XenQore network mode.
    
    0: bNN network mode, 1: nNM network mode
    """
    return _lib.xq_set_network_mode(ctypes.byref(device), ctypes.c_ubyte(nwtype))


def bnn_load_model(device, path):
    """Load inference model from file"""
    result = SUCCESS
    
    # create byte objects from the strings
    b_path = path.encode('utf-8')
    result = _lib.xq_bnn_load_model(ctypes.c_char_p(b_path))
    
    return result


def bnn_predict(device, input, length):
    """Predict 'input' vector which categories it belongs to."""
    result = SUCCESS
    cat = ctypes.c_ushort(0)
    result = _lib.xq_bnn_predict(ctypes.byref(device), input.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), \
        ctypes.c_uint32(length), ctypes.byref(cat))
    
    return result, cat.value

