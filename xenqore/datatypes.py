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
Data structure and types for the Neuromem API library package for Python.

Copyright (C) 2018-2019 by nepes Corp. All Rights Reserved

To use, simply 'import nmengine'
"""


import ctypes

from enum import Enum

from xenqore.constants import *


class Device(ctypes.Structure):
    """Definition of data type for device connection.
    
    Use id for device selection.
    """

    _fields_ = [('dev', ctypes.POINTER(ctypes.c_void_p)),
                ('handle', ctypes.POINTER(ctypes.c_void_p)),
                ('lock', ctypes.POINTER(ctypes.c_void_p)),
                ('type', ctypes.c_ushort),
                ('id', ctypes.c_ushort),
                ('vid', ctypes.c_ushort),
                ('pid', ctypes.c_ushort),
                ('version', ctypes.c_ushort),
                ('is_open', ctypes.c_ubyte)]   
                

class NetworkInfo(ctypes.Structure):
    """Definition of data type for device network information.

    neuron_size: number of neurons.
    neuron_memory_size: memory size of each neuron.
    version: version of device firmware.
    """

    _fields_ = [('neuron_count', ctypes.c_uint),    
                ('neuron_memory_size', ctypes.c_ushort),
                ('version', ctypes.c_ushort)] 


class NetworkStatus(ctypes.Structure):
    """Definition of data type for network status.
    
    network_used: number of used/committed neurons.
    network_type: RBF or KNN.
    context: current/global context.
    norm: current norm type.
    """
	
    _fields_ = [('network_used', ctypes.c_uint),
                ('context', ctypes.c_ushort), 
                ('network_type', ctypes.c_ubyte),
                ('norm', ctypes.c_ubyte)] 


class Context(ctypes.Structure):
    """Definition of data type for context.
    
    context: context id.
    norm: norm type.
    minif: minimum influence field.
    maxif: maximum influence field.
    """
	
    _fields_ = [('context', ctypes.c_ushort),    
                ('norm', ctypes.c_ushort),
                ('minif', ctypes.c_ushort),
                ('maxif', ctypes.c_ushort)] 


class Neuron(ctypes.Structure):
    """Definition of data type for neuron.
    
    nid: neuron id
    size: vector length used in neuron memory. (vector size)
    ncr: neuron context. (context id, norm)
    aif: active influence field. (threshold of activation function)
    minif: minimum influence field.
    model: prototype (stored weight memory/rbf center).
    """
	
    _fields_ = [('nid', ctypes.c_uint),    
                ('size', ctypes.c_ushort),
                ('ncr', ctypes.c_ushort),
                ('aif', ctypes.c_ushort),
                ('minif', ctypes.c_ushort),
                ('cat', ctypes.c_ushort),
                ('model', ctypes.c_ubyte*256)] 


class LearnReq(ctypes.Structure):
    """Definition of data type for learning request/response.
    <<input>>
    query_affected: flag for whether to retrieve affected neuron information.
        generally flagging is not required.
    category: category of input data(vector).
    vector_size: input data size.
    vector: input data.

    <<output>>
    status: network status of learning
    affected_count: number of affected neurons
    affected_neurons[affected_count]: list of affected neurons
    """
	
    _fields_ = [('status', ctypes.c_uint),
	            ('affected_neurons', Neuron*10),
	            ('affected_count', ctypes.c_ushort),
	            ('category', ctypes.c_ushort),
	            ('size', ctypes.c_ushort),
	            ('vector', ctypes.c_ubyte*256),
	            ('query_affected', ctypes.c_ubyte)] 


class LearnBatchReq(ctypes.Structure):
    """Definition of data type for batch learning request/response.

    <<input>>
    iterable: flag for whether to iterate batch learning.
    iter_count: number of iteration (epoch).
    vector_count: number of vectors.
    vector_size: input data size.
    vectors: list of input data.
    categories: list of category of input data(vector).

    <<output>>
    iter_result: the number of committed neurons from each iteration.
    """

    _fields_ = [('vector_count', ctypes.c_uint),
                ('iter_result', ctypes.POINTER(ctypes.c_uint)),
                ('iter_count', ctypes.c_ushort),
                ('iterable', ctypes.c_ushort),
                ('vector_size', ctypes.c_ushort),
                ('categories', ctypes.POINTER(ctypes.c_ushort)),
                ('vectors', ctypes.POINTER(ctypes.c_ubyte))] 


def learn_batch_req_factory(max_iter, vector_count, vector_size):
    """Factory method to create LearnBatchReq instance which has dynamic arrays"""

    batch = LearnBatchReq()

    batch.iter_count = max_iter
    batch.iter_result = (ctypes.c_uint*max_iter)()
    batch.vector_count = vector_count
    batch.vector_size = vector_size
    batch.categories = (ctypes.c_ushort*vector_count)()
    batch.vectors = (ctypes.c_ubyte*(vector_count*vector_size))()

    return batch


class ClusterizeReq(ctypes.Structure):
    """Definition of data type for clusterize request/response.

    <<input>>
    initial_category: initial category id (it must be greater than 0).
    incrementof: unit of increasement of category id.
    vector_count: number of vectors.
    vector_size: input data size.
    vectors: list of input data.
    """

    _fields_ = [('vector_count', ctypes.c_uint),
            ('vector_size', ctypes.c_ushort),
            ('initial_category', ctypes.c_ushort),
            ('incrementof', ctypes.c_ushort),
            ('vectors', ctypes.POINTER(ctypes.c_uint8))]
            

def clusterize_req_factory(vector_count, vector_size):
    """Factory method to create ClusterizeReq instance which has dynamic arrays"""
    
    req = ClusterizeReq()
    
    req.vector_count = vector_count
    req.vector_size = vector_size
    req.vectors = (ctypes.c_uint8*(vector_count*vector_size))()

    return req
    

class ModelInfo(ctypes.Structure):
    """Definition of data type for knowledge(trained) model

    This class matches nm_model_info structure
    count: number of neurons used/committed
    max_context: the largest context id (1~127)
    max_category: the largest cateogry id (1~32766)
	"""

    _fields_ = [('count', ctypes.c_uint),
	            ('max_context', ctypes.c_ushort),
	            ('max_category', ctypes.c_ushort)] 


class ModelStat(ctypes.Structure):
    """Definition of data type for knowledge(trained) model analysis

    it shows distribution of neuron per category
    <<input>>
    context: target context id for analysis

    <<output>>
    count: number of neurons used/committed in given context
    histo_cat[the largest cateogry id + 1]: number of neurons per cateogory
    histo_deg[the largest cateogry id + 1]: nember of degenerated neuron per category
    """
    
    _fields_ = [('context', ctypes.c_ushort),
                ('count', ctypes.c_uint),
                ('histo_cat', ctypes.POINTER(ctypes.c_ushort)),
                ('histo_deg', ctypes.POINTER(ctypes.c_ushort))] 


def model_stat_factory(max_category):
    """Factory method to create ModelStat instance which has dynamic arrays"""
    
    stat = ModelStat()
    
    stat.count = max_category
    stat.histo_cat = (ctypes.c_ushort * (max_category + 1))()
    stat.histo_deg = (ctypes.c_ushort * (max_category + 1))()

    return stat




