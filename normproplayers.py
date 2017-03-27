# coding: utf-8

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Reshape, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from keras.layers.pooling import _Pooling2D
from keras.layers.convolutional import _Conv
from keras.layers.core import Layer
from keras import regularizers, activations, initializers, constraints
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras import backend as K


class AveragePooling2D_NormProp(_Pooling2D):
    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super().__init__(pool_size, strides, padding, data_format, **kwargs)
        self.c1 = np.float32(1 / 2.0)
        self.c2 = np.float32(0.0)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format,
                          pool_mode='avg')
        return (output - self.c2) / self.c1
    

class MaxPooling2D_NormProp(_Pooling2D):
    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super().__init__(pool_size, strides, padding, data_format, **kwargs)
        self.c1 = np.float32(np.sqrt(0.491715))
        self.c2 = np.float32(1.02938)
        #для Post-Relu
        #(1.0457222640382573, 0.67103792603969736)
        #self.c1 = np.float32(0.67103792603969736)
        #self.c2 = np.float32(1.0457222640382573)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format,
                          pool_mode='max')
        return (output - self.c2) / self.c1
    

class Dense_NormProp(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.set_c1_c2()

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight((input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.activation == 'relu':
            self.gamma = self.add_weight((self.units,),
                                         initializer=self.my_init,
                                         name='gamma')
        else: 
            self.gamma = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        kernel_normalized = K.l2_normalize(self.kernel, axis=0)
        output = K.dot(inputs, kernel_normalized)
        if self.gamma is not None:
            output *= self.gamma
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return (output - self.c2) / self.c1

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def set_c1_c2(self):
        if self.activation.__name__ == 'sigmoid':
            self.c1 = np.float32(np.sqrt(0.043379))
            self.c2 = np.float32(0.5)
            return
        if self.activation.__name__ == 'tanh':
            self.c1 = np.float32(np.sqrt(0.39429449))
            self.c2 = np.float32(0.0)
            return
        if self.activation.__name__ == 'relu':
            self.c1 = np.float32(np.sqrt(0.5 * (1 - 1 / np.pi)))
            self.c2 = np.float32(1 / np.sqrt(2 * np.pi))
            return
        self.c1 = np.float32(1.0)
        self.c2 = np.float32(0.0)
        
    def my_init(self, shape, name=None):
        var = K.variable(1/1.21*np.ones(shape), name=name)
        return var
    

class Conv2D_NormProp(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)
        #может тут другие c1 и с2?
        self.set_c1_c2()
        #self.c1 = np.float32(1.0)
        #self.c2 = np.float32(0.0)

    def call(self, inputs):
        kernel_normalized = K.l2_normalize(self.kernel, axis=(0, 1, 2))
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                kernel_normalized,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return (outputs - self.c2) / self.c1
        
    def get_config(self):
        config = super().get_config()
        config.pop('rank')
        return config
    
    def set_c1_c2(self):
        if self.activation.__name__ == 'sigmoid':
            self.c1 = np.float32(np.sqrt(0.043379))
            self.c2 = np.float32(0.5)
            return
        if self.activation.__name__ == 'tanh':
            self.c1 = np.float32(np.sqrt(0.39429449))
            self.c2 = np.float32(0.0)
            return
        if self.activation.__name__ == 'relu':
            self.c1 = np.float32(np.sqrt(0.5 * (1 - 1 / np.pi)))
            self.c2 = np.float32(1 / np.sqrt(2 * np.pi))
            return
        self.c1 = np.float32(1.0)
        self.c2 = np.float32(0.0)
        
    def my_init(self, shape, name=None):
        var = K.variable(1/1.21*np.ones(shape), name=name)
        return var