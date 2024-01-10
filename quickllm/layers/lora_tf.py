# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：lora_tf.py
    @Author  ：ys
    @Time    ：2023/12/12 15:22
    TF2实现，暂时作为学习用，无法在项目中直接运行，请留意
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Lora(Layer):

    def __init__(self, r, output_size, alpha, dropout, use_bias=False, name='lora', **kwargs):
        super().__init__(name=name, **kwargs)
        self.low = None
        self.up = None
        self.r = r
        self.output_size = output_size
        self.scaling = alpha / r
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.low = tf.keras.layers.Dense(self.r, kernel_initializer=tf.keras.initializers.he_uniform(),
                                         use_bias=self.use_bias, name='A')
        self.up = tf.keras.layers.Dense(self.output_size, kernel_initializer=tf.keras.initializers.zeros(),
                                        use_bias=self.use_bias, name='B')
        return super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.low(inputs)
        x = self.up(x)
        x = self.dropout(x, training=training)
        x = x * self.scaling

        return x