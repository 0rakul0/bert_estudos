import numpy as np
import tensorflow as tf
from keras import layers
import ScaledDotProductAttention as sdpa

class MultiHeadAttention(layers.Layer):
    def __init__(self, nb_proj):
        super(MultiHeadAttention, self).__init__
        self.nb_proj = nb_proj
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.nb_proj == 0
        self.d_proj = self.d_model // self.nb_proj

        self.query_lin = layers.Dense(units= self.d_model)
        self.key_lin = layers.Dense(units= self.d_model)
        self.values_lin = layers.Dense(units= self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size):
        shape = (batch_size, -1, self.nb_proj, self.d_proj)
        split_inputs = tf.reshape(inputs, shape = shape)
        return tf.transpose(split_inputs, perm=[0, 2, 1, 3])

    def call(self, queries, key, values, mask):
        batch_size = tf.shape(queries)[0]
        queries = self.query_lin(queries)
        key = self.key_lin(key)
        values = self.values_lin(values)

        queries = self.split_proj(queries, batch_size)
        key = self.split_proj(key, batch_size)
        values = self.split_proj(values, batch_size)

        attention = sdpa.scaled_dot_product_attention(queries=queries, key=key, values=values, mask=mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.d_model))

        outputs = self.final_lin(concat_attention)
        return outputs