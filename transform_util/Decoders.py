import numpy as np
import tensorflow as tf
from keras import layers
import transform_util.MultiHeadAttention as mha
import transform_util.PositionEnconding as pe

class DecoderLayer(layers.Layer):
    def __init__(self, FFN_units, nb_proj, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention_mask = mha.MultiHeadAttention(nb_proj=self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention = mha.MultiHeadAttention(nb_proj=self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.dense_2 = layers.Dense(units=self.d_model, activation='relu')
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)

        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_attention_mask(inputs, inputs, inputs, mask_1)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)

        attention_2 = self.multi_head_attention(attention, enc_outputs, enc_outputs, mask_2)
        attention_2 = self.dropout_2(attention_2, training=training)
        attention_2 = self.norm_2(attention_2 + inputs)

        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training=training)
        outputs = self.norm_3(outputs + attention_2)

        return outputs

class Decoder(layers.Layer):
    def __init__(self, nb_layers, FFN_units, nb_proj, dropout_rate, vocab_size, d_model, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_enconding = pe.PositionsEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.dec_layers = [DecoderLayer(FFN_units, nb_proj, dropout_rate) for _ in range(nb_layers)]

    def call(self, inputs, dec_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs += tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_enconding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs, dec_outputs, mask_1, mask_2, training)

        return outputs