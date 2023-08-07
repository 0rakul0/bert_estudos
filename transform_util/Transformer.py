import numpy as np
import tensorflow as tf
from keras import layers
import transform_util.Encoders as e
import transform_util.MultiHeadAttention as mha

class Transformer(tf.keras.Model):
    def __init__(self,vocab_size_enc, vocab_size_dec, nb_layers, FFN_units, nb_proj, dropout_rate,  d_model, name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.encoder = e.Encoder(nb_layers, FFN_units, nb_proj, dropout_rate, vocab_size_enc, d_model)
        self.decoder = e.Encoder(nb_layers, FFN_units, nb_proj, dropout_rate, vocab_size_dec, d_model)

        self.last_linear = layers.Dense(units=vocab_size_dec, name='lin_output')

    def create_paddind_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1- tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_paddind_mask(enc_inputs)
        dec_mask_1 = tf.maximum(self.create_paddind_mask(dec_inputs), self.create_look_ahead_mask(dec_inputs))
        dec_mask_2 = self.create_paddind_mask(enc_inputs)

        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_mask_1, dec_mask_2, training)

        outputs = self.last_linear(dec_outputs)

        return outputs