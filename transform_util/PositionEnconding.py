import numpy as np
import tensorflow as tf
from keras import layers

class PositionsEncoding(layers.Layer):
    def __init__(self):
        super(PositionsEncoding, self).__init__

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles
    def call(self, inputs):
        seq_lenght = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_lenght)[:, np.newaxis], np.arange(d_model[np.newaxis, :], d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        pos_enconding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_enconding, tf.float32)