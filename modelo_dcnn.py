
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

class DCNN(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim=128,
                 nb_filters=50, ffn_units=512,
                 nb_classes=2, dropout_rate=0.1,
                 training=False, name='dcnn'):
        super(DCNN, self).__init__(name=name)
        # primeira camada embedding
        self.embedding = layers.Embedding(vocab_size, emb_dim)
        # o Conv1D só retorna um vetor de uma dimenção
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding='same', activation='relu') #bigramas
        self.trigram =layers.Conv1D(filters=nb_filters, kernel_size=3, padding='same', activation='relu') #trigramas
        self.fourgram=layers.Conv1D(filters=nb_filters, kernel_size=4, padding='same', activation='relu') #tetragramas
        self.pool = layers.GlobalMaxPool1D()
        self.dense_1 = layers.Dense(units=ffn_units, activation='relu')
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation='sigmoid')
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):

        x = self.embedding(inputs)

        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)

        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)

        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)

        marged = tf.concat([x_1, x_2, x_3], axis=-1)
        marged = self.dense_1(marged)
        marged = self.dropout(marged, training)
        output = self.last_dense(marged)

        return output
