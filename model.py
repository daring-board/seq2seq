import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
                    self.enc_units,
                    return_sequences=True,
                    return_state=True,
                    recurrent_initializer='glorot_uniform'
                )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
                    self.dec_units,
                    return_sequences=True,
                    return_state=True,
                    recurrent_initializer='glorot_uniform'
                )
        self.fc = tf.keras.layers.Dense(vocab_size, kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01))
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(
                tf.nn.tanh(
                    self.W1(values) + self.W2(hidden_with_time_axis)
                )
            )
        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = tf.keras.layers.Dropout(0.3)(attention_weights)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, batch_sz, vocab, optimizer, loss_object, encoder, decoder):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([vocab['<start>']] * batch_sz, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions, loss_object)
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

def evaluate(inputs, vocab, index, units, maxlen, encoder, decoder):
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab['<start>']], 0)

    result = ''
    ret_nums, attention = [], []
    for t in range(maxlen):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += index[predicted_id] + ' '
        ret_nums.append(predicted_id)
        attention.append(attention_weights)

        if index[predicted_id] == '<end>':
            return result, ret_nums, attention
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, ret_nums, attention
