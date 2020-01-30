import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding,
        })
        return config

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)
        return outputs

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'depth': self.depth,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'dense': self.dense,
        })
        return config

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="encoder_layer"):
        super(EncoderLayer, self).__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=d_model)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, padding_mask):
        x = self.mha(inputs, inputs, inputs, padding_mask)
        x = self.dropout1(x)
        x = self.layer_norm1(x + inputs)
    
        y = self.dense1(x)
        y = self.dense2(y)
        y = self.dropout2(y)
        y = self.layer_norm2(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mha': self.mha,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'layer_norm1': self.layer_norm1,
            'layer_norm2': self.layer_norm2,
            'dense1': self.dense1,
            'dense2': self.dense2,
        })
        return config

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_layer = EncoderLayer(units, d_model, num_heads, dropout)

    def call(self, inputs, padding_mask):
        x = self.emb(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pe(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.encoder_layer(x, padding_mask)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'emb': self.emb,
            'pe': self.pe,
            'dropout': self.dropout,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'encoder_layer': self.encoder_layer,
        })
        return config

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        super(DecoderLayer, self).__init__(name=name)
        self.mha1 = MultiHeadAttention(d_model, num_heads, name="MultiHeadAttention1") 
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(d_model, num_heads, name="MultiHeadAttention2")
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=d_model)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        x = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        x = self.layer_norm1(x + inputs)
        y = self.mha2(x, enc_outputs, enc_outputs, padding_mask)
        y = self.dropout1(y)
        y = self.layer_norm2(y + x)

        z = self.dense1(y)
        z = self.dense2(z)
        z = self.dropout2(z)
        z = self.layer_norm3(z + y)
        return z

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mha1': self.mha1,
            'mha2': self.mha2,
            'layer_norm1': self.layer_norm1,
            'layer_norm2': self.layer_norm2,
            'layer_norm3': self.layer_norm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'dense1': self.dense1,
            'dense2': self.dense2,
        })
        return config

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(vocab_size, d_model)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.num_layers = num_layers
        self.decoder_layer = DecoderLayer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        x = self.emb(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pe(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.decoder_layer(x, enc_outputs, look_ahead_mask, padding_mask)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'emb': self.emb,
            'pe': self.pe,
            'dropout': self.dropout,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'decoder_layer': self.decoder_layer,
        })
        return config

class TransformerEX():
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
        self.lambda1 = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')
        self.lambda2 = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')
        self.lambda3 = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')
        self.encoder = Encoder(
                vocab_size=vocab_size,
                num_layers=num_layers,
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.decoder = Decoder(
                vocab_size=vocab_size,
                num_layers=num_layers,
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.dense = tf.keras.layers.Dense(units=vocab_size, name="outputs")

    def build_model(self):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = self.lambda1(inputs)
        look_ahead_mask = self.lambda2(dec_inputs)
        dec_padding_mask = self.lambda3(inputs)

        encode = self.encoder(inputs, enc_padding_mask)
        decode = self.decoder(dec_inputs, encode, look_ahead_mask, dec_padding_mask)

        outputs = self.dense(decode)
        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs)

class Functions():
    def __init__(self, model, loss_object, maxlen, vocab):
        self.model = model
        self.loss_object = loss_object
        self.maxlen = maxlen
        self.vocab = vocab

    def loss_function(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = self.loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask    
        return tf.reduce_mean(loss_)

    def accuracy(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.maxlen - 1))
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        return accuracy

    def evaluate(self, sentence):
        output = tf.expand_dims([self.vocab['<start>']], 0)

        for i in range(self.maxlen):
            predictions = self.model([sentence, output], training=False)

            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.vocab['<end>']): break
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)


    def predict(self, sentence):
        prediction = evaluate(sentence)

        predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))
        return predicted_sentence

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class OptimizeLearningRate():
    def __init__(self, d_model):
        self.d_model = d_model
        self.warmup_epoch = 5

    def calc_lr(self, epoch):
        if epoch > self.warmup_epoch:
            lr =  1 / (np.sqrt(self.d_model) * np.sqrt(epoch))
        else:
            lr = epoch / (np.sqrt(self.d_model) * self.warmup_epoch ** 1.5)
        return lr

    def initial_lr(self):
        return 1 / np.sqrt(self.d_model)

class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size):
        self.batch = batch_size
        self.X, self.Y = X, Y
        self.length = len(X)

    def __getitem__(self, idx):
        X = np.asarray(self.X[idx*self.batch: (idx+1)*self.batch])
        Y = np.asarray(self.Y[idx*self.batch: (idx+1)*self.batch])
        return [X, Y[:, :-1]], Y[:, 1:]

    def __len__(self):
        return int(self.length / self.batch)

    def on_epoch_end(self):
        ''' 何もしない'''
        pass
