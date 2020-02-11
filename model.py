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
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="encoder_layer"):
        super(EncoderLayer, self).__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        '''
        mha1 is self-attentionlayer
        '''
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        '''
        mha1 is self-attentionlayer
        '''
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
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
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
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
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

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

class Execution(tf.Module):
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size):
        self.batch = batch_size
        self.X, self.Y = X, Y
        self.length = len(X)

    def __getitem__(self, idx):
        X = np.asarray(self.X[idx*self.batch: (idx+1)*self.batch])
        Y = np.asarray(self.Y[idx*self.batch: (idx+1)*self.batch])
        return [X, Y[:, :-1]], Y[:, 1:]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, 
                                        True,
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def evaluate(self, inp_sentence, vocab, maxlen, expect=None):
        encoder_input = tf.expand_dims(inp_sentence, 0)
        decoder_input = np.asarray([vocab['<start>']])
        output = tf.expand_dims(decoder_input, 0)
            
        for i in range(maxlen):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            predictions, attention_weights = self.transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if predicted_id == vocab['<end>']:
                return tf.squeeze(output, axis=0), attention_weights
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0), attention_weights

def generate(inp_sentence, vocab, maxlen, model):
    encoder_input = tf.expand_dims(inp_sentence, 0)
    decoder_input = np.asarray([vocab['<start>']])
    output = tf.expand_dims(decoder_input, 0)
        
    for i in range(maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        predictions, attention_weights = model(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == vocab['<end>']:
            return tf.squeeze(output, axis=0), attention_weights
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights
