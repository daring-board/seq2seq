import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                        np.arange(d_model)[np.newaxis, :],
                        d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)  
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
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
    def __init__(self, self_attn, ffn, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = self_attn
        self.ffn = ffn
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
    def __init__(self, self_attn, ffn, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = self_attn
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = ffn
        self.ffn2 = point_wise_feed_forward_network(d_model, dff)
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
        out1 = self.ffn1(out1)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn2(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, self_attn, ffn, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(self_attn, ffn, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)    
        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, self_attn, ffn, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(self_attn, ffn, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights

class TransformerEX(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerEX, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = point_wise_feed_forward_network(d_model, dff)
        self.ffn2 = point_wise_feed_forward_network(d_model, dff)

        self.encoder1 = Encoder(self.mha1, self.ffn1, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.encoder2 = Encoder(self.mha1, self.ffn1, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.encoder3 = Encoder(self.mha1, self.ffn1, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(self.mha2, self.ffn2, num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.concat = tf.keras.layers.Concatenate()
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output1 = self.encoder1(inp[0], training, enc_padding_mask[0])  # (batch_size, inp_seq_len, d_model)
        enc_output2 = self.encoder2(inp[1], training, enc_padding_mask[1])  # (batch_size, inp_seq_len, d_model)
        enc_output3 = self.encoder3(inp[2], training, enc_padding_mask[2])  # (batch_size, inp_seq_len, d_model)
        dec_output1, attention_weights1 = self.decoder(tar, enc_output1, training, look_ahead_mask[0], dec_padding_mask[0])
        dec_output2, attention_weights2 = self.decoder(tar, enc_output2, training, look_ahead_mask[1], dec_padding_mask[1])
        dec_output3, attention_weights3 = self.decoder(tar, enc_output3, training, look_ahead_mask[2], dec_padding_mask[2])
        dec_output = self.concat([dec_output1, dec_output2, dec_output3])
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, [attention_weights1, attention_weights2, attention_weights3]

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, pe_input, pe_target, rate=0.1):
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

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)        
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


class Execution(tf.Module):
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    def __init__(self, transformer, loss_object, train_loss, train_accuracy, optimizer):
        self.transformer = transformer
        self.loss_object = loss_object
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.optimizer = optimizer

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask    
        return tf.reduce_mean(loss_)

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp1, inp2, inp3, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask1, combined_mask1, dec_padding_mask1 = create_masks(inp1, tar_inp)
        enc_padding_mask2, combined_mask2, dec_padding_mask2 = create_masks(inp2, tar_inp)
        enc_padding_mask3, combined_mask3, dec_padding_mask3 = create_masks(inp3, tar_inp)
        enc_padding_mask = [enc_padding_mask1, enc_padding_mask2, enc_padding_mask3]
        combined_mask = [combined_mask1, combined_mask2, combined_mask3]
        dec_padding_mask = [dec_padding_mask1, dec_padding_mask2, dec_padding_mask3]
        
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer([inp1, inp2, inp3], tar_inp, 
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
        encoder_input1 = tf.expand_dims(inp_sentence[0], 0)
        encoder_input2 = tf.expand_dims(inp_sentence[1], 0)
        encoder_input3 = tf.expand_dims(inp_sentence[2], 0)
        encoder_input = [encoder_input1, encoder_input2, encoder_input3]
        decoder_input = np.asarray([vocab['<start>']])
        output = tf.expand_dims(decoder_input, 0)
            
        for i in range(maxlen):
            enc_padding_mask1, combined_mask1, dec_padding_mask1 = create_masks(encoder_input1, output)
            enc_padding_mask2, combined_mask2, dec_padding_mask2 = create_masks(encoder_input2, output)
            enc_padding_mask3, combined_mask3, dec_padding_mask3 = create_masks(encoder_input3, output)
            enc_padding_mask = [enc_padding_mask1, enc_padding_mask2, enc_padding_mask3]
            combined_mask = [combined_mask1, combined_mask2, combined_mask3]
            dec_padding_mask = [dec_padding_mask1, dec_padding_mask2, dec_padding_mask3]
            predictions, attention_weights = self.transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if predicted_id == vocab['<end>']:
                return tf.squeeze(output, axis=0), attention_weights
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0), attention_weights

def generate(inp_sentence, vocab, maxlen, model):
    encoder_input1 = tf.expand_dims(inp_sentence[0], 0)
    encoder_input2 = tf.expand_dims(inp_sentence[1], 0)
    encoder_input3 = tf.expand_dims(inp_sentence[3], 0)
    encoder_input = [encoder_input1, encoder_input2, encoder_input3]
    decoder_input = np.asarray([vocab['<start>']])
    output = tf.expand_dims(decoder_input, 0)
        
    for i in range(maxlen):
        enc_padding_mask1, combined_mask1, dec_padding_mask1 = create_masks(encoder_input1, output)
        enc_padding_mask2, combined_mask2, dec_padding_mask2 = create_masks(encoder_input2, output)
        enc_padding_mask3, combined_mask3, dec_padding_mask3 = create_masks(encoder_input3, output)
        enc_padding_mask = [enc_padding_mask1, enc_padding_mask2, enc_padding_mask3]
        combined_mask = [combined_mask1, combined_mask2, combined_mask3]
        dec_padding_mask = [dec_padding_mask1, dec_padding_mask2, dec_padding_mask3]
        predictions, attention_weights = model(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if predicted_id == vocab['<end>']:
            return tf.squeeze(output, axis=0), attention_weights
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights