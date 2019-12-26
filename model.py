import numpy as np
import keras_bert
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps
from keras_multi_head import MultiHeadAttention
import keras
import keras.backend as K

class BertFineTune():
    def __init__(self, model_dir, config_filename, ckpt_filename, seq_len):
        print('Construnct BertFineTune')
        self.bert = load_trained_model_from_checkpoint(
                model_dir+config_filename,
                model_dir+ckpt_filename,
                training=True,
                trainable=True,
                seq_len=seq_len
            )
        
    def build_model(self, vocab_size):
        x = self.bert.get_layer(name='Encoder-12-FeedForward-Norm').output
        output = keras.layers.Dense(vocab_size)(x)
        model = keras.Model(inputs=[self.bert.input[0], self.bert.input[1]], outputs=output)
        return model

def loss_function(real, pred):
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits=True)
    return K.tf.reduce_mean(loss_)

def acc_function(real, pred):
    pred_ = K.tf.to_int32(K.tf.argmax(pred, 2))
    real = K.cast(real, 'int32')
    correct = K.cast(K.equal(real, pred_), 'int32')
    acc = K.tf.reduce_mean(
            K.cast(
                K.tf.equal(pred_, real), 'float32'
            ),
            axis=0,
            name='acc'
        )
    return acc

def get_optimizer(input_size, batch_size, epochs, lr):
    decay_steps, warmup_steps = calc_train_steps(
            input_size,
            batch_size=batch_size,
            epochs=epochs,
        )
    return AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=lr)

