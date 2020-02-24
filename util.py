import os
import pickle
import time
import numpy as np
import tensorflow as tf
import sentencepiece as spm

from model import *

class ChatEngine():
    def __init__(self):
        self.checkGPU()
        self.maxlen = 32
        with open('./data/dict.pkl', 'rb') as f:
            self.index = pickle.load(f)
        self.vocab = {v: k for k, v in self.index.items()}

        vocab_size = len(self.vocab) + 1
        num_layers = 3
        d_model = 64
        dff = 512
        num_heads = 8
        dropout_rate = 0.1 
        self.transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size, 
                          pe_input=vocab_size, 
                          pe_target=vocab_size,
                          rate=dropout_rate
                    )
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = "./models/training_checkpoints/"
        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print ('Latest checkpoint restored!!')
        mpath = 'models/sentensepice'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(mpath+'.model')

    def checkGPU(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def response(self, sentences):
        line = sentences[1]
        parts = self.sp.encode_as_pieces(line)
        parts = ['<start>'] + parts + ['<end>']
        num_parts = [self.vocab[part] for part in parts]
        inp = np.asarray(num_parts)
        in_sentence, ret_sentence = '', ''
        ret, _ = generate(inp, self.vocab, self.maxlen, self.transformer)

        for n in ret.numpy():
            if n == self.vocab['<end>']: break
            ret_sentence += self.index[n]
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '')
        return ret_sentence
