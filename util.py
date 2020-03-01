import os
import pickle
import time
import numpy as np
import tensorflow as tf
import neologdn
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from tensorflow.keras.preprocessing import sequence

from model import *

class ChatEngine():
    def __init__(self):
        self.checkGPU()
        with open('./data/dict.pkl', 'rb') as f:
            self.index = pickle.load(f)
        self.vocab = {v: k for k, v in self.index.items()}

        vocab_size = len(self.vocab) + 1
        self.maxlen = 64
        num_layers = 3
        d_model = 128
        dff = 256
        num_heads = 8
        dropout_rate = 0.2
        self.d_model = d_model

        self.transformer = TransformerVAE(num_layers, d_model, num_heads, dff,
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

        tokenizer = Tokenizer()
        char_filters = [UnicodeNormalizeCharFilter()]
        token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
        self.analyzer = Analyzer(char_filters, tokenizer, token_filters)

    def split_sentence(self, analyzer, l):
        return [token for token in analyzer.analyze(l)]

    def checkGPU(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def response(self, sentences):
        line1, line2 = sentences[0], sentences[1]
        line1 = neologdn.normalize(line1)
        line2 = neologdn.normalize(line2)
        line1 = self.split_sentence(analyzer, line1)
        line2 = self.split_sentence(analyzer, line2)
        parts1 = self.sp.encode_as_pieces(line1)
        parts2 = self.sp.encode_as_pieces(line2)
        parts = ['<start>'] + parts1 + ['<sep>'] + parts2 + ['<end>']
        num_parts = [self.vocab[part] for part in parts]
        inp = sequence.pad_sequences([num_parts], maxlen=self.maxlen, padding='post', truncating='pre')
        inp = np.asarray(inp)[0]
        in_sentence, ret_sentence = '', ''
        ret = estimate(self.transformer, inp, self.vocab, self.maxlen, self.d_model)

        for n in ret.numpy():
            if n == self.vocab['<end>']: break
            ret_sentence += self.index[n]
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '')
        return ret_sentence
