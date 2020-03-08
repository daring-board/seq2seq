import os
import pickle
import time
import numpy as np
import neologdn
import tensorflow as tf
import neologdn
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

from model import *

class ChatEngine():
    def __init__(self):
        self.checkGPU()
        with open('./data/dict.pkl', 'rb') as f:
            self.index = pickle.load(f)
        self.vocab = {v: k for k, v in self.index.items()}

        vocab_size = len(self.vocab) + 1
        num_layers = 4
        d_model = 256
        dff = 512
        num_heads = 8
        dropout_rate = 0.2
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

        tokenizer = Tokenizer()
        char_filters = [UnicodeNormalizeCharFilter()]
        token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
        self.analyzer = Analyzer(char_filters, tokenizer, token_filters)
        self.history = ['<end>']

    def checkGPU(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def split_sentence(self, l):
        return [token for token in self.analyzer.analyze(l)]

    def response(self, sentences):
        line = sentences[1]
        line = neologdn.normalize(line)
        line = ' '.join(self.split_sentence(line))
        parts = self.sp.encode_as_pieces(line)
        if len(self.history) == 1:
            parts = ['<start>'] + parts + ['<end>']
        else:
            parts = self.history + parts + ['<end>']
        num_parts = [self.vocab[part] for part in parts]
        inp = sequence.pad_sequences([num_parts], maxlen=self.maxlen, padding='post', truncating='pre')
        inp = np.asarray(inp)[0]
        in_sentence, ret_sentence = '', ''
        ret = estimate(self.transformer, inp, self.vocab, self.maxlen, self.d_model)

        for n in ret.numpy():
            if n == self.vocab['<end>']: break
            ret_sentence += self.index[n]
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '')
        line = neologdn.normalize(ret_sentence)
        parts = ' '.join(self.split_sentence(line))
        parts = self.sp.encode_as_pieces(parts)
        if len(self.history) == 1:
            self.history = ['<start>'] + parts + ['<sep>']
        else:
            self.history = ['<start>'] + self.history + ['<sep>'] + parts + ['<sep>']
        return ret_sentence
