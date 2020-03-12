import os
import pickle
import time
import numpy as np
import neologdn
import tensorflow as tf
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

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
        d_model = 512
        dff = 128
        num_heads = 32
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

    def split_line(self, line):
        line = neologdn.normalize(line)
        parts = ' '.join(self.split_sentence(line))
        parts = self.sp.encode_as_pieces(parts)
        return parts

    def join_parts(self, parts):
        sentence = ''
        for n in parts:
            sentence += self.index[n]
            if n == self.vocab['<end>']: break
        sentence = sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        return sentence

    def history2parts(self, history):
        parts = ['<start>']
        for l in history:
            tmp = self.split_line(l)
            parts += tmp + ['<sep>']
        parts = parts[:-1] + ['<end>']
        return parts

    def response(self, sentences):
        history, line = sentences[0], sentences[1]

        history += [line]
        hi_sentence = '\n  '.join(history)
        print('history: %s'%hi_sentence)

        parts = self.history2parts(history)
        num_parts = [self.vocab[part] for part in parts]
        inp = np.asarray(num_parts)
        in_sentence, ret_sentence = '', ''
        ret, _ = generate(inp, self.vocab, self.maxlen, self.transformer)

        ret_sentence = self.join_parts(ret.numpy())
        history += [ret_sentence]
        return ret_sentence, history[-5:]
