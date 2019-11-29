import os
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

from model import *

import re
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

def checkGPU():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

class Morph:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def analyze(self, text):
        char_filters = [UnicodeNormalizeCharFilter()]
        token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
        analyzer = Analyzer(char_filters, self.tokenizer, token_filters)
        parts = [token for token in analyzer.analyze(text)]
        return parts

if __name__ == '__main__':
    checkGPU()
    maxlen = 16
    with open('./data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    vocab_size = len(vocab) + 1
    embedding_dim = 32
    units = 256
    BATCH_SIZE = 64

    encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = './models/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
                    optimizer=optimizer,
                    encoder=encoder,
                    decoder=decoder
                )
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    morph = Morph()

    count = 0
    while True:
        print('Conversation: %d'%count)
        line = input('> ')
        if not line: break
        parts = ['<start>'] + morph.analyze(line) + ['<end>']
        num_parts = [vocab[part] for part in parts]
        embs = sequence.pad_sequences([num_parts], maxlen=maxlen, padding='post', truncating='post')
        inp = np.asarray(embs)

        in_sentence = ''
        ret, nums, attention = evaluate(inp, vocab, index, units, maxlen, encoder, decoder)
        for n in num_parts:
            if n == 0: continue
            else: in_sentence += index[n]
        print('query: %s'%in_sentence[7:-5])
        print('response: %s'%ret[:-6].replace(' ', ''))
        print()
        count += 1
