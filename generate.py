import os
import pickle
import time
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

from model import *

def checkGPU():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

if __name__ == '__main__':
    checkGPU()
    maxlen = 32
    with open('./data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    vocab_size = len(vocab) + 1
    num_layers = 3
    d_model = 64
    dff = 512
    num_heads = 8
    dropout_rate = 0.2

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size, 
                          pe_input=vocab_size, 
                          pe_target=vocab_size,
                          rate=dropout_rate
                    )
    execution = Execution(transformer, loss_object, train_loss, train_accuracy, optimizer)

    checkpoint_path = "./models/training_checkpoints/"
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    

    mpath = 'models/sentensepice'
    sp = spm.SentencePieceProcessor()
    sp.load(mpath+'.model')

    while True:
        print('Conversation:')
        line = input('> ')
        if not line: break
        parts = sp.encode_as_pieces(line)
        parts = ['<start>'] + parts + ['<end>']
        num_parts = [vocab[part] for part in parts]
        inp = np.asarray(num_parts)

        in_sentence, ret_sentence = '', ''
        ret, _ = execution.evaluate(inp, vocab, maxlen)
        for n in inp:
            in_sentence += index[n]
            if n == vocab['<end>']: break
        in_sentence = in_sentence.replace('<start>', '').replace('<end>', '')
        for n in ret.numpy():
            ret_sentence += index[n]
            if n == vocab['<end>']: break
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '')
        print('response: %s'%ret_sentence)
        print()
