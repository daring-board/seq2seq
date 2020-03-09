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
from tensorflow.keras.preprocessing import sequence

from model import *

def checkGPU():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")

def split_sentence(analyzer, l):
    return [token for token in analyzer.analyze(l)]

if __name__ == '__main__':
    checkGPU()
    maxlen = 128
    with open('./data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    vocab_size = len(vocab) + 1
    num_layers = 4
    d_model = 256
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

    tokenizer = Tokenizer()
    char_filters = [UnicodeNormalizeCharFilter()]
    token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
    analyzer = Analyzer(char_filters, tokenizer, token_filters)

    history = ['<end>']
    expect = []
    train_loss.reset_states()
    train_accuracy.reset_states()
    while True:
        print('Conversation:')
        line = input('> ')
        if not line: break
        line = neologdn.normalize(line)
        parts = ' '.join(split_sentence(analyzer, line))
        parts = sp.encode_as_pieces(parts)
        is_first = True if len(history) == 1 else False
        if is_first:
            parts = ['<start>'] + parts + ['<end>']
        else:
            parts = history + parts + ['<end>']

            hi_sentence = ''
            num_parts = [vocab[part] for part in parts]
            for n in num_parts:
                hi_sentence += index[n]
                if n == vocab['<end>']: break
            hi_sentence = hi_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
            print('history: %s'%hi_sentence)

        num_parts = [vocab[part] for part in parts]
        inp = np.asarray(num_parts)

        in_sentence, ret_sentence = '', ''
        ret, _ = execution.evaluate(inp, vocab, maxlen)
        for n in inp:
            in_sentence += index[n]
            if n == vocab['<end>']: break
        in_sentence = in_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        for n in ret.numpy():
            ret_sentence += index[n]
            if n == vocab['<end>']: break
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        print('response: %s'%ret_sentence)

        if is_first:
            history = ['<start>'] + parts + ['<sep>']
        else:
            line = neologdn.normalize(ret_sentence)
            parts = ' '.join(split_sentence(analyzer, line))
            parts = sp.encode_as_pieces(parts)
            history = ['<start>'] + history + ['<sep>'] + parts + ['<sep>']
        
        inp_maxlen = 256
        exp_maxlen = 128
        inp = history[:-1] + ['<end>']
        num_parts = [vocab[part] for part in parts]
        inp = np.asarray(num_parts)
        if not is_first:
            inp = sequence.pad_sequences([inp], maxlen=inp_maxlen, padding='post', truncating='pre')
            expect = sequence.pad_sequences([expect.numpy()], maxlen=exp_maxlen, padding='post', truncating='post')
            execution.train_step(inp, np.asarray(expect))
            ckpt_save_path = ckpt_manager.save()
            print ('Loss {:.4f} Accuracy {:.4f}'.format(train_loss.result(), train_accuracy.result()))
        else:
            inp = sequence.pad_sequences([inp], maxlen=inp_maxlen, padding='post', truncating='pre')
        
        expect, _ = execution.evaluate(inp[0], vocab, maxlen)
        exp_sentence = ''
        for n in expect.numpy():
            exp_sentence += index[n]
            if n == vocab['<end>']: break
        exp_sentence = exp_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        print('expect: %s'%exp_sentence)
        print()
