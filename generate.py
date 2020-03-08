import os
import pickle
import time
import random
import numpy as np
import neologdn
import tensorflow as tf
import sentencepiece as spm

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
    with open('data/X_corpus.pkl', 'rb') as f:
        X_corpus = pickle.load(f)
    with open('data/Y_corpus.pkl', 'rb') as f:
        Y_corpus = pickle.load(f)
    with open('data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    X_train, Y_train = [], []
    for idx in random.sample(range(len(X_corpus)), len(X_corpus)):
        X_train.append(X_corpus[idx].tolist())
        Y_train.append(Y_corpus[idx].tolist())
    del X_corpus, Y_corpus
    print(X_train[0])
    print(Y_train[0])

    rate = 0.9
    length = len(X_train)
    X_train, X_test = X_train[: int(length*rate)], X_train[int(length*rate):int(length*(rate+0.1))]
    Y_train, Y_test = Y_train[: int(length*rate)], Y_train[int(length*rate):int(length*(rate+0.1))]
    print(len(X_train))
    print(len(X_test))
    print(X_train[0])

    vocab_size = len(vocab) + 1
    num_layers = 4
    d_model = 256
    dff = 512
    num_heads = 8
    dropout_rate = 0.2

    BATCH_SIZE = 512
    steps_per_epoch = int(len(X_train) / BATCH_SIZE)

    learning_rate = CustomSchedule(d_model, warmup_steps=50000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = TransformerVAE(
                        num_layers, d_model, num_heads, dff,
                        vocab_size, vocab_size, 
                        pe_input=vocab_size, 
                        pe_target=vocab_size,
                        rate=dropout_rate
                    )

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
    while True:
        print('Conversation:')
        line = input('> ')
        if not line: break
        line = neologdn.normalize(line)
        parts = ' '.join(split_sentence(analyzer, line))
        parts = sp.encode_as_pieces(parts)
        if len(history) == 1:
            parts = ['<start>'] + parts + ['<end>']
        else:
            parts = history + parts + ['<end>']
        num_parts = [vocab[part] for part in parts]
        inp = np.asarray(num_parts)

    start = time.time()
    for idx in range(3):
        in_sentence, ret_sentence = '', ''
        inp = np.asarray(X_test[idx])
        expect = np.asarray(Y_test[idx])
        ret = estimate(transformer, inp, vocab, maxlen, d_model)
        for n in inp:
            in_sentence += index[n] + ' '
            if n == vocab['<end>']: break
        in_sentence = in_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        for n in ret.numpy():
            ret_sentence += index[n] + ' '
            if n == vocab['<end>']: break
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '').replace('▁', '').replace(' ', '')
        print('response: %s'%ret_sentence)
        print()
        line = neologdn.normalize(ret_sentence)
        parts = ' '.join(split_sentence(analyzer, line))
        parts = sp.encode_as_pieces(parts)
        if len(history) == 1:
            history = ['<start>'] + parts + ['<sep>']
        else:
            history = ['<start>'] + history + ['<sep>'] + parts + ['<sep>']
