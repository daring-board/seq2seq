import os
import pickle
import time
import numpy as np
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

if __name__ == '__main__':
    checkGPU()
    maxlen = 32
    with open('./data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    vocab_size = len(vocab) + 1
    num_layers = 2
    d_model = 64
    dff = 256
    num_heads = 8
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = TransformerEX(num_layers, d_model, num_heads, dff,
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

    line1 = ''
    while True:
        print('Conversation:')
        line2 = input('> ')
        if not line2: break
        parts1 = sp.encode_as_pieces(line1)
        parts2 = sp.encode_as_pieces(line2)
        parts1 = ['<start>'] + parts1 + ['<end>']
        parts2 = ['<start>'] + parts2 + ['<end>']
        num_parts1 = [vocab[part] for part in parts1]
        num_parts2 = [vocab[part] for part in parts2]
        inp1 = np.asarray(num_parts1)
        inp2 = np.asarray(num_parts2)

        in_sentence1, in_sentence2, ret_sentence = '', '', ''
        ret, _ = execution.evaluate([inp1, inp2], vocab, maxlen)
        for n in inp1:
            if n == vocab['<end>']: break
            in_sentence1 += index[n]
        in_sentence1 = in_sentence1.replace('<start>', '').replace('<end>', '')
        print('prequery: %s'%in_sentence1[1:])
        for n in inp2:
            if n == vocab['<end>']: break
            in_sentence2 += index[n]
        in_sentence2 = in_sentence2.replace('<start>', '').replace('<end>', '')
        print('query: %s'%in_sentence2[1:])
        for n in ret.numpy():
            if n == vocab['<end>']: break
            ret_sentence += index[n]
        ret_sentence = ret_sentence.replace('<start>', '').replace('<end>', '')
        print('response: %s'%ret_sentence[1:])
        print()
        line1 = ret_sentence
