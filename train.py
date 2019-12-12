import os, sys
import pickle
import time
import numpy as np
import tensorflow as tf

from model import *

if __name__ == '__main__':
    maxlen = 32
    with open('data/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    with open('data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}
    print(index[0])

    X_train, Y_train, Z_train = [], [], []
    for idx, l in enumerate(corpus[:-1]):
        li0 = l.tolist()
        li1 = corpus[idx+1].tolist()
        X_train.append(li0)
        Y_train.append(li1)

    rate = 0.9
    length = len(X_train)
    X_train, X_test = X_train[: int(length*rate)], X_train[int(length*rate):int(length*(rate+0.1))]
    Y_train, Y_test = Y_train[: int(length*rate)], Y_train[int(length*rate):int(length*(rate+0.1))]
    print(len(X_train))
    print(len(X_test))
    print(X_train[0])

    vocab_size = len(vocab) + 1
    num_layers = 2
    d_model = 64
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    BATCH_SIZE = 1024

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    model = transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout_rate)
    mtx_obj = Functions(model ,loss_object, maxlen, vocab)

    tf.keras.backend.clear_session()
    model.compile(optimizer=optimizer, loss=mtx_obj.loss_function, metrics=[mtx_obj.accuracy])
    model.summary()

    EPOCHS = 150
    for epoch in range(EPOCHS):
        print('EPOCH: %d'%epoch)
        start = time.time()
                
        steps_per_epoch = int(len(X_train) / BATCH_SIZE)
        print('BATCH: ')
        print(steps_per_epoch)
        for batch in range(steps_per_epoch):
            sys.stdout.write('\r{}'.format(batch))
            inp = np.asarray(X_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            trg = np.asarray(Y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            model.fit([inp, trg[:, :-1]], trg[:, 1:], epochs=1, verbose=0)

        for idx in range(3):
            in_sentence, ret_sentence = '', ''
            inp = np.asarray([X_test[idx]])
            ret = mtx_obj.evaluate(inp)
            for n in inp[0]:
                if n == vocab['<end>']: break
                in_sentence += index[n] + ' '
            print(in_sentence)

            for n in ret.numpy():
                if n == vocab['<end>']: break
                ret_sentence += index[n] + ' '
            print(ret_sentence)
        if epoch % 5 == 0:
            model.save_weights('./models/graph_model/param.hdf5')
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
