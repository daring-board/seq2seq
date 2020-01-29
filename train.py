import os, sys
import pickle
import time
import numpy as np
import tensorflow as tf

from model import *

if __name__ == '__main__':
    maxlen = 64
    with open('data/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    with open('data/response.pkl', 'rb') as f:
        response = pickle.load(f)
    with open('data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    X_train, Y_train, Z_train = [], [], []
    for idx, l in enumerate(corpus):
        li0 = l.tolist()
        li1 = response[idx]
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
    dff = 256
    num_heads = 8
    dropout_rate = 0.2
    BATCH_SIZE = 512
    output_max_len = 32
    steps_per_epoch = int(len(X_train) / BATCH_SIZE)

    learning_rate = CustomSchedule(d_model, steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    model = transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout_rate)
    mtx_obj = Functions(model ,loss_object, output_max_len, vocab)

    model.compile(optimizer=optimizer, loss=mtx_obj.loss_function, metrics=[mtx_obj.accuracy])
    model.summary()
    # model.load_weights('./models/param.hdf5')

    EPOCHS = 10
    train_gen = DataSequence(X_train, Y_train, BATCH_SIZE)
    test_gen = DataSequence(X_test, Y_test, BATCH_SIZE)
    for _ in range(10):
        start = time.time()
        model.fit_generator(
            train_gen,
            len(train_gen),
            EPOCHS,
            validation_data=test_gen,
            validation_steps=len(test_gen)
        )

        for idx in range(3):
            in_sentence, ret_sentence = '', ''
            inp = np.asarray([X_test[idx]])
            ret = mtx_obj.evaluate(inp)
            for n in inp[0]:
                in_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(in_sentence)

            for n in ret.numpy():
                ret_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(ret_sentence)
        model.save_weights('./models/param.hdf5')
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
