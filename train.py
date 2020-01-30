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
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    BATCH_SIZE = 512
    output_max_len = 32
    EPOCHS = 20
    steps_per_epoch = int(len(X_train) / BATCH_SIZE) * EPOCHS
    warmup_step = int(steps_per_epoch / 10)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    transformer = TransformerEX(vocab_size, num_layers, dff, d_model, num_heads, dropout_rate)
    model = transformer.build_model()
    mtx_obj = Functions(model ,loss_object, output_max_len, vocab)

    model.compile(optimizer='adam', loss=mtx_obj.loss_function, metrics=[mtx_obj.accuracy])
    model.summary()
    # model.load_weights('./models/param.hdf5')

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001, verbose=1)
    ]
    train_gen = DataSequence(X_train, Y_train, BATCH_SIZE)
    test_gen = DataSequence(X_test, Y_test, BATCH_SIZE)
    for _ in range(10):
        start = time.time()
        model.fit_generator(
            train_gen,
            len(train_gen),
            EPOCHS,
            validation_data=test_gen,
            validation_steps=len(test_gen),
            callbacks=callbacks,
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
