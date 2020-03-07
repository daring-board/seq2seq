import os
import pickle
import time
import random
import numpy as np
import tensorflow as tf

from model import *

if __name__ == '__main__':
    maxlen = 32
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
    num_layers = 3
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.2
    BATCH_SIZE = 32
    EPOCHS = 300
    steps_per_epoch = int(len(X_train) / BATCH_SIZE)

    learning_rate = CustomSchedule(d_model, warmup_steps=(EPOCHS * steps_per_epoch) / 2)
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

    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for batch in range(steps_per_epoch):
            inp = np.asarray(X_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            trg = np.asarray(Y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            execution.train_step(inp, trg)
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                        train_loss.result(), 
                                                        train_accuracy.result()))

        for idx in range(3):
            in_sentence, exp_sentence, ret_sentence = '', '', ''
            inp = np.asarray(X_test[idx])
            expect = np.asarray(Y_test[idx])
            ret, _ = execution.evaluate(inp, vocab, maxlen, expect)
            for n in inp:
                in_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(in_sentence.replace('▁', '').replace(' ', ''))
            for n in expect:
                exp_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(exp_sentence.replace('▁', '').replace(' ', ''))
            for n in ret.numpy():
                ret_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(ret_sentence.replace('▁', '').replace(' ', ''))
            print()
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    tf.saved_model.save(execution, 'models/')
