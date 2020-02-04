import os
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
    with open('data/distribution.pkl', 'rb') as f:
        count = pickle.load(f)
    vocab = {v: k for k, v in index.items()}
    count = {vocab[k]: v for k, v in count.items()}

    X1_train, X2_train, X3_train, Y_train = [], [], [], []
    for idx, l in enumerate(corpus[:-3]):
        li0 = l.tolist()
        li1 = corpus[idx+1].tolist()
        li2 = corpus[idx+2].tolist()
        li3 = corpus[idx+3].tolist()
        X1_train.append(li0)
        X2_train.append(li1)
        X3_train.append(li2)
        Y_train.append(li3)

    rate = 0.9
    length = len(X1_train)
    X1_train, X1_test = X1_train[: int(length*rate)], X1_train[int(length*rate):int(length*(rate+0.1))]
    X2_train, X2_test = X2_train[: int(length*rate)], X2_train[int(length*rate):int(length*(rate+0.1))]
    X3_train, X3_test = X3_train[: int(length*rate)], X3_train[int(length*rate):int(length*(rate+0.1))]
    Y_train, Y_test = Y_train[: int(length*rate)], Y_train[int(length*rate):int(length*(rate+0.1))]
    print(len(X1_train))
    print(len(X1_test))
    print(X1_train[0])

    vocab_size = len(vocab) + 1
    num_layers = 2
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.2
    BATCH_SIZE = 128
    steps_per_epoch = int(len(X1_train) / BATCH_SIZE)

    learning_rate = CustomSchedule(d_model, warmup_steps=2000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = TransformerEX(
                        num_layers, d_model, num_heads, dff,
                        vocab_size, vocab_size, 
                        pe_input=vocab_size, 
                        pe_target=vocab_size,
                        rate=dropout_rate
                    )
    execution = Execution(transformer, loss_object, train_loss, train_accuracy, optimizer)

    checkpoint_path = './models/training_checkpoints/'
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    EPOCHS = 10
    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for batch in range(steps_per_epoch):
            inp1 = np.asarray(X1_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            inp2 = np.asarray(X2_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            inp3 = np.asarray(X3_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            trg = np.asarray(Y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            execution.train_step(inp1, inp2, inp3, trg)
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        for idx in range(3):
            in1_sentence, in2_sentence, in3_sentence, ret_sentence = '', '', '', ''
            inp1 = np.asarray(X1_test[idx])
            inp2 = np.asarray(X2_test[idx])
            inp3 = np.asarray(X3_test[idx])
            expect = np.asarray(Y_test[idx])
            ret, _ = execution.evaluate([inp1, inp2, inp3], vocab, maxlen, expect)
            for n in inp1:
                in1_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(in1_sentence)
            for n in inp2:
                in2_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(in2_sentence)
            for n in inp3:
                in3_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(in3_sentence)
            for n in ret.numpy():
                ret_sentence += index[n] + ' '
                if n == vocab['<end>']: break
            print(ret_sentence)
            print()
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    tf.saved_model.save(execution, 'models/')
