import os
import pickle
import time
import numpy as np
import tensorflow as tf

from model import *

if __name__ == '__main__':
    maxlen = 32
    with open('./data/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    with open('./data/dict.pkl', 'rb') as f:
        index = pickle.load(f)
    vocab = {v: k for k, v in index.items()}

    X_train, Y_train = [], []
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

    vocab_size = len(vocab) + 1
    embedding_dim = 64
    units = 512
    BATCH_SIZE = 64

    encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction='none'
                )

    checkpoint_dir = './models/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
                    optimizer=optimizer,
                    encoder=encoder,
                    decoder=decoder
                )
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    EPOCHS = 1
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        enc_hidden = encoder.initialize_hidden_state()

        steps_per_epoch = int(len(X_train) / BATCH_SIZE)
        for batch in range(steps_per_epoch):
            inp = np.asarray(X_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            trg = np.asarray(Y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE])
            batch_loss = train_step(inp, trg, enc_hidden, BATCH_SIZE, vocab, optimizer, loss_object, encoder, decoder)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1,
                    batch,
                    batch_loss.numpy()
                ))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(
            epoch + 1, 
            total_loss / steps_per_epoch
        ))

        for idx in range(3):
            in_sentence = ''
            inp = np.asarray([X_test[idx]])
            ret, nums, _ = evaluate(inp, vocab, index, units, maxlen, encoder, decoder)
            print(inp)
            print(nums)
            print(ret)
        print('Time taken for 1 epoch {} sec\n'.format(
            time.time() - start
        ))
