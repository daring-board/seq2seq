import sys, os
import pickle
import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
import bert

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, Z, batch_size):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.batch_size = batch_size

    def __getitem__(self, idx):
        start, end = idx*self.batch_size, (idx+1)*self.batch_size
        X, Y = np.array(self.X[start: end]), np.array(self.Y[start: end])
        Z = np.array(self.Z[start: end])
        return [X, Y], Z

    def __len__(self):
        return int(len(self.X) / self.batch_size)

    def on_epoch_end(self):
        pass

if __name__ == '__main__':
    root_path = './bert_model/albert/'
    maxlen = 32
    sp = spm.SentencePieceProcessor()
    sp.Load(root_path + 'wiki-ja_albert.model')

    corpus = pickle.load(open('data/train_corpus.pkl', 'rb'))
    labels = pickle.load(open('data/label_corpus.pkl', 'rb'))
    segment = pickle.load(open('data/segment.pkl', 'rb'))
    output_vocab = pickle.load(open('data/output_vocab.pkl', 'rb'))
    length = len(corpus)

    SEQ_LEN = 2 * maxlen
    BATCH_SIZE = 16
    EPOCH = 2
    learning_rate = 1e-4

    rate = 0.9
    train, valid = corpus[:int(length*rate)], corpus[int(length*rate):int(length*(rate+0.02))]
    seg_t, seg_v = segment[:int(length*rate)], segment[int(length*rate):int(length*(rate+0.02))]
    lbl_t, lbl_v = labels[:int(length*rate)], labels[int(length*rate):int(length*(rate+0.02))]
    train_gen = DataGenerator(train, seg_t, lbl_t, BATCH_SIZE)
    valid_gen = DataGenerator(valid, seg_v, lbl_v, BATCH_SIZE)

    model_dir = root_path
    config = 'albert_config.json'
    checkpoint = 'model.ckpt-1400000'

    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="albert")

    max_seq_len = SEQ_LEN
    l_input_ids      = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    x = l_bert([l_input_ids, l_token_type_ids])
    output = tf.keras.layers.Dense(len(output_vocab))(x)
    model = tf.keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)]) 
    model.summary()

    bert.load_albert_weights(l_bert, model_dir + checkpoint)
    print("fa")
    model.compile(
        optimizer=get_optimizer(len(train), BATCH_SIZE, EPOCH, learning_rate),
        loss=loss_function,
        metrics=[acc_function]
    )
    model.fit_generator(
        train_gen,
        steps_per_epoch=int(length*rate / BATCH_SIZE),
        epochs=EPOCH
    )
    model.save('./models/bert_finetune.h5')

    x = model.predict([np.array([corpus[0]]), np.array([segment[0]])])
    ret = []
    ret = [np.argmax(t) for t in x[0]]
    print(ret)
    print(labels[0])