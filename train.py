import sys, os
import pickle
import numpy as np
import pandas as pd
from model import *
import sentencepiece as spm

if __name__ == '__main__':
    root_path = './bert_model/japanese/'
    maxlen = 32
    sp = spm.SentencePieceProcessor()
    sp.Load(root_path + 'wiki-ja.model')

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
    config = 'bert_config.json'
    checkpoint = 'model.ckpt-1400000'
    bert_ft = BertFineTune(model_dir, config, checkpoint, SEQ_LEN)
    model = bert_ft.build_model(len(output_vocab))
    model.summary()
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