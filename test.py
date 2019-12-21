import sys
import pickle
import numpy as np
import pandas as pd
from model import *
import sentencepiece as spm
import tensorflow as tf
import bert

if __name__ == '__main__':
    maxlen = 32
    sp = spm.SentencePieceProcessor()
    sp.Load('./bert_model/wiki-ja.model')

    corpus = pickle.load(open('data/train_corpus.pkl', 'rb'))
    labels = pickle.load(open('data/label_corpus.pkl', 'rb'))
    segment = pickle.load(open('data/segment.pkl', 'rb'))
    output_vocab = pickle.load(open('data/output_vocab.pkl', 'rb'))
    length = len(corpus)

    SEQ_LEN = 2 * maxlen
    BATCH_SIZE = 16
    EPOCH = 100
    d_model, dff, num_head = 768, 4 * 768, 12

    model_dir = "./bert_model/"
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    loss = Loss(loss_obj)
    bert_ft = BertFineTune(l_bert, d_model, dff, num_head)
    model = bert_ft.build_model(SEQ_LEN, len(output_vocab))
    model.summary()
    model.compile(loss=loss.loss_function, metrics=[accuracy])

    rate = 0.8
    train, valid = corpus[:int(length*rate)], corpus[int(length*rate):int(length*(rate+0.02))]
    seg_t, seg_v = segment[:int(length*rate)], segment[int(length*rate):int(length*(rate+0.02))]
    lbl_t, lbl_v = labels[:int(length*rate)], labels[int(length*rate):int(length*(rate+0.02))]
    train_gen = DataGenerator(train, seg_t, lbl_t, BATCH_SIZE)
    valid_gen = DataGenerator(valid, seg_v, lbl_v, BATCH_SIZE)
    model.fit_generator(
        train_gen,
        steps_per_epoch=int(length*rate / BATCH_SIZE),
        epochs=EPOCH
    )

    x = model.predict([np.array([corpus[0]]), np.array([segment[0]])])
    ret = []
    ret = [np.argmax(t) for t in x[0]]
    print(ret)
    print(labels[0])