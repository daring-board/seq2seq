import os
import pickle
import time
import random
import numpy as np
import keras
from keras_bert import get_custom_objects
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
    in_idx2out_idx = pickle.load(open('data/in_idx2out_idx.pkl', 'rb'))
    length = len(corpus)

    SEQ_LEN = 2 * maxlen
    BATCH_SIZE = 16
    EPOCH = 5
    learning_rate = 1e-4

    model = keras.models.load_model(
            'models/bert_finetune.h5',
            compile=False,
            custom_objects=get_custom_objects()
        )
    model.summary()
    
    num = 25
    text = corpus[num]
    seg_ids = segment[num]
    for t in text:
        if t == 0:
            print()
            break
        print(sp.id_to_piece(t), end='')
    start = 0
    sep_id = sp.piece_to_id('[SEP]')
    cls_id = sp.piece_to_id('[CLS]')
    mask_id = sp.piece_to_id('[MASK]')
    for i, t in enumerate(text):
        if t == mask_id: start = i

    print('Distributed')
    out_idx2in_idx = {val: key for key, val in in_idx2out_idx.items()}

    print(text)
    print(seg_ids)

    count = 0
    predict = model.predict([np.array([text]), np.array([seg_ids])])
    predict = [np.argmax(t) for t in predict[0]][start]
    predict = out_idx2in_idx[predict]
    while predict != sep_id and count < maxlen and predict != 0:
        text[start + count] = predict
        seg_ids[start + count] = 1
        count += 1
        text[start + count] = mask_id
        seg_ids[start + count] = 1

        predict = model.predict([np.array([text]), np.array([seg_ids])])
        predict = [np.argmax(t) for t in predict[0]][start + count]
        predict = out_idx2in_idx[predict]

    print(text)
    for t in text:
        if t == 0: break
        print(sp.id_to_piece(t), end='')
    print()
    
