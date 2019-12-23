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
    
    org_text = corpus[0]
    for t in org_text:
        if t == 0:
            print()
            break
        print(sp.id_to_piece(t), end='')
    print(org_text)
    seg_ids = segment[0]
    text, flg, start = [], False, 0
    sep_id = sp.piece_to_id('[SEP]')
    cls_id = sp.piece_to_id('[CLS]')
    mask_id = sp.piece_to_id('[MASK]')
    for i, t in enumerate(corpus[0]):
        text.append(t)
        if t == sep_id:
            start = i + 1
            break
    for i in range(start, len(corpus[0])):
        text.append(0)
    length = random.randint(3, 31)
    end = start + length

    for i in range(SEQ_LEN):
        if start <= i and i < end: seg_ids[i] = 1
        else: seg_ids[i] = 0
    vocab_size = len(output_vocab)

    for i in range(start, end):
        t = random.randint(1, vocab_size)
        while t == sep_id or t == cls_id or t == mask_id:
            t = random.randint(1, vocab_size)
        text[i] = t
    text[end] = sep_id

    print('Distributed')
    out_idx2in_idx = {val: key for key, val in in_idx2out_idx.items()}
    mask = start
    for l in range(120):
        print(mask)
        text[mask] = mask_id
        predict = model.predict([np.array([text]), np.array([seg_ids])])
        predict = [np.argmax(t) for t in predict[0]]
        if predict[mask] == 0:
            mask += 1
            continue
        if predict[mask] == mask_id:
            random.randint(1, vocab_size)
            continue
        text[mask] = out_idx2in_idx[predict[mask]]
        print(text)
        mask += 1
        if mask == end:
            mask = start

    for t in text:
        if t == 0: break
        print(sp.id_to_piece(t), end='')
    print()
    
