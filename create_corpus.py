import sys, random
import pickle
import numpy as np
import sentencepiece as spm

if __name__ == '__main__':
    # f_name = 'test.txt'
    f_name = 'full_conv.txt'
    path = 'data/'+f_name
    texts = [l.strip() for l in open(path, 'r', encoding='utf8') if l!='\n']

    sp = spm.SentencePieceProcessor()
    sp.Load('./bert_model/japanese/wiki-ja.model')

    maxlen = 64
    corpus, labels , input_segment = [], [], []
    output_vocab = {'<unk>': 0}
    for n in range(len(texts[:-1])):
        text1, text2 = texts[n], texts[n+1]
        tokens = [0] * maxlen

        t_list = sp.encode_as_pieces(text1)
        if len(t_list) > (maxlen / 2 - 2):
            t_list = t_list[: int(maxlen / 2 - 2)]
        t_list = ['[CLS]'] + t_list + ['[SEP]']
        for i, t_i in enumerate(t_list):
            tokens[i] = sp.piece_to_id(t_i)
            output_vocab[t_i] = tokens[i]
        length = len(t_list)

        t_list = sp.encode_as_pieces(text2)
        if len(t_list) > (maxlen / 2 - 2):
            t_list = t_list[: int(maxlen / 2 - 2)]
        t_list = t_list + ['[SEP]']

        segment = [0] * length
        label = [0] * length
        for j, t_j in enumerate(t_list):
            output_vocab[t_j] = tokens[i]
            tmp = [t for t in tokens]
            tmp[length + j] = sp.piece_to_id('[MASK]')
            t_id = sp.piece_to_id(t_j)
            label.append(t_id)
            segment.append(1)

            labels.append(label + [0] * (maxlen - len(label)))
            corpus.append(tmp)
            input_segment.append(segment + [0] * (maxlen - len(segment)))
            tokens[length + j] = t_id
            label = [l for l in label]
            segment = [s for s in segment]

    vcb2idx = {key: idx for idx, key in enumerate(output_vocab.keys())}
    output_vocab = {idx: key for idx, key in enumerate(output_vocab.keys())}
    in_idx2out_idx = {sp.piece_to_id(val): key for key, val in output_vocab.items()}

    tmps = []
    for label in labels:
        tmp = [vcb2idx[sp.id_to_piece(t)] for t in label]
        tmps.append(tmp)
    labels = tmps

    for i in range(30):
        print(corpus[i])
        print(labels[i])
        print(input_segment[i])

    print(len(corpus))
    print(len(labels))
    print(len(input_segment))

    with open('data/train_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
    with open('data/label_corpus.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('data/segment.pkl', 'wb') as f:
        pickle.dump(input_segment, f)
    print(len(output_vocab))
    pickle.dump(output_vocab, open('data/output_vocab.pkl', 'wb'))
    pickle.dump(in_idx2out_idx, open('data/in_idx2out_idx.pkl', 'wb'))
