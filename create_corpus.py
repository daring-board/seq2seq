import sys, random
import pickle
import numpy as np
import sentencepiece as spm

if __name__ == '__main__':
    # f_name = 'org_data.txt'
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

        segment = [0] * length + [1] * len(t_list) + [0] * (maxlen - (length + len(t_list)))
        for j in range(length, length + len(t_list)):
            tokens[j] = sp.piece_to_id(t_list[j - length])
        for i, t in enumerate(t_list):
            output_vocab[t] = tokens[i]

        idx_list = [random.choice(range(len(t_list[:-1]))) for _ in range(2)]
        for idx in idx_list:
            t_list[idx] = '[MASK]'
        mask_id = sp.piece_to_id('[MASK]')

        tmp = [0] * maxlen
        for i, t in enumerate(tokens[:length]):
            tmp[i] = t
        for j in range(length, length + len(t_list)):
            tmp[j] = sp.piece_to_id(t_list[j - length])
        
        tokens = [tokens[idx] if tmp[idx] == mask_id else 0 for idx in range(len(tokens))]
        labels.append(tokens)
        corpus.append(tmp)
        input_segment.append(segment)

    vcb2idx = {key: idx for idx, key in enumerate(output_vocab.keys())}
    output_vocab = {idx: key for idx, key in enumerate(output_vocab.keys())}
    in_idx2out_idx = {sp.piece_to_id(val): key for key, val in output_vocab.items()}

    tmps = []
    for label in labels:
        tmp = [vcb2idx[sp.id_to_piece(t)] for t in label]
        tmps.append(tmp)
    labels = tmps

    for i in range(10):
        print(corpus[i])
        print(labels[i])
        print(input_segment[i])

    with open('data/train_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
    with open('data/label_corpus.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('data/segment.pkl', 'wb') as f:
        pickle.dump(input_segment, f)
    print(len(output_vocab))
    pickle.dump(output_vocab, open('data/output_vocab.pkl', 'wb'))
    pickle.dump(in_idx2out_idx, open('data/in_idx2out_idx.pkl', 'wb'))
