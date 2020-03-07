import sys
import glob
import pickle
import neologdn
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from tensorflow.keras.preprocessing import sequence

def split_sentence(analyzer, l):
    return [token for token in analyzer.analyze(l)]

if __name__ == '__main__':
    files = glob.glob('data/tweet/*.txt')
    texts = []
    for path in files:
        ts = [l[5:].strip() if l!='\n' else l for l in open(path, 'r')]
        texts += ts

    # tokenizer = Tokenizer(mmap=True)
    tokenizer = Tokenizer()
    char_filters = [UnicodeNormalizeCharFilter()]
    token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
    analyzer = Analyzer(char_filters, tokenizer, token_filters)
    path = 'data/corpus.txt'
    with open(path, 'w', encoding='utf8') as f:
        for l in texts:
            l = neologdn.normalize(l)
            l = split_sentence(analyzer, l)
            f.write(' '.join(l) + '\n')

    mpath = 'models/sentensepice'
    template = '--input=%s --model_prefix=%s --vocab_size=10000'
    spm.SentencePieceTrainer.train(template%(path, mpath))
    sp = spm.SentencePieceProcessor()
    sp.load(mpath+'.model')

    dataset, conv = [], []
    for l in texts:
        if l != '\n':
            l = neologdn.normalize(l)
            l = ' '.join(split_sentence(analyzer, l))
            sn = sp.encode_as_pieces(l)
            conv.append(sn)
        else:
            dataset.append(conv)
            conv = []
    del analyzer
    del token_filters
    del char_filters
    del tokenizer
    print(dataset[:10])

    index, count = {}, {}
    for conv in dataset:
        for l in conv:
            for t in l:
                index[t] = 1
                if t not in count:
                    count[t] = 0
                count[t] += 1
    for piece in ['<start>', '<end>', '<sep>']:
        index[piece] = 1

    for idx, k in enumerate(index):
        index[k] = idx + 1
    vocab_size = len(index) + 1
    print(vocab_size)

    with open('data/dict.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in index.items()}, f)
    with open('data/distribution.pkl', 'wb') as f:
        pickle.dump(count, f)

    start, sep, end = index['<start>'], index['<sep>'], index['<end>']
    maxlen = 256
    history = []
    for conv in dataset:
        for idx, l in enumerate(conv[:-1]):
            tmp = [start]
            for i in range(idx+1):
                tmp += [index[t] for t in conv[i]] + [sep]
            tmp = tmp[:-1] + [end]
            history.append(tmp)
    history = sequence.pad_sequences(history, maxlen=maxlen, padding='post', truncating='pre')
    print(history[1])
    with open('data/X_corpus.pkl', 'wb') as f:
        pickle.dump(history, f)

    maxlen = 256
    response = []
    for conv in dataset:
        for idx, l in enumerate(conv[1:]):
            response.append([start] + [index[t] for t in l] + [end])
    response = sequence.pad_sequences(response, maxlen=maxlen, padding='post', truncating='post')
    print(response[1])
    with open('data/Y_corpus.pkl', 'wb') as f:
        pickle.dump(response, f)