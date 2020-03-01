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
    l = glob.glob('data/tweet/*.txt')
    texts = []
    for path in l:
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
    del analyzer
    del token_filters
    del char_filters
    del tokenizer

    mpath = 'models/sentensepice'
    template = '--input=%s --model_prefix=%s --vocab_size=2000'
    spm.SentencePieceTrainer.train(template%(path, mpath))
    sp = spm.SentencePieceProcessor()
    sp.load(mpath+'.model')

    comp = []
    for l in texts:
        sn = sp.encode_as_pieces(l)
        comp.append(sn)

    index, count = {}, {}
    for l in comp:
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
    maxlen = 64
    corpus = [[start]+[index[t] for t in l]+[end] for l in comp[0::2]][:-1]
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen, padding='post', truncating='pre')
    print(corpus[0])
    with open('data/X_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)

    maxlen = 64
    corpus = [[start]+[index[t] for t in l]+[end] for l in comp[1::2]]
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen, padding='post', truncating='post')
    print(corpus[0])
    with open('data/Y_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)