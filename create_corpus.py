import pickle
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from tensorflow.keras.preprocessing import sequence

if __name__ == '__main__':
    f_name = 'full_conv.txt'
    path = 'data/'+f_name
    texts = [l.strip() for l in open(path, 'r', encoding='utf8') if l!='\n']

    # tokenizer = Tokenizer(mmap=True)
    tokenizer = Tokenizer()
    char_filters = [UnicodeNormalizeCharFilter()]
    token_filters = [LowerCaseFilter(), ExtractAttributeFilter(att='surface')]
    analyzer = Analyzer(char_filters, tokenizer, token_filters)

    comp = []
    for l in texts:
        sn = [token for token in analyzer.analyze(l)]
        sn = ['<start>'] + sn + ['<end>']
        comp.append(sn)

    index, count = {}, {}
    for l in comp:
        for t in l:
            index[t] = 1
            if t not in count:
                count[t] = 0
            count[t] += 1
    for idx, k in enumerate(index):
        index[k] = idx + 1
    vocab_size = len(index) + 1
    print(vocab_size)
    with open('new_data/dict.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in index.items()}, f)
    with open('new_data/distribution.pkl', 'wb') as f:
        pickle.dump(count, f)

    del analyzer
    del token_filters
    del char_filters
    del tokenizer

    maxlen = 32
    corpus = [[index[t] for t in l] for l in comp]
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen, padding='post', truncating='post')
    print(corpus[0])
    with open('new_data/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)