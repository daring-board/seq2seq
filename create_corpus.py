import sys
import pickle
import sentencepiece as spm
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from tensorflow.keras.preprocessing import sequence

if __name__ == '__main__':
    f_name = 'full_conv.txt'
    path = 'data/'+f_name
    texts = [l.strip() for l in open(path, 'r', encoding='utf8') if l!='\n']

    flag = sys.argv[1]
    print(flag)
    if flag == 'janome':
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
        del analyzer
        del token_filters
        del char_filters
        del tokenizer
    else:
        mpath = 'models/sentensepice'
        template = '--input=%s --model_prefix=%s --vocab_size=8000'
        spm.SentencePieceTrainer.train(template%(path, mpath))
        sp = spm.SentencePieceProcessor()
        sp.load(mpath+'.model')

        comp, anno = [], []
        for idx, l in enumerate(texts[:-5]):
            sn1 = sp.encode_as_pieces(l)
            sn2 = sp.encode_as_pieces(texts[idx+1])
            sn3 = sp.encode_as_pieces(texts[idx+2])
            sn4 = sp.encode_as_pieces(texts[idx+3])
            sn5 = sp.encode_as_pieces(texts[idx+4])
            sn = ['<start>'] + sn1 + ['<sep>']
            sn += sn2 + ['<sep>'] + sn3 + ['<sep>']+ sn4 + ['<end>']
            comp.append(sn)
            sn = ['<start>'] + sn5 + ['<end>']
            anno.append(sn)

    index, count = {}, {}
    for l in comp:
        for t in l:
            index[t] = 1
            if t not in count:
                count[t] = 0
            count[t] += 1
    for idx, k in enumerate(index):
        index[k] = idx+1
    vocab_size = len(index) + 1
    print(vocab_size)
    with open('data/dict.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in index.items()}, f)
    with open('data/distribution.pkl', 'wb') as f:
        pickle.dump(count, f)

    maxlen = 64
    corpus = [[index[t] for t in l] for l in comp]
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen, padding='post', truncating='post', value=0)
    print(corpus[0])
    with open('data/corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
    
    maxlen = 32
    corpus = [[index[t] for t in l] for l in anno]
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen, padding='post', truncating='post', value=0)
    print(corpus[0])
    with open('data/response.pkl', 'wb') as f:
        pickle.dump(corpus, f)