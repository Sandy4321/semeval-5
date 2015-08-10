from pystacks.utils.text import Vocab, Senna
import os
import autograd.numpy as np
from nltk import word_tokenize


mydir = os.path.abspath(os.path.dirname(__file__))

trainfile = os.path.join(mydir, 'dataset', 'train.txt')
devfile = os.path.join(mydir, 'dataset', 'dev.txt')
testfile = os.path.join(mydir, 'dataset', 'test.txt')

def numericalize(fname, vocabs, train=False):
    cache, X, Y = [], [], []
    word_vocab, rel_vocab = vocabs

    def process_cache():
        sentence = cache[0].strip("\n").lower()
        sentence = sentence.replace('<e1>', 'E1BEGIN ').replace('<e2>', 'E2BEGIN ').replace('</e1>', ' E1END').replace('</e2>', ' E2END')
        label = cache[1].strip("\n")
        sentence = sentence.split("\t")[1].strip('"').split()
        tokens = word_vocab.sent2index(sentence, add=train)
        X.append(tokens)
        Y.append(rel_vocab.add(label))

    with open(fname) as f:
        for line in f:
            if line.strip("\n\r") == "":
                process_cache()
                cache = []
            else:
                cache.append(line)
    return X, Y

if __name__ == '__main__':
    word_vocab, rel_vocab = vocabs = Senna('/Users/victor/Developer/pystacks/examples/data/senna'), Vocab()
    train = numericalize(trainfile, vocabs, True)
    dev = numericalize(devfile, vocabs)
    test = numericalize(testfile, vocabs)

    print 'train', len(train[0])
    print 'dev', len(dev[0])
    print 'test', len(test[0])
    print 'word', len(word_vocab)
    print 'rel', len(rel_vocab)

    import cPickle as pkl
    with open('numericalized.pkl', 'wb') as f:
        pkl.dump((train, dev, test, word_vocab, rel_vocab), f)

