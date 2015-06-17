from text.vocab import Vocab
import os
import numpy as np

mydir = os.path.abspath(os.path.dirname(__file__))

trainfile = os.path.join(mydir, 'dataset', 'train.txt')
devfile = os.path.join(mydir, 'dataset', 'dev.txt')
testfile = os.path.join(mydir, 'dataset', 'test.txt')

def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes), dtype='float32')
    Y[np.arange(y.size), y] = 1.
    return Y

class Example(object):

    def __init__(self, lines):
        sent, self.label, self.comment = lines
        id, sent = sent.replace('.', '').replace(',', '').split("\t")
        self.id = int(id)
        self.words = sent.lower().split()

class Dataset(object):

    def __init__(self):
        print 'constructing training set'
        self.train = Split(trainfile)
        self.word_vocab, self.label_vocab = self.train.parse(add=True)
        print 'pruning rare words'
        self.word_vocab = self.word_vocab.prune_rares(cutoff=2)
        self.train = Split(trainfile, vocabs=(self.word_vocab, self.label_vocab))
        self.word_vocab, self.label_vocab = self.train.parse(add=False)
        print 'constructing dev set'
        self.dev = Split(devfile, vocabs=(self.word_vocab, self.label_vocab))
        self.dev.parse(add=False)
        print 'constructing test set'
        self.test = Split(testfile, vocabs=(self.word_vocab, self.label_vocab))
        self.test.parse(add=False)

class Split(object):

    def __init__(self, fname, vocabs=(None, None)):
        self.fname = fname
        self.word_vocab, self.label_vocab = vocabs
        if not self.word_vocab: self.word_vocab = Vocab(unk=True)
        if not self.label_vocab: self.label_vocab = Vocab(unk=False)
        self.X, self.Y, self.lengths = [], [], {}

    def parse(self, add=False):
        self.X, self.Y = [], []
        # parse train
        with open(self.fname) as f:
            cache = []
            for line in f:
                line = line.strip("\n\r")
                if not line:
                    ex = Example(cache)
                    self.X.append(self.word_vocab.sent2index(ex.words, add))
                    self.Y.append(self.label_vocab.add(ex.label))
                    cache = []
                else:
                    cache.append(line)
        self.Y = np.array(self.Y)

        for i, x in enumerate(self.X):
            self.X[i] = np.array(x)
            x = self.X[i]
            if x.size not in self.lengths:
                self.lengths[x.size] = []
            self.lengths[x.size].append(i)

        return self.word_vocab, self.label_vocab

    def __iter__(self):
        len_keys = self.lengths.keys()
        lengths = [len_keys[i] for i in np.random.permutation(len(self.lengths))]
        for length in lengths:
            match_inds = self.lengths[length]
            X = [self.X[i] for i in match_inds]
            Y = self.Y[match_inds]
            yield np.array(X), one_hot(Y, len(self.label_vocab))

if __name__ == '__main__':
    d = Dataset()
    for x, y in d.train:
        print x.shape, y.shape