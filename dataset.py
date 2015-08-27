import os
from nltk import word_tokenize
from collections import Counter
import numpy as np
import cPickle as pkl


mydir = os.path.abspath(os.path.dirname(__file__))

trainfile = os.path.join(mydir, 'dataset', 'train.txt')
devfile = os.path.join(mydir, 'dataset', 'dev.txt')
testfile = os.path.join(mydir, 'dataset', 'test.txt')

class Vocab(object):

    def __init__(self, unk=''):
        self.word2index = {}
        self.index2word = []
        self.counts = Counter()
        self.unk = unk

        if self.unk:
            self.add(self.unk)

    def clear_counts(self):
        self.counts = Counter()

    def __repr__(self):
        return str(self.word2index)

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, word):
        if self.unk:
            return self.word2index.get(word, self.word2index[self.unk])
        else:
            return self.word2index[word]

    def __contains__(self, word):
        return word in self.word2index

    def add(self, word, count=1):
        if word not in self.word2index:
            self.word2index[word] = len(self)
            self.index2word.append(word)
        self.counts[word] += count
        return self.word2index[word]

    def get(self, word, add=False):
        return self.add(word) if add else self[word]

    def sent2index(self, sent, add=False):
        if add:
            return [self.add(w) for w in sent]
        else:
            return [self[w] for w in sent]

    def index2sent(self, indices):
        return [self.index2word[i] for i in indices]

    def prune_rares(self, cutoff=2):
        v = self.__class__(unk=self.unk)
        for w in self.index2word:
            if self.counts[w] > cutoff or w == self.unk:
                v.add(w, count=self.counts[w])
        return v


class Senna(Vocab):

    def __init__(self, root):
        super(Senna, self).__init__("UNKNOWN")
        self.root = root

    def load_words(self):
        # load the text file of words
        with open(os.path.join(self.root, 'words.lst')) as f:
            words = [l.strip("\n") for l in f]

        # load the words and clear the counts
        self.sent2index(words, add=True)
        self.clear_counts()
        return self

    def load_embeddings(self):
        with open(os.path.join(self.root, 'words.lst')) as f:
            words = [l.strip("\n") for l in f]
        embs = np.loadtxt(os.path.join(self.root, 'embeddings.txt'))
        word2emb = dict(zip(words, embs))
        E = np.random.uniform(-0.1, 0.1, size=(len(self), 50))
        for i, word in enumerate(self.index2word):
            if word in word2emb:
                E[i] = word2emb[word]
        return E

    def prune_rares(self, cutoff=2):
        v = self.__class__(root=self.root)
        for w in self.index2word:
            if self.counts[w] > cutoff or w == self.unk:
                v.add(w, count=self.counts[w])
        return v


def numericalize(fname, vocabs, train=False):
    cache, X, Y = [], [], []
    word_vocab, rel_vocab = vocabs

    def process_cache():
        sentence = cache[0].strip("\n\r").lower()
        sentence = sentence.replace('<e1>', 'E1BEGIN ').replace('<e2>', 'E2BEGIN ')
        sentence = sentence.replace('</e1>', ' E1END').replace('</e2>', ' E2END')
        label = cache[1].strip("\n\r")
        sentence = word_tokenize(sentence.split("\t")[1].strip('"'))
        tokens = word_vocab.sent2index(sentence, add=train)
        X.append(np.array(tokens, dtype='int32'))
        Y.append(rel_vocab.add(label))

    with open(fname) as f:
        for line in f:
            if line.strip("\n\r") == "":
                process_cache()
                cache = []
            else:
                cache.append(line)
    return X, np.array(Y, dtype='int32')

if __name__ == '__main__':
    print 'loading pretrained vocabulary'
    word_vocab, rel_vocab = vocabs = Senna('dataset/senna'), Vocab()

    print 'numericalizing data'
    numericalize(trainfile, vocabs, True)
    old_num_words = len(word_vocab)
    word_vocab = word_vocab.prune_rares(cutoff=1)
    vocabs = word_vocab, rel_vocab
    print 'pruned rare words: from %s to %s' % (old_num_words, len(word_vocab))
    Xtrain, Ytrain = numericalize(trainfile, vocabs, False)
    Xdev, Ydev = numericalize(devfile, vocabs)
    Xtest, Ytest = numericalize(testfile, vocabs)

    print 'train', len(Xtrain)
    print 'dev', len(Xdev)
    print 'test', len(Xtest)
    print 'word', len(word_vocab)
    print 'rel', len(rel_vocab)

    print 'converting data format to fuel'

    from fuel.datasets import IndexableDataset
    from collections import OrderedDict

    train = IndexableDataset(
        indexables=OrderedDict([('word_indices', Xtrain), ('relation', Ytrain)]),
        axis_labels={'word_indices': ('batch', 'index'), 'relation': ('batch', 'index')}
    )

    dev = IndexableDataset(
        indexables=OrderedDict([('word_indices', Xdev), ('relation', Ydev)]),
        axis_labels={'word_indices': ('batch', 'index'), 'relation': ('batch', 'index')}
    )

    test = IndexableDataset(
        indexables=OrderedDict([('word_indices', Xtest), ('relation', Ytest)]),
        axis_labels={'word_indices': ('batch', 'index'), 'relation': ('batch', 'index')}
    )


    print 'saving'
    with open('train.pkl', 'wb') as f:
        pkl.dump(train, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('dev.pkl', 'wb') as f:
        pkl.dump(dev, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('test.pkl', 'wb') as f:
        pkl.dump(test, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('vocabs.pkl', 'wb') as f:
        pkl.dump({'word': word_vocab, 'rel': rel_vocab}, f, protocol=pkl.HIGHEST_PROTOCOL)

    print 'testing iteration'
    from fuel.schemes import *
    from fuel.streams import DataStream
    from fuel.transformers import *

    train_stream = DataStream(dataset=train, iteration_scheme=ShuffledScheme(examples=train.num_examples, batch_size=128))
    train_stream = Padding(train_stream, mask_sources=('word_indices',))
    train_stream = Mapping(train_stream, lambda data: tuple(array.T for array in data))

    for batch in train_stream.get_epoch_iterator():
        Xbatch, Xmask, Ybatch = batch
        print 'batch shape', Xbatch.shape, Xmask.shape, Ybatch.shape

    print 'done'