import autograd.numpy as np
import cPickle as pkl
from pystacks.param import ParamServer
from pystacks.layers.recurrent import LSTMMemoryLayer
from pystacks.layers.embedding import LookupTable
from pystacks.regularizers import Dropout
from pystacks.layers.core import Dense
from pystacks.layers.activations import LogSoftmax
from pystacks.initialization import Hardcode
from pystacks.optimizers import *
from pystacks.utils.logging import Progbar
from pystacks.utils.math import make_batches_by_len
from pystacks.grad_transformer import *
from pystacks.utils.text import Vocab, Senna
from pystacks.utils.math import one_hot, make_batches_by_len

from pprint import pprint
from time import time
from autograd import value_and_grad
from autograd.util import quick_grad_check

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

if __name__ == '__main__':
    with open('numericalized.pkl') as f:
        train, dev, test, word_vocab, rel_vocab = pkl.load(f)

    state_size = 64
    emb_size = 50
    num_epoch = 10
    learning_rate = 1.0
    batch_size = 128
    np.random.seed(1)

    server = ParamServer()

    lookup_layer = LookupTable(len(word_vocab), emb_size)
    lookup_layer.E.init = Hardcode(word_vocab.load_embeddings())
    lookup_layer.register_params(server)
    lstm_layer = LSTMMemoryLayer(emb_size, state_size, dropout=Dropout(0.1)).register_params(server)

    rel_output_layer = Dense(state_size, len(rel_vocab)).register_params(server)
    rel_softmax_layer = LogSoftmax().register_params(server)

    server.finalize()

    def pred_fun(weights, x, train=False):
        server.param_vector = weights
        ht, ct = None, None
        for t in xrange(x.shape[1]):
            emb = lookup_layer.forward(x[:, t], train=train)
            ht, ct = lstm_layer.forward(emb, ht, ct, train=train)
            pt_rel = rel_output_layer.forward(ht)
        return rel_softmax_layer.forward(pt_rel)

    def loss_fun(weights, x, targets, train=False):
        logprobs_rel = pred_fun(weights, x, train=train)
        loss_sum = - np.sum(logprobs_rel * targets)
        return loss_sum / float(x.shape[0])

    def score(weights, inputs, targets, train=False):
        targs = np.argmax(targets, axis=-1)
        logprobs_rel = pred_fun(weights, inputs, train)
        preds = np.argmax(logprobs_rel, axis=-1)
        acc = accuracy_score(targs, preds)
        return targs, preds, acc

    # Build gradient of loss function using autograd.
    loss_and_grad = value_and_grad(loss_fun)

    # Check the gradients numerically, just to be safe
    xsmall, ysmall = np.array(train[0][0]).reshape(1, -1), one_hot(np.array(train[0][1]).reshape(1, -1), len(rel_vocab))
    quick_grad_check(loss_fun, server.param_vector, (xsmall, ysmall))

    print("Training LSTM...")
    optimizer = Adam()
    start = time()
    for epoch in xrange(1, num_epoch+1):
        epoch_loss = 0.
        batches = make_batches_by_len([len(x) for x in train[0]], batch_size)
        print 'epoch', epoch
        bar = Progbar(len(train[0]), 'train', track=['loss', 'acc'])
        for batch in batches:
            x, y = np.array([train[0][i] for i in batch], dtype='int32'), one_hot(np.array([train[1][i] for i in batch], dtype='int32'), len(rel_vocab))
            loss, dparams = loss_and_grad(server.param_vector, x, y, train=True)
            epoch_loss += loss
            server.update_params(optimizer, dparams, learning_rate)
            targs_, preds_, acc = score(server.param_vector, x, y, train=False)
            bar.update(len(y), new_values={'loss':loss, 'acc':acc})
        bar.finish()

        batches = make_batches_by_len([len(x) for x in dev[0]], batch_size)
        bar = Progbar(len(dev[0]), 'eval', track=['acc'])
        for batch in batches:
            x, y = np.array([dev[0][i] for i in batch], dtype='int32'), one_hot(np.array([dev[1][i] for i in batch], dtype='int32'), len(rel_vocab))
            targs_, preds_, acc = score(server.param_vector, x, y, train=False)
            bar.update(len(y), new_values={'acc':acc})
        bar.finish()

        print("epoch %s train loss %s in %s" % (epoch, epoch_loss, time() - start))
        start = time()