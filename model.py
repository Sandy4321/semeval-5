__author__ = 'victor'
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.extra_ops import to_one_hot
from blocks.initialization import *
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks import *
from blocks.bricks.cost import CategoricalCrossEntropy


class LSTMModel(object):

    def __init__(self, n_vocab, n_mem, n_class, n_emb=50):
        x1 = T.imatrix('parse1')
        x2 = T.imatrix('parse2')
        mask1 = T.imatrix('parse1_mask')
        mask2 = T.imatrix('parse2_mask')
        y = T.ivector('relation')

        lookup = LookupTable(
            n_vocab, n_emb,
            name='lookup',
            weights_init=Uniform(0., width=0.01),
        )
        emb1 = lookup.apply(x1)
        emb2 = lookup.apply(x2)

        trans1 = Linear(
            n_emb, n_mem*4,
            name='emb_to_h1',
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.)
        ).apply(emb1)
        trans2 = Linear(
            n_emb, n_mem*4,
            name='emb_to_h2',
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.)
        ).apply(emb2)

        h1, c1 = LSTM(
            n_mem,
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.),
            name='lstm1'
        ).apply(trans1, mask=T.cast(mask1, theano.config.floatX))

        h2, c2 = LSTM(
            n_mem,
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.),
            name='lstm2'
        ).apply(trans2, mask=T.cast(mask2, theano.config.floatX))

        h_concat = T.concatenate([h1[-1], h2[-1]], axis=-1)

        # debugging
        self.x1, self.x2, self.mask1, self.mask2 = x1, x2, mask1, mask2
        self.trans1, self.trans2 = trans1, trans2
        self.emb1, self.emb2, self.h1, self.h2, self.h_concat = emb1, emb2, h1, h2, h_concat

        score = Linear(
            2 * n_mem, n_class,
            name='output',
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.),
        ).apply(h_concat)

        epsilon = 1e-7
        self.y_prob = Softmax().apply(T.clip(score, epsilon, 1-epsilon))

        self.cost = CategoricalCrossEntropy().apply(to_one_hot(y, n_class), self.y_prob)
        self.cost.name = 'CrossEntropyCost'

        eq = T.eq(self.y_prob.argmax(axis=-1), y)
        self.acc = T.cast(eq, 'float32').mean()
        self.acc.name = 'Accuracy'

        self.lookup = lookup

if __name__ == '__main__':
    model = LSTMModel(10, 10, 5)
    print('debugging {}'.format(model))

    print(model.h1.eval({
        model.x1: np.array([[1, 2, 3], [3, 4, 3]], dtype='int32'),
        model.mask1: np.array([[1, 1, 0], [1, 1, 1]], dtype='int32'),
    }).shape)

    print(model.h2.eval({
        model.x2: np.array([[2, 3, 4], [1, 0, 2], [0, 2, 1]], dtype='int32'),
        model.mask2: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype='int32')
    }).shape)

    print(model.h_concat.eval({
        model.x1: np.array([[1, 2, 3], [3, 4, 3]], dtype='int32'),
        model.mask1: np.array([[1, 1, 0], [1, 1, 1]], dtype='int32'),
        model.x2: np.array([[2, 3, 4], [1, 0, 2], [0, 2, 1]], dtype='int32'),
        model.mask2: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype='int32')
    }).shape)

