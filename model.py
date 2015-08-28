__author__ = 'victor'
import theano.tensor as T
from theano import function
from theano.tensor.extra_ops import to_one_hot
from blocks.bricks import *
from blocks.model import *
from blocks.bricks.lookup import *
from blocks.initialization import *
from blocks.bricks.recurrent import *
from blocks.bricks.cost import *

class BasicModel(object):

    def __init__(self, state_size, vocab_size, output_size, emb_size=50):
        x = T.imatrix('word_indices')
        mask = T.imatrix('word_indices_mask')
        y = T.ivector('relation')

        lookup = LookupTable(
            vocab_size, emb_size,
            name='lookup',
            weights_init=Uniform(0., width=0.01),
        )
        emb = lookup.apply(x)

        x_to_h = Linear(
            emb_size, state_size*4,
            name='x_to_h',
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.)
        )
        x_transform = x_to_h.apply(emb)

        lstm = LSTM(
            state_size,
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.),
            name='lstm'
        )
        h, c = lstm.apply(x_transform, mask=T.cast(mask, 'float32'))

        h_to_o = Linear(
            state_size, output_size,
            name='output',
            weights_init=IsotropicGaussian(0.1),
            biases_init=Constant(0.),
        )

        score = h_to_o.apply(h[-1])
        softmax = Softmax()
        epsilon = 1e-7
        y_prob = softmax.apply(T.clip(score, epsilon, 1-epsilon))

        self.cost = CategoricalCrossEntropy().apply(to_one_hot(y, output_size), y_prob)
        self.cost.name = 'CrossEntropyCost'

        eq = T.eq(y_prob.argmax(axis=-1), y)
        self.acc = T.cast(eq, 'float32').mean()
        self.acc.name = 'Accuracy'

        self.x, self.y = x, y
        self.lookup, self.x_to_h, self.lstm, self.h_to_o = lookup, x_to_h, lstm, h_to_o
        # self.debug_emb = function([x], emb)
        # self.debug_x_transform = function([x], x_transform)
        # self.debug_h = function([x, mask], h)
        # self.debug_score = function([x, mask], [score, y_prob])
        # self.debug_cost = function([x, mask, y], self.cost)
