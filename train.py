from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Linear
from blocks.extensions import *
from blocks.graph import ComputationGraph
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
import numpy as np
import cPickle as pkl
from fuel.transformers import *

from theano import tensor as T, function
from theano.tensor.extra_ops import to_one_hot

from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks import bricks
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks import initialization
from blocks.monitoring import aggregation

from blocks.extensions.monitoring import *


if __name__ == '__main__':

    state_size = 500
    emb_size = 50
    num_epoch = 30
    learning_rate = 1.0
    batch_size = 128

    from dataset import Vocab, Senna
    with open('vocabs.pkl') as f:
        vocabs = pkl.load(f)
        word_vocab, rel_vocab = vocabs['word'], vocabs['rel']

    with open('train.pkl') as f:
        train = pkl.load(f)

    np.random.seed(1)

    # dataset
    train_stream = DataStream(dataset=train, iteration_scheme=ShuffledScheme(examples=train.num_examples, batch_size=128))
    train_stream = Padding(train_stream, mask_sources=('word_indices',))
    train_stream = Mapping(train_stream, lambda data: tuple(array.T for array in data))
    train_stream = Cast(train_stream, 'int32')

    # model
    def build(state_size, vocab_size, output_size, emb_size=50):
        x = T.imatrix('word_indices')
        mask = T.imatrix('word_indices_mask')
        y = T.ivector('relation')

        lookup = LookupTable(
            vocab_size, emb_size,
            name='lookup',
            weights_init=initialization.Uniform(0., width=0.01),
        )
        emb = lookup.apply(x)

        x_to_h = Linear(
            emb_size, state_size*4,
            name='x_to_h',
            weights_init=initialization.IsotropicGaussian(0.01),
            biases_init=initialization.Constant(0.)
        )
        x_transform = x_to_h.apply(emb)

        lstm = LSTM(
            state_size,
            weights_init=initialization.IsotropicGaussian(0.01),
            biases_init=initialization.Constant(0.),
            name='lstm'
        )
        h, c = lstm.apply(x_transform, mask=T.cast(mask, 'float32'))

        h_to_o = Linear(
            state_size, output_size,
            name='output',
            weights_init=initialization.IsotropicGaussian(0.01),
            biases_init=initialization.Constant(0.),
        )

        score = h_to_o.apply(h[-1])
        softmax = bricks.Softmax()
        y_prob = softmax.apply(score)

        cost = CategoricalCrossEntropy().apply(to_one_hot(y, output_size), y_prob)

        for brick in [lookup, x_to_h, lstm, h_to_o]:
            brick.initialize()

        return cost

    cost = build(state_size, len(word_vocab), len(rel_vocab))

    algorithm = GradientDescent(
        cost=cost,
        parameters=ComputationGraph(cost).parameters,
        step_rule=Adam(),
    )

    train_monitor = TrainingDataMonitoring(
        variables=[
            cost,
            aggregation.mean(algorithm.total_gradient_norm),
            aggregation.mean(algorithm.total_step_norm)
        ],
        prefix="train", after_epoch=True)

    main_loop = MainLoop(
        algorithm=algorithm,
        data_stream=train_stream,
        model=Model(cost),
        extensions=[
            train_monitor,
            FinishAfter(after_n_epochs=10),
            Printing(),
            ProgressBar(),
        ])

    main_loop.run()
