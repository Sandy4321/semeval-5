#!/usr/bin/env python
"""train.py

Usage:
    train.py [--n_mem=<dim>] [--dropout=<p>]

Options:
    --n_mem=<dim> Dimension of LSTM     [default: 300]
    --dropout=<p> strength of dropout   [default: 0.3]

"""

import theano
import logging
import numpy as np
import pprint
import json
import cPickle as pkl
from collections import OrderedDict

from fuel.transformers import *
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from blocks.model import Model
from blocks.extensions.monitoring import *
from blocks.algorithms import *
from blocks.main_loop import MainLoop
from blocks.extensions import *
from blocks.extensions.saveload import Checkpoint
from blocks.graph import *

from model import LSTMModel

np.random.seed(1)


def _transpose(data):
    return tuple(array.T for array in data)


def wrap_stream(split):
    parses, relations = split
    parse1, parse2 = zip(*parses)

    dataset = IndexableDataset(
        indexables=OrderedDict([('parse1', parse1), ('parse2', parse2), ('relation', np.array(relations))]),
        axis_labels={'parse1': ('batch', 'index'), 'parse2': ('batch', 'index'), 'relation': ('batch', 'index')}
    )

    stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(examples=dataset.num_examples, batch_size=128))
    stream = Padding(stream, mask_sources=('parse1', 'parse2'))
    stream = Mapping(stream, _transpose, )
    stream = Cast(stream, 'int32')
    return stream


if __name__ == '__main__':
    from docopt import docopt
    logging.basicConfig(level=logging.INFO)
    args = docopt(__doc__)

    n_mem = int(args['--n_mem'])
    dropout = float(args['--dropout'])
    n_epoch = 10

    logging.info(pprint.pformat(args))

    with open('dataset/vocab.pkl') as f:
        vocabs = pkl.load(f)
        word_vocab, rel_vocab = vocabs['word'], vocabs['rel']

    with open('dataset/trainXY.json') as f:
        train = json.load(f)
        train = wrap_stream(train)

    with open('dataset/testXY.json') as f:
        test = json.load(f)
        test = wrap_stream(test)

    model = LSTMModel(len(vocabs['word']), n_mem, len(vocabs['rel']))
    cg = ComputationGraph(model.cost)

    bricks_model = Model(model.cost)
    for brick in bricks_model.get_top_bricks():
        brick.initialize()
    model.lookup.W.set_value(vocabs['word'].get_embeddings().astype(theano.config.floatX))

    if dropout:
        pass
        # logger.info('Applying dropout of {}'.format(dropout))
        # lstm_dropout = [v for v in cg.intermediary_variables if v.name in {'W_cell_to_in', 'W_cell_to_out'}]
        # cg = apply_dropout(cg, lstm_dropout, drop_prob=dropout)

    # summary of what's going on
    parameters = bricks_model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, value.get_value().shape, value.get_value().mean()) for key, value
                     in parameters.items()],
                    width=120))

    algorithm = GradientDescent(cost=model.cost, parameters=cg.parameters, step_rule=Adam())

    # Fetch variables useful for debugging
    observables = [model.cost, model.acc, algorithm.total_step_norm, algorithm.total_gradient_norm ]
    for name, parameter in parameters.items():
        observables.append(parameter.norm(2).copy(name=name + "_norm"))
        observables.append(algorithm.gradients[parameter].norm(2).copy(name=name + "_grad_norm"))

    train_monitor = TrainingDataMonitoring(variables=observables, prefix="train", after_batch=True)
    test_monitor = DataStreamMonitoring(variables=[model.cost, model.acc], data_stream=test, prefix="test")

    average_monitoring = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)

    main_loop = MainLoop(
            model=bricks_model,
            data_stream=train,
            algorithm=algorithm,
            extensions=[
                Timing(),
                train_monitor,
                test_monitor,
                average_monitoring,
                FinishAfter(after_n_epochs=n_epoch),
                Checkpoint('model.save', every_n_batches=500,
                           save_separately=["model", "log"]),
                Printing(every_n_epochs=1)])
    main_loop.run()
