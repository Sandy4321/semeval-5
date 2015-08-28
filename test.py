from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph

__author__ = 'victor'
from model import BasicModel
from blocks.model import Model
import numpy as np


def generate_data(batch_size, num_batches, max_seq_length=20):
    x = []
    x_mask = []
    y = []
    for i in xrange(num_batches):
        seq_length = np.random.randint(1, max_seq_length)
        # x_batch has shape (T, B, F=1)
        x_batch = np.random.random_integers(0, 1, (seq_length, batch_size))
        # y_batch has shape (B, F=1)
        y_batch = x_batch.sum(axis=(0,)) % 2
        x.append(x_batch.astype('int32'))
        x_mask.append(np.ones_like(x_batch).astype('int32'))
        y.append(y_batch.astype('int32'))
    return {'word_indices': x, 'word_indices_mask': x_mask, 'relation': y}


if __name__ == '__main__':
    state_size = 10
    vocab_size = 20
    emb_size = 3
    output_size = 2
    batch_size = 10
    num_epochs = 5
    model = BasicModel(state_size, vocab_size, output_size, emb_size)

    model_ = Model(model.cost)
    for brick in model_.get_top_bricks():
        brick.initialize()

    import numpy as np
    x = np.array([[1, 2, 3], [0, 3, 2]], dtype='int32')
    mask = np.array([[1, 1, 1], [1, 1, 0]], dtype='int32')
    y = np.random.randint(0, 5, size=3).astype('int32')

    # print 'x'
    # print x.shape, x
    #
    # print 'emb'
    # got = model.debug_emb(x)
    # print got.shape, got
    #
    # print 'x_transform'
    # got = model.debug_x_transform(x)
    # print got.shape, got
    #
    # print 'mask'
    # print 'h'
    # got = model.debug_h(x, mask)
    # print got.shape, got
    #
    # got_score, got_prob = model.debug_score(x, mask)
    # print 'score'
    # print got_score.shape, got_score
    # print 'prob'
    # print got_prob.shape, got_prob
    #
    # print 'cost'
    # print model.debug_cost(x, mask, y)

    from fuel.datasets import IterableDataset
    from fuel.streams import DataStream

    dataset_train = IterableDataset(generate_data(batch_size, 900))
    dataset_test = IterableDataset(generate_data(batch_size, 100))

    stream_train = DataStream(dataset=dataset_train)
    stream_test = DataStream(dataset=dataset_test)

    cg = ComputationGraph(model.cost)

    from blocks.algorithms import *
    from blocks.extensions.monitoring import *
    from blocks.main_loop import MainLoop
    from blocks.extensions import *

    algorithm = GradientDescent(cost=model.cost, parameters=cg.parameters,
                                step_rule=Adam())
    test_monitor = DataStreamMonitoring(variables=[model.cost, model.acc],
                                        data_stream=stream_test, prefix="test")
    train_monitor = TrainingDataMonitoring(variables=[model.cost, model.acc], prefix="train",
                                           after_epoch=True)

    main_loop = MainLoop(algorithm, stream_train,
                         extensions=[test_monitor, train_monitor,
                                     FinishAfter(after_n_epochs=num_epochs),
                                     Printing(), ProgressBar()])
    main_loop.run()