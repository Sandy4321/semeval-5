from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
import numpy as np
import cPickle as pkl
import pprint
import math
from fuel.transformers import *

from blocks.model import Model
from blocks.extensions.monitoring import *

def _transpose(data):
    return tuple(array.T for array in data)


def wrap_stream(name):
    with open(name) as f:
        dataset = pkl.load(f)
    stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(examples=dataset.num_examples, batch_size=128))
    stream = Padding(stream, mask_sources=('word_indices',))
    stream = Mapping(stream, _transpose)
    stream = Cast(stream, 'int32')
    return stream


if __name__ == '__main__':

    state_size = 500
    emb_size = 50
    num_epochs = 3
    learning_rate = 1.0
    batch_size = 128

    from dataset import Vocab, Senna
    with open('vocabs.pkl') as f:
        vocabs = pkl.load(f)
        word_vocab, rel_vocab = vocabs['word'], vocabs['rel']

    np.random.seed(1)

    # dataset
    stream_train = wrap_stream('train.pkl')
    stream_test = wrap_stream('dev.pkl')

    from model import BasicModel

    model = BasicModel(state_size, len(word_vocab), len(rel_vocab))
    cg = ComputationGraph(model.cost)

    bricks_model = Model(model.cost)
    for brick in bricks_model.get_top_bricks():
        brick.initialize()

    # summary of what's going on
    parameters = bricks_model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, value.get_value().shape, value.get_value().mean()) for key, value
                     in parameters.items()],
                    width=120))

    from blocks.algorithms import *
    from blocks.extensions.monitoring import *
    from blocks.main_loop import MainLoop
    from blocks.extensions import *
    algorithm = GradientDescent(cost=model.cost, parameters=cg.parameters, step_rule=Adam())

    # Fetch variables useful for debugging
    observables = [model.cost, model.acc, algorithm.total_step_norm, algorithm.total_gradient_norm ]
    for name, parameter in parameters.items():
        observables.append(named_copy(parameter.norm(2), name + "_norm"))
        observables.append(named_copy(algorithm.gradients[parameter].norm(2), name + "_grad_norm"))

    train_monitor = TrainingDataMonitoring(variables=observables, prefix="train", after_batch=True)
    test_monitor = DataStreamMonitoring(variables=[model.cost, model.acc], data_stream=stream_test, prefix="test")

    average_monitoring = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)

    main_loop = MainLoop(
            model=bricks_model,
            data_stream=stream_train,
            algorithm=algorithm,
            extensions=[
                Timing(),
                train_monitor,
                test_monitor,
                average_monitoring,
                FinishAfter(after_n_epochs=num_epochs),
                Checkpoint('model.save', every_n_batches=500,
                           save_separately=["model", "log"]),
                Printing(every_n_epochs=1)])
    main_loop.run()
