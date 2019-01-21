import argparse
import datetime
import os

import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainer.training import Trainer
from chainer.training.extensions import Evaluator, LogReport, PlotReport, PrintReport, ProgressBar, snapshot_object
from chainer.training.updaters import StandardUpdater

from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    kwargs = {
        'type': int,
        'default': 100,
        'help': 'The number of times of learning. default: 100'
    }
    parser.add_argument('-e', '--epochs', **kwargs)
    kwargs = {
        'type': int,
        'default': 10,
        'help': 'The frequency of saving model. default: 10'
    }
    parser.add_argument('-c', '--checkpoint_interval', **kwargs)
    kwargs = {
        'type': int,
        'default': 1,
        'help': 'The number of samples contained per mini batch. default: 1'
    }
    parser.add_argument('-b', '--batch_size', **kwargs)
    args = parser.parse_args()

    # Prepare training data.
    dataset = cp.load('./temp/dataset.npz')
    train_x = dataset['train_x']
    train_y = dataset['train_y']
    test_x = dataset['test_x']
    test_y = dataset['test_y']

    train = [(x, y) for x, y in zip(train_x, train_y)]
    test = [(x, y) for x, y in zip(test_x, test_y)]

    # Prepare model.
    predictor = SegNet()
    model = L.Classifier(
        predictor,
        lossfun=F.sigmoid_cross_entropy,
        accfun=lambda y, t: F.binary_accuracy(y - 0.5, t))
    chainer.backends.cuda.get_device_from_id(0).use()
    model.to_gpu()

    # Prepare optimizer.
    optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # Prepare training.
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f'./temp/{timestamp}/'
    os.makedirs(directory)

    def converter(batch, device=None, padding=None):
        return concat_examples([(cp.array(x), cp.array(y)) for x, y in batch], device, padding)

    train_iter = SerialIterator(train, args.batch_size)
    test_iter = SerialIterator(test, args.batch_size, repeat=False, shuffle=False)
    updater = StandardUpdater(train_iter, optimizer, converter=converter)
    extensions = [
        Evaluator(test_iter, model, converter=converter),
        snapshot_object(target=model,
                        filename='model-{.updater.epoch:04d}.npz'),
        LogReport(log_name='log'),
        PlotReport(y_keys=['main/loss', 'validation/main/loss'],
                   x_key='epoch',
                   file_name='loss.png'),
        PlotReport(y_keys=['main/accuracy', 'validation/main/accuracy'],
                   x_key='epoch',
                   file_name='accuracy.png'),
        PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
        ProgressBar(update_interval=1)
    ]
    trainer = Trainer(updater, stop_trigger=(args.epochs, 'epoch'), out=directory, extensions=extensions)
    trainer.run()


if __name__ == '__main__':
    main()
