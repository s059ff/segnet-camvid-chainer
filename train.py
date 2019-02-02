import argparse
import datetime
import os

import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp
import numpy as np
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainer.training import Trainer
from chainer.training.extensions import (Evaluator, LogReport, PlotReport,
                                         PrintReport, ProgressBar,
                                         snapshot_object)
from chainer.training.updaters import StandardUpdater

from model import SegNet


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', **{
        'type': int,
        'default': 100,
        'help': 'The number of times of learning. default: 100'
    })
    parser.add_argument('-c', '--checkpoint_interval', **{
        'type': int,
        'default': 10,
        'help': 'The frequency of saving model. default: 10'
    })
    parser.add_argument('-b', '--batch_size', **{
        'type': int,
        'default': 1,
        'help': 'The number of samples contained per mini batch. default: 1'
    })
    parser.add_argument('-g', '--gpu', **{
        'type': int,
        'default': 0,
        'help': 'GPU number to use. Exceptionally, if -1, use CPU. default: 0'
    })
    parser.add_argument('-m', '--memory', **{
        'type': str,
        'default': 'cpu',
        'help': 'The memory storage to store training data, "cpu" or "gpu". default: cpu'
    })
    args = parser.parse_args()

    # Prepare training data.
    dataset = (cp if args.memory == 'gpu' and 0 <= args.gpu else np).load('./temp/dataset.npz')
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
    if 0 <= args.gpu:
        chainer.backends.cuda.get_device_from_id(0).use()
        model.to_gpu()

    # Prepare optimizer.
    optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # Prepare training.
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f'./temp/{timestamp}/'
    os.makedirs(directory)

    if 0 <= args.gpu and args.memory == 'cpu':
        def converter(batch, device=None, padding=None):
            return concat_examples([(cp.array(x), cp.array(y)) for x, y in batch], device, padding)
    else:
        converter = concat_examples

    train_iter = SerialIterator(train, args.batch_size, repeat=True, shuffle=True)
    test_iter = SerialIterator(test, args.batch_size, repeat=False, shuffle=False)
    updater = StandardUpdater(train_iter, optimizer, converter=converter)
    trainer = Trainer(updater, stop_trigger=(args.epochs, 'epoch'), out=directory)
    trainer.extend(Evaluator(test_iter, model, converter=converter),)
    trainer.extend(snapshot_object(target=model, filename='model-{.updater.epoch:04d}.npz'), trigger=(args.checkpoint_interval, 'epoch'))
    trainer.extend(LogReport(log_name='log'))
    trainer.extend(PlotReport(y_keys=['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(PlotReport(y_keys=['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(ProgressBar(update_interval=1))
    trainer.run()


if __name__ == '__main__':
    main()
