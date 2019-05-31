# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - metrics.py
# md
# --------------------------------------------------------------------------------------------------------

import torch as th
from .callbacks import Callback
import numpy as np


class MetricContainer:
    def __init__(self, metrics=None, prefix=''):
        self.metrics = metrics or []
        self.prefix = 'metric.'  # prefix is added to the metric name in logs, so we know all metrics in callbacks

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __call__(self, y_pred, y_true, logs):
        for metric in self.metrics:
            logs[self.prefix + metric._name] = metric(y_pred, y_true)
        return logs


class Metric:

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')


class MetricCallback(Callback):
    """
    Callback for metrics
    """
    def __init__(self, container):  # Container is MetrixContainer
        super(MetricCallback, self).__init__()
        self.container = container

    def on_epoch_begin(self, epoch, logs):
        # print(logs)
        self.container.reset()

    def on_epoch_end(self, epoch, logs):
        y_pred, y_true = logs['y_pred'], logs['y_true']  # y_pred is of type [[a,b],[c,d], ...]
        self.container(y_pred, y_true, logs)  # calls MetricContainer.__call__()


class Accuracy(Metric):
    def __init__(self):
        # self.accuracy = 0
        self._name = 'acc'

    def reset(self):
        # self.accuracy = 0
        pass

    def __call__(self, y_pred, y_true):
        y_pred = np.argmax(y_pred.data, axis=1)
        accuracy = np.sum(y_pred == y_true) / len(y_pred)
        return accuracy


class PredictionEntropy(Metric):
    """
    Calculates the average entropy of y_pred. It assumes that y_pred are probabilities. Use softmax(y_pred) if needed.
    """

    def __init__(self):
        self._name = 'pred_entropy'

    def reset(self):
        pass

    def __call__(self, y_pred, y_true):
        pass
        p = [p_i for p_i in y_pred + 1e-16]  # add very small number to avoid log(0)
        entropy = -np.sum(p * np.log2(p)) / len(p)
        # Scale entropy to [0,1] for n_classes>2
        entropy = entropy / np.log2(len(p[0]))  # len(p[0]) = n_classes
        return entropy
