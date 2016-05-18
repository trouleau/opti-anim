import numpy as np

from core import Algorithm
from sgd import SGD
from util import ConvergenceError, cast_as_func


class Momentum(Algorithm):

    name = "Momentum"

    def _custom_init(self, momentum, **kwargs):
        self._momentum = cast_as_func(momentum)
        self.v = np.zeros(2)

    @property
    def momentum(self):
        return self._momentum(self._num_iter)
    
    def _update_theta(self):
        self.v = self.momentum * self.v + \
                 self.learning_rate * self._current_gradient
        self.theta -= self.v


class Momentum_SGD(SGD):

    name = "Momentum SGD"

    def _custom_init(self, momentum, **kwargs):
        super(Momentum_SGD, self)._custom_init()
        self._momentum = cast_as_func(momentum)
        self.v = np.zeros(2)

    @property
    def momentum(self):
        return self._momentum(self._num_iter)
    
    def _update_theta(self):
        self.v = self.momentum * self.v + self.learning_rate * self._current_gradient
        self.theta -= self.v
