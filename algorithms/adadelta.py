import numpy as np

from core import Algorithm
from sgd import SGD
from util import ConvergenceError, cast_as_func


class AdaDelta(Algorithm):

    name = "AdaDelta"

    def _custom_init(self, momentum, **kwargs):
        self._momentum = cast_as_func(momentum)
        self.smoothing = 1e-8
        self._acc_grad = np.zeros(2)
        self._acc_diff = np.zeros(2)
        self.adjusted_grad = np.zeros(2)

    def rms(self, acc):
        return np.sqrt(acc + self.smoothing)

    @property
    def momentum(self):
        return self._momentum(self._num_iter)

    def _update_theta(self):
        self._acc_grad = self.momentum * self._acc_grad + (1-self.momentum) * (self._current_gradient**2) 

        theta_diff = -self._current_gradient * self.learning_rate * self.rms(self._acc_diff)/self.rms(self._acc_grad)
        self.theta += self.learning_rate * theta_diff

        self._acc_diff = self.momentum * self._acc_diff + (1-self.momentum) * (theta_diff**2) 
        

class AdaDelta_SGD(SGD):

    name = "SGD with AdaDelta"

    def _custom_init(self, momentum, **kwargs):
        super(AdaDelta_SGD, self)._custom_init(**kwargs)
        self._momentum = cast_as_func(momentum)
        self.smoothing = 1e-8
        self._acc_grad = np.zeros(2)
        self._acc_diff = np.zeros(2)
        self.adjusted_grad = np.zeros(2)

    def rms(self, acc):
        return np.sqrt(acc + self.smoothing)

    @property
    def momentum(self):
        return self._momentum(self._num_iter)

    def _update_theta(self):
        self._acc_grad = self.momentum * self._acc_grad + (1-self.momentum) * (self._current_gradient**2) 
        theta_diff = -self._current_gradient * self.rms(self._acc_diff)/self.rms(self._acc_grad)
        self._acc_diff = self.momentum * self._acc_diff + (1-self.momentum) * (theta_diff**2) 
        self.theta += self.learning_rate * theta_diff
