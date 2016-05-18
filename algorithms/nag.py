import numpy as np

from core import Algorithm
from sgd import SGD
from momentum import Momentum
from util import ConvergenceError, cast_as_func
import util

class NAG(Algorithm):
    """Nesterov Accelerated Gradient"""
   
    name = "Nesterov Accelerated Gradient"

    def _custom_init(self, **kwargs):
        self.v = np.zeros(2)
        self._epsilon = self.learning_rate

    @property
    def epsilon(self):
        for i in range(10):
            next_espilon = self._epsilon * 2**-i
            yt = self.theta + self.mu * self.v
            grad_yt = self._func.gradient(yt)
            if ((self._func.eval(yt) - self._func.eval(yt - next_espilon * grad_yt)) >= (next_espilon * 0.5 * grad_yt**2)).any():
                self._epsilon = next_espilon
                return self._epsilon
        return self._epsilon

    @property
    def mu(self):
        return 1.0 - 3.0 / (self._num_iter + 5)

    def _update_theta(self):
        gradient = self._func.gradient(self.theta + self.mu * self.v)
        self.v = self.mu * self.v - self.epsilon * gradient
        self.theta += self.v


class NAG2(Momentum):
    """Nesterov Accelerated Gradient with arbitrary parameters"""
    
    name = "Approx. Nesterov Accelerated Gradient"

    def _update_theta(self):
        next_gradient = self._func.gradient(self.theta - self._momentum * self.v)
        self.v = self.momentum * self.v + self.learning_rate * next_gradient
        self.theta -= self.v


class NAG2_SGD(SGD):
    """Stochastic Nesterov Accelerated Gradient with arbitrary parameters"""

    name = "Nesterov Accelerated SGD"

    def _custom_init(self, momentum, **kwargs):
        super(NAG2_SGD, self)._custom_init()
        self._momentum = cast_as_func(momentum)
        self.v = np.zeros(2)

    @property
    def momentum(self):
        return self._momentum(self._num_iter)

    def _compute_gradient(self):
        self._current_gradient = self._func.stoch_gradient(self.theta + self.momentum * self.v, util._ord[self._idx])

    def _update_theta(self):
        self.v = self.momentum * self.v - self.learning_rate * self._current_gradient
        self.theta += self.v
