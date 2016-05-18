import numpy as np

from core import Algorithm
from sgd import SGD


class AdaGrad(Algorithm):
    
    name = "AdaGrad"

    def _custom_init(self, **kwargs):
        self.smoothing = 1e-8
        self._acc_grad = np.zeros(2)
        self.adjusted_grad = np.zeros(2)

    def _update_theta(self):
        self._acc_grad += self._current_gradient**2
        self.adjusted_grad = self._current_gradient / (np.sqrt(self._acc_grad) + self.smoothing) 
        self.theta -= self.learning_rate * self.adjusted_grad


class AdaGrad_SGD(SGD):
    
    name = "SGD with AdaGrad"

    def _custom_init(self, **kwargs):
        super(AdaGrad_SGD, self)._custom_init(**kwargs)
        self.smoothing = 1e-8
        self._acc_grad = np.zeros(2)
        self.adjusted_grad = np.zeros(2)

    def _update_theta(self):
        self._acc_grad += self._current_gradient**2
        self.adjusted_grad = self._current_gradient / (np.sqrt(self._acc_grad) + self.smoothing) 
        self.theta -= self.learning_rate * self.adjusted_grad

