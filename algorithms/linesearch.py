from scipy.optimize import line_search
import numpy as np

from core import Algorithm
from sgd import SGD
import util


class LineSearchGD(Algorithm):

    name = "GD with Backtracking Line Search"

    def _custom_init(self, amax=1, **kargs):
        self.amax = amax

    @property
    def learning_rate(self):
        """Learning rate"""
        alpha,_,_,_,_,_ = line_search(f=self._func.eval, 
                                      myfprime=self._func.gradient, 
                                      xk=self.theta, 
                                      pk=-self._current_gradient,
                                      amax=self.amax)
        if alpha is None:
            alpha = self._learning_rate(self._num_iter)
        return alpha

    def _update_theta(self):
        self.theta -= self.learning_rate * self._current_gradient


class MyLineSearchGD(Algorithm):

    name = "GD with Backtracking Line Search"

    def _custom_init(self, amax=1.0, c=0.2, beta=0.9, **kwargs):
        self.amax = float(amax)
        self.c = float(c)
        self.beta = float(beta)

    @property
    def learning_rate(self):
        """Learning rate"""
        alpha = self.amax
        gradient = self._func.gradient(self.theta)
        grad_norm = np.sum(gradient**2)
        for i in range(20):
            fval_old = self._func.eval(self.theta)
            fval_new = self._func.eval(self.theta - alpha * gradient)
            if fval_new >= fval_old - alpha * self.c * grad_norm:
                alpha *= self.beta
            else:
                break
        return alpha

    def _update_theta(self):
        self.theta -= self.learning_rate * self._current_gradient


class LineSearch_SGD(SGD):

    name = "SGD with Backtracking Line Search"

    c = 0.25
    beta = 0.9

    def _custom_init(self, amax=1.0, c=0.2, beta=0.9, **kargs):
        super(LineSearch_SGD, self)._custom_init()
        self.amax = float(amax)
        self.c = float(c)
        self.beta = float(beta)

    @property
    def learning_rate(self):
        """Learning rate"""
        alpha = self.amax
        gradient = self._func.stoch_gradient(self.theta, util._ord[self._idx])
        grad_norm = np.sum(gradient**2)
        for i in range(20):
            fval_old = self._func.stoch_eval(self.theta, util._ord[self._idx])
            fval_new = self._func.stoch_eval(self.theta - alpha * gradient, util._ord[self._idx])
            if fval_new >= fval_old - alpha * self.c * grad_norm:
                alpha *= self.beta
            else:
                break
        return alpha

    def _update_theta(self):
        self.theta -= self.learning_rate * self._current_gradient
