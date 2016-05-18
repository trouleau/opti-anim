from scipy.optimize import line_search, fmin_cg
import numpy as np

from core import Algorithm
from sgd import SGD
import util


class ConjugateGD(Algorithm):
    """Conjugate Gradient descent using Polak-Ribiere update (scipy)"""

    name = "Conjugate Gradient Descent (scipy)"

    def _custom_init(self, **kwargs):
        _,self._allvecs = fmin_cg(f=self._func.eval, 
                                         fprime=self._func.gradient, 
                                         x0=self.theta, retall=True, disp=0)

    def _update_theta(self):
        if self._num_iter < len(self._allvecs):
            self.theta =  self._allvecs[self._num_iter]

    def _has_converged(self):
        if self._num_iter == len(self._allvecs)-1:
            self.has_converged = True


class MyConjugateGD(Algorithm):
    """Conjugate Gradient descent using Polak-Ribiere update"""

    name = "Conjugate Gradient Descent"

    def _custom_init(self, amax=50.0, c1=1e-4, c2=0.9, **kwargs):
        self.amax = amax
        self.c1 = c1
        self.c2 = c2
        self.p = -self._func.gradient(self.theta)

    @property
    def learning_rate(self):
        """Learning rate"""
        alpha,_,_,_,_,_ = line_search(f=self._func.eval, 
                              myfprime=self._func.gradient, 
                              xk=self.theta, 
                              pk=self.p,
                              amax=self.amax,
                              c1=self.c1, c2=self.c2)
        if alpha is None:
            # If line search does not converge (expect troubles!)
            return self._learning_rate(self._num_iter)
        return alpha

    def _update_theta(self):
        gfk = self._current_gradient
        dotgfk = np.dot(gfk,gfk)
        lrate = self.learning_rate
        self.theta += lrate * self.p
        gfkp1 = self._func.gradient(self.theta)
        yk = gfkp1-gfk
        self.beta = max(0, np.dot(gfkp1, yk) / dotgfk)
        self.p = -gfkp1 + self.beta * self.p

        
class MyConjugate_SGD(SGD):
    """Stochastic Conjugate Gradient descent using Hestenes-Stiefel update"""

    name = "Conjugate SGD"
    c1 = 1e-4
    c2 = 0.2

    def _custom_init(self, amax=50.0, **kwargs):
        super(MyConjugate_SGD, self)._custom_init(**kwargs)
        self.amax = amax
        self.p = -self._func.stoch_gradient(self.theta, util._ord[self._idx])

    @property
    def learning_rate(self):
        """Learning rate"""
        alpha,_,_,_,_,_ = line_search(f=self.stoch_func_eval, 
                              myfprime=self.stoch_func_gradient, 
                              xk=self.theta, 
                              pk=self.p,
                              amax=self.amax,
                              c1=self.c1, c2=self.c2)
        if alpha is None:
            return self._learning_rate(self._num_iter)
        return alpha

    def _update_theta(self):
        old_gradient = self._func.stoch_gradient(self.theta, util._ord[self._idx])
        self.theta += self.learning_rate * self.p
        new_gradient = self._func.stoch_gradient(self.theta, util._ord[self._idx])
        self.beta = np.dot(new_gradient, new_gradient-old_gradient) * 1.0 / np.dot(new_gradient-old_gradient, self.p)
        self.p = -new_gradient + max(self.beta, 0) * self.p
      