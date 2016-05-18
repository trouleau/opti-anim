from functools import partial
import copy
import numpy as np

from core import Algorithm
from util import init_noise, init_ord
import util


class SGD(Algorithm):
    """Stochastic gradient descent""" 

    name = "SGD"

    def _custom_init(self, **kargs):
        self._idx = 0
        util.init_ord(self._func.n_samples)

    def _update_idx(self):
        if self._idx < len(util._ord)-1:
            self._idx += 1
        else:
            self._idx = 0

    def _update_stoch_func(self):
        self.stoch_func_eval = partial(self._func.stoch_eval, 
                                       i=util._ord[self._idx])
        self.stoch_func_gradient = partial(self._func.stoch_gradient, 
                                           i=util._ord[self._idx])

    def _compute_gradient(self):
        self._current_gradient = self.stoch_func_gradient(self.theta)

    def _update_theta(self):
        self.theta -= self.learning_rate * self._current_gradient

    def do_iteration(self):
        self._update_idx()
        self._update_stoch_func()
        super(SGD, self).do_iteration()


class ASGD(SGD):
    """Averaged SGD"""   
    
    name = "Averaged SGD"
    
    def _custom_init(self, warm_time=30, **kwargs):
        super(ASGD, self)._custom_init()
        self.online_theta = copy.deepcopy(self.theta)
        self.count = 1.0
        self.warm_time = warm_time

    def _compute_gradient(self):
        self._current_gradient = self._func.stoch_gradient(self.online_theta, util._ord[self._idx])

    def _update_theta(self):
        self.online_theta -= self.learning_rate * self._current_gradient
        if self._num_iter > self.warm_time:
            self.theta = self.count/(self.count+1) * self.theta + 1/(self.count+1) * self.online_theta
            self.count += 1
        else:
            self.theta = self.online_theta


class SGDsim(Algorithm):
    """SGD for simulations. i.e. noise is artificially added to the gradient"""

    name = "SGD"

    def _custom_init(self, cheat=False, **kwargs):
        self._idx = 0
        util.init_noise(1000)
        util.init_ord(1000)
        self.cheat = cheat

    def _update_idx(self):
        if self._idx < len(util._ord)-1:
            self._idx += 1
        else:
            self._idx = 0

    def _compute_gradient(self):
        self._current_gradient = self._func.gradient(self.theta) + util._noise[self._idx]

    def _update_theta(self):
        # Cheat for demo !!
        if ~self.cheat or np.linalg.norm(self._func.gradient(self.theta - self.learning_rate * self._current_gradient), 2) <= 1e-2:
            self.theta -= self.learning_rate * self._current_gradient

    def do_iteration(self):
        self._update_idx()
        super(SGDsim, self).do_iteration()


class ASGDsim(SGDsim):
    """Averaged SGD for simulations. i.e. noise is artificially added to the 
    gradient"""

    name = "Averaged SGD"

    def _custom_init(self, warm_time=50, **kwargs):
        super(ASGDsim, self)._custom_init(**kwargs)
        self.online_theta = copy.deepcopy(self.theta)
        self.count = 1.0
        self.warm_time = warm_time

    def _compute_gradient(self):
        self._current_gradient = self._func.gradient(self.online_theta) + util._noise[self._idx]

    def _update_theta(self):
        self.online_theta -= self.learning_rate * self._current_gradient
        if self._num_iter > self.warm_time:
            self.theta = self.count/(self.count+1) * self.theta + 1./(self.count+1) * self.online_theta
            self.count += 1
        else:
            self.theta = self.online_theta
