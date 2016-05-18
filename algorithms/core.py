import numpy as np

from util import ConvergenceError, cast_as_func


class Algorithm(object):
    """Abstract class for gradient descent algorithms"""

    name = "Algorithm"

    def __init__(self, function, start_theta, learning_rate, max_iter, accuracy, **kwargs):
        """

        Arguments:
        ---------
        function : callable
            Function to optimize
        start_theta : array-like
            Starting point of the algorithm
        """
        self._func = function
        self.theta = np.array(start_theta, dtype='float')
        self._learning_rate = cast_as_func(learning_rate)
        self._num_iter = 0
        self._max_iter = max_iter
        self._accuracy = accuracy
        self._custom_init(**kwargs)
        self.has_converged = False

    @property
    def learning_rate(self):
        """Learning rate"""
        return self._learning_rate(self._num_iter)

    def _custom_init(self, **kwargs):
        """Custom initialization specific to each algorithm"""

    def _compute_gradient(self):
        self._current_gradient = self._func.gradient(self.theta)

    def _has_converged(self):
        """Check if the algorithm has converged"""
        if np.isnan(self.theta).sum() > 0:
            self.has_converged = True
            raise ConvergenceError('"{0}" has diverged!'.format(self.name))
        if self._num_iter >= self._max_iter:
            self.has_converged = True
            raise ConvergenceError('"{0}" reached the max number of steps ({1})!'.format(self.name, self._max_iter))
        if self._accuracy is not None:
            if np.linalg.norm(self._func.gradient(self.theta),2) < self._accuracy:
                # print "Algorithm {0} has converged in {1} iterations.".format(
                #     self.name, self._num_iter)
                self.has_converged = True

    def do_iteration(self):
        if not self.has_converged:
            self._num_iter += 1
            self._compute_gradient()
            self._update_theta()
            self._has_converged()
        return self.has_converged





