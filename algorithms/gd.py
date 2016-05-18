from core import Algorithm


class GD(Algorithm):

    name = "Gradient Descent"

    def _update_theta(self):
        self.theta -= self.learning_rate * self._current_gradient