## Motivation

This package provides an interface to experiment with the design of Gradient descent algorithms:
* Gradient descent
* Stochastic gradient descent
* Line search
* Conjugate gradient descent
* Momentum
* Nesterov accelerated gradient
* AdaGrad
* AdaDelta
* ...

All algorithms can be easily vizualized as 2d animations.

## Dependencies

This package relies on Numpy & Scipy for computations and on Matplotlib for visualization.

To load the real toy-example datasets, this package uses Pandas and Scikit-learn, but both dependencies are only required if the datasets are loaded.

Run ```pip install -r requirements.txt``` to install the dependencies.

## Demos

To run the tutorial demos, run ```python anim.py --demo N``` where N can be an integer from 1 to 10.

Note that the demos have been tested on Mac OS X. The backend used by matplotlib might need to be changed to work as expected on other platforms.