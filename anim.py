import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import numpy as np
from optparse import OptionParser
import copy
import sys

import algorithms

import core
import util
import config

METHODS = ['SGD', 'SGDsim', 'GD', 'LineSearchGD', 'MyLineSearchGD', 'LineSearch_SGD',
           'NAG', 'NAG2_SGD', 'Momentum', 'Momentum_SGD', 
           'AdaGrad', 'AdaDelta', 'AdaGrad_SGD', 'AdaDelta_SGD',
           'ASGD', 'ASGDsim', 'ConjugateGD', 'MyConjugateGD', 'Conjugate_SGD', 'MyConjugate_SGD']
COLORS = dict(zip(METHODS, ['red', 'red','#0046eb','#4daf4a', '#4daf4a', '#4daf4a',
                            '#984ea3', '#984ea3','#00D6D6', '#00D6D6',
                            '#ffff33','#a65628','#ffff33','#a65628',
                            'green', 'green', '#f781bf', '#f781bf', '#f781bf', '#f781bf']))


def make_cmap():
    c_array = np.genfromtxt('data/BkBlAqGrYeOrReViWh200_normed.rgb')
    plt.register_cmap(name='BkBlAqGrYeOrReViWh200', 
        data={key: tuple(zip(np.linspace(0,1,c_array.shape[0]), c_array[:,i], c_array[:,i])) 
                                             for key, i in zip(['red','green','blue'], (0,1,2))})
make_cmap()

class Experiment(object):

    method_list = []

    def __init__(self, **options):
        global gfx
        gfx = core.GraphicsContainer(**options)
        self._init_algorithms(**options)
        gfx.set_legend([m.get_legend_handle() for m in Experiment.method_list])

    def _init_algorithms(self, **options):
        for name in options['methods_to_use']:
            params = copy.deepcopy(options)
            del params['methods_to_use']
            params.update(options['methods_params'][name])
            algo = getattr(algorithms, name)(**params)
            viz = core.VizAlgoWrapper(algo, COLORS[name], gfx)
            Experiment.method_list.append(viz)

    @staticmethod
    def _draw(i):
        global gfx
        if i > 0:
            for _ in range(STEPS-1):
                gfx.update(Experiment.method_list)
            return gfx.update(Experiment.method_list)
        else:
            return []

    def run(self, blit=True, interval=10, repeat=False):
        anim = animation.FuncAnimation(gfx.fig, self._draw, blit=blit, interval=interval, repeat=repeat)
        plt.show()


def normalize_options(optargs, default_options):
    options = default_options
    if optargs.start_theta:
        options['start_theta'] = eval(optargs.start_theta)
    
    if optargs.methods_to_use:
        options['methods_to_use'] = eval(optargs.methods_to_use)
    elif not options.has_key('methods_to_use'):
        options['methods_to_use'] = ['GD', 'LineSearchGD', 'NAG', 'Momentum', 'ConjugateGD']
    
    if optargs.learning_rate:
        for m in options['methods_to_use']:
            options['methods_params'][m]['learning_rate'] = optargs.learning_rate
    
    if optargs.momentum:
        for m in options['methods_to_use']:
            options['methods_params'][m]['momentum'] = optargs.momentum

    options.setdefault('max_iter', MAX_ITER)
    options.setdefault('accuracy', ACCURACY) 
    for m in options['methods_to_use']:
        if not options['methods_params'][m].has_key('accuracy'):
            options['methods_params'][m]['accuracy'] = options['accuracy']

    if not options.has_key('gfx'):
        options['gfx'] = {'fvalnorm': 0.9, 'fvalscale': 'linear'}
    else:
        if not options['gfx'].has_key('fvalnorm'):
            options['gfx']['fvalnorm'] = 0.9
        if not options['gfx'].has_key('fvalscale'):
            options['gfx']['fvalscale'] = 'linear'
    return options

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--demo", dest="demo", type="int", default=0)
    parser.add_option("--fps", dest="fps", type="int", default=100)
    parser.add_option("--func", dest="func", type="string")
    
    parser.add_option("--steps", dest="steps", type="int", default=1)
    parser.add_option("--max_iter", dest="max_iter", type="int", default=500)
    parser.add_option("--accuracy", dest="accuracy", type="float", default=1e-2)

    parser.add_option("-m", dest="methods_to_use", type="string", default=None)
    parser.add_option("--momentum", dest="momentum", type="float", default=None)
    parser.add_option("--start", dest="start_theta", type="string", default=None)
    parser.add_option("--lrate", dest="learning_rate", type="float", default=None)
    (optargs, args) = parser.parse_args()

    global STEPS
    STEPS = optargs.steps
    global MAX_ITER
    MAX_ITER = optargs.max_iter
    global ACCURACY
    ACCURACY = optargs.accuracy


    DEMO_CONFIG = {}


    DEMO_CONFIG[1] = {
    'function': config.quad_func,
    'methods_to_use': ['SGDsim', 'GD'],
    'max_iter': 200,
    'noise': True,
    'x_range': [-1.0, 1.0],
    'y_range': [-1.0, 1.0],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-0.5,-0.9],
    'projection': 'contourf',
    'gfx': {'fvalnorm': 0.15, 'fvalscale': 'linear'},
    'methods_params': {
        'SGDsim': {
            'learning_rate': 0.9 * 1./6, 'accuracy': 1e-3, 'cheat': False },
        'GD': {
            'learning_rate': 0.9 * 1./6, 'accuracy': 1e-3 },
        'ASGDsim': {
            'learning_rate': 0.9 * 1./6, 'accuracy': 1e-6, 'cheat': True, 
            'warm_time': 37 },
        }
    }

    DEMO_CONFIG[2] = {
    'function': config.quad_func,
    'methods_to_use': ['GD'],
    'accuracy': 1e-2,
    'x_range': [-100, 100],
    'y_range': [-100, 100],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-40,-68],
    'projection': 'contourf',
    'gfx': {'fvalnorm': 1.0, 'fvalscale': 'linear'},
    'methods_params': {
        'GD': {
            'learning_rate': 2./6 },
        }
    }

    DEMO_CONFIG[3] = {
    'function': config.quad_func,
    'methods_to_use': ['GD','LineSearchGD'],
    'accuracy': 1e-3,
    'x_range': [-100, 100],
    'y_range': [-100, 100],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-40,-68],
    'projection': 'contourf',
    'gfx': {'fvalnorm': 1.0, 'fvalscale': 'log'},
    'methods_params': {
        'GD': {
            'learning_rate': 2./6 },
        'LineSearchGD': {
            'learning_rate': 0.05 },
        'MyLineSearchGD': {
            'learning_rate': 0.05, 'c': 0.5 },
        }
    }

    DEMO_CONFIG[4] = {
    'function': config.quad_func,
    'methods_to_use': ['GD','LineSearchGD','MyConjugateGD'],
    'accuracy': 1e-2,
    'x_range': [-70, 50],
    'y_range': [-70, 50],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-40,-68],
    'projection': 'contourf',
    'gfx': {'fvalnorm': 0.7, 'fvalscale': 'linear'},
    'methods_params': {
        'GD': {
            'learning_rate': 0.05 },
        'Momentum': {
            'learning_rate': 0.05, 
            'momentum': 0.9 },
        'NAG': {
            'learning_rate': 0.05 },
        'LineSearchGD': {
            'learning_rate': 0.05 },
        'MyLineSearchGD': {
            'learning_rate': 0.05, 'c': 0.5 },
        'ConjugateGD': {
            'learning_rate': 0.05 },
        'MyConjugateGD': {
            'learning_rate': 0.05, 'c2': 0.2 },
        }
    }

    DEMO_CONFIG[5] = {
        'function': config.rosenbrock_func,
        'methods_to_use': ['GD','LineSearchGD','MyConjugateGD'],
        'accuracy': 5e-2,
        'x_range': [-2, 2],
        'y_range': [-1.5, 3.5],
        'min_point': [1.0, 1.0, 0.0],
        'start_theta': [-0.4, 2.5],
        'projection': 'contourf',
        'gfx': {'fvalnorm': 0.2, 'fvalscale': 'log'},
        'methods_params': {
            'GD': {
                'learning_rate': 0.001 },
            'Momentum': {
                'learning_rate': 0.001, 
                'momentum': 0.9 },
            'NAG': {
                'learning_rate': 0.001 },
            'AdaGrad': {
                'learning_rate': 1 },
            'AdaDelta': {
                'learning_rate': 1,
                'momentum': 0.9 },
            'LineSearchGD': {
                'learning_rate': 0.001,
                'amax': 0.1 },
            'MyLineSearchGD': {
                'learning_rate': 0.001,
                'amax': 0.01 },
            'ConjugateGD': {
                'learning_rate': 0.001 },
            'MyConjugateGD': {
                'learning_rate': 0.001, 'c2': 0.6 },
        }
    }

    DEMO_CONFIG[6] = {
    'function': config.quad_func,
    'methods_to_use': ['GD', 'Momentum'],
    'accuracy': 1e-2,
    'x_range': [-100, 100],
    'y_range': [-100, 100],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-40,-68],
    'projection': 'contourf',
    'gfx': {'fvalnorm': 0.7, 'fvalscale': 'linear'},
    'methods_params': {
        'GD': {
            'learning_rate': 2./6 },
        'Momentum': {
            'learning_rate': 2./6, 
            'momentum': 0.5 }
        }
    }

    DEMO_CONFIG[7] = {
        'function': config.rosenbrock_func,
        'methods_to_use': ['GD', 'Momentum'],
        'accuracy': 1e-2,
        'max_iter': 600,
        'x_range': [-2, 2],
        'y_range': [-2.0, 2.5],
        'min_point': [1.0, 1.0, 0.0],
        'start_theta': [-0.45, 2.3],
        'projection': 'contourf',
        'gfx': {'fvalnorm': 0.7, 'fvalscale': 'linear'},
        'methods_params': {
            'GD': {
                'learning_rate': 0.0015 },
            'Momentum': {
                'learning_rate': 0.001, 
                'momentum': 0.95 }
        }
    }

    DEMO_CONFIG[8] = {
        'function': config.rosenbrock_func,
        'methods_to_use': ['GD', 'Momentum', 'NAG'],
        'accuracy': 1e-2,
        'x_range': [-2, 2],
        'y_range': [-2.0, 2.5],
        'min_point': [1.0, 1.0, 0.0],
        'start_theta': [-0.45, 2.3],
        'projection': 'contourf',
        'gfx': {'fvalnorm': 0.5, 'fvalscale': 'linear'},
        'methods_params': {
            'GD': {
                'learning_rate': 0.0015 },
            'Momentum': {
                'learning_rate': 0.001, 
                'momentum': 0.95 },
            'NAG': {
                'learning_rate': 0.001}
        }
    }

    wine_func = None
    if optargs.demo in [9,10]: 
        wine_func = util.build_wine_logreg_func(stoch=True) 
    DEMO_CONFIG[9] = {
        'function': wine_func,
        'methods_to_use': ['Momentum', 'NAG', 'AdaGrad', 'LineSearchGD', 'AdaDelta', 'GD', 'MyConjugateGD'],
        'accuracy': 1e-3,
        'max_iter': 250,
        'x_range': [-4, 2],
        'y_range': [-2, 4],
        'min_point': None,
        'start_theta': [1.7, 3.7],
        'min_point': [-1.74, 0.99, 0.0],
        'projection': 'contourf',
        'gfx': {'fvalnorm': 0.5, 'fvalscale': 'log', 'fvalxmin': 1e-2},
        'methods_params': {
            'GD': {
                'learning_rate': 1.0 },
            'NAG': {
                'learning_rate': 0.25,
                'momentum': 0.9 },
            'Momentum': {
                'learning_rate': 0.25,
                'momentum': 0.8 },
            'LineSearchGD': {
                'learning_rate': 0.5,
                'amax': 1.0 },
            'MyConjugateGD': {
                'learning_rate': 10.0, 'c2': 0.1 },
            'AdaGrad': {
                'learning_rate': 2.5, },
            'AdaDelta': {
                'learning_rate': 2.5, 
                'momentum': 0.9999},

            'SGD': {
                'learning_rate': lambda k: 0.5 * k**-0.5 },
            'NAG2_SGD': {
                'learning_rate': lambda k: 0.1 * k**-0.3,
                'momentum': 0.95 },
            'Momentum_SGD': {
                'learning_rate': lambda k: 0.1 * k**-0.3,
                'momentum': 0.95 },
            'LineSearch_SGD': {
                'learning_rate': 0.05,
                'amax': 0.5 },
            'AdaGrad_SGD': {
                'learning_rate': 1.0, },
            'AdaDelta_SGD': {
                'learning_rate': 4.5, 
                'momentum': 0.999999},
            'MyConjugate_SGD': {
                'learning_rate': 10.0, 'amax': 100.0, 'c2': 0.9 },
        }
    }
    DEMO_CONFIG[10] = copy.deepcopy(DEMO_CONFIG[9])
    DEMO_CONFIG[10]['methods_to_use'] = ['LineSearch_SGD', 'AdaGrad_SGD', 'AdaDelta_SGD', 'Momentum_SGD', 'NAG2_SGD', 'SGD', 'MyConjugate_SGD']
    DEMO_CONFIG[10]['max_iter'] = 1200
    DEMO_CONFIG[10]['accuracy'] = 5e-3

    if optargs.demo >= 1:
        options = normalize_options(optargs, DEMO_CONFIG[optargs.demo])
        experiment = Experiment(**options).run(interval=optargs.fps)

    elif optargs.func is not None:
        if not config.default_options.has_key(optargs.func):
            print "Function '{0}' not found...".format(optargs.func)
            sys.exit(1)
        options = normalize_options(optargs, config.default_options[optargs.func])
        experiment = Experiment(**options).run(interval=optargs.fps)
