import numpy as np
import copy

from core import Function2dWrapper, StochasticFunction2dWrapper

default_options = dict()

beale_func = Function2dWrapper(
    lambda x,y: (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2,
    lambda x,y: np.array([
        2*(1.5-x+x*y)*(y-1) + 2*(2.25-x+x*y**2)*(y**2-1) + 2*(2.625-x+x*y**3)*(y**3-1), 
        2*(1.5-x+x*y)*x + 2*(2.25-x+x*y**2)*(2*x*y) + 2*(2.625-x+x*y**3)*(3*x*y**2)
        ])
)
default_options['beale'] = {
    'function': beale_func,
    'x_range': [-3.5, 3.5],
    'y_range': [-3.5, 3.5],
    'z_range': [0, 20000],
    'min_point': [3.0, 0.5, 0.0],
    'start_theta': [2.0, 1.5],
    'projection': 'contourf',
    'view_init': [20, -20],
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.001 },
        'Momentum': {
            'learning_rate': 0.001, 
            'momentum': 0.9 },
        'NAG': {
            'learning_rate': 0.001},
        'LineSearchGD': {
            'learning_rate': 0.001 },
        'ConjugateGD': {
            'learning_rate': 0.001 },
    }
}

rosenbrock_func = Function2dWrapper(
    lambda x,y: 100*(y-x**2)**2 + (x-1)**2,
    lambda x,y: np.array([-400*x*(y-x**2)+2*(x-1), 200*(y-x**2)])
)
default_options['rosenbrock'] = {
    'function': rosenbrock_func,
    'x_range': [-2.0, 2.0],
    'y_range': [-2.0, 2.5],
    'z_range': [0, 2500],
    'min_point': [1.0, 1.0, 0.0],
    'start_theta': [-1.6, 0.0],
    'projection': 'contourf',
    'view_init': [20,-90],
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.001 },
        'Momentum': {
            'learning_rate': 0.0007, 
            'momentum': lambda k: (k<20)*0.5 + (k>=20)*0.9 },
        'NAG': {
            'learning_rate': 0.0007 },
        'LineSearchGD': {
            'learning_rate': 0.001 },
        'ConjugateGD': {
            'learning_rate': 0.001 },
    }
}

booth_func = Function2dWrapper(
    lambda x,y: (x+2*y-7)**2 + (2*x+y-5)**2,
    lambda x,y: np.array([2*(x+2*y-7) + 4*(2*x+y-5),
                          4*(x+2*y-7) + 2*(2*x+y-5)])
)
default_options['booth'] = {
    'function': booth_func,
    'x_range': [-5, 5],
    'y_range': [-5, 5],
    'z_range': [0, 2500],
    'min_point': [1.0, 3.0, 0.0],
    'start_theta': [-2.5, -4.5],
    'projection': '3d',
    'view_init': [30,-30],
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.01 },
        'Momentum': {
            'learning_rate': 0.01, 
            'momentum': lambda k: (k<20)*0.5 + (k>20)*0.9 },
        'NAG': {
            'learning_rate': 0.001 },
        'LineSearchGD': {
            'learning_rate': 0.001 },
        'ConjugateGD': {
            'learning_rate': 0.001 },
    }
}

dixon_price_func = Function2dWrapper(
    lambda x,y: (x-1)**2 + 2*(2*y**2-x)**2,
    lambda x,y: np.array([2*(x-1) - 4*(2*y**2-x),
                          16*y*(2*y**2-x)])
)
default_options['dixon'] = {
    'function': dixon_price_func,
    'x_range': [-10, 10],
    'y_range': [-10, 10],
    'z_range': [0, 9e4],
    'min_point': [1.0, 0.5, 0.0],
    'start_theta': [-9, 9],
    'projection': '3d',
    'view_init': [30,-30],
    'gfx': {'fvalnorm': 1.0, 'fvalscale': 'log', 'fvalxmin': 1e-4},
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.00035 },
        'Momentum': {
            'learning_rate': 0.00035, 
            'momentum': lambda k: (k<20)*0.1 + (k>20)*0.9 },
        'NAG': {
            'learning_rate': 0.00035 },
        'LineSearchGD': {
            'learning_rate': 0.00035 },
        'ConjugateGD': {
            'learning_rate': 0.00035 },
    }
}

three_hump_camel_func = Function2dWrapper(
    lambda x,y: 2*x**2 - 1.05*x**4 + 1./6*x**6 - x*y + y**2,
    lambda x,y: np.array([x*((x**2-2.1)**2 - 0.41) - y,
                          2*y - x])
)
default_options['camel'] = {
    'function': three_hump_camel_func,
    'x_range': [-4, 4],
    'y_range': [-4, 4],
    'z_range': [0, 2000],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-3.9, 3.9],
    'projection': '3d',
    'view_init': [30,-30],
    'gfx': {'fvalnorm': 1.0, 'fvalscale': 'log', 'fvalxmin': 1e-8},
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.003 },
        'Momentum': {
            'learning_rate': 0.003, 
            'momentum': lambda k: (k<10)*0.1 + (k>10)*0.9 },
        'NAG': {
            'learning_rate': 0.003 },
        'LineSearchGD': {
            'learning_rate': 0.001 },
        'ConjugateGD': {
            'learning_rate': 0.001 },
    }
}


quad_func = Function2dWrapper(
    lambda x, y: 1.5 * ((x+y)**2 * 1./16 + (x-y)**2),
    lambda x, y: np.array([3./16*(x+y)+3*(x-y), 3./16*(x+y)-3*(x-y)])
)
default_options['quad'] = {
    'function': quad_func,
    'max_iter': 200,
    'x_range': [-100, 100],
    'y_range': [-100, 100],
    'z_range': [0, 20],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-40,-68],
    'projection': 'contourf',
    'view_init': [30,-30],
    'gfx': {'fvalnorm': 0.5, 'fvalscale': 'log', 'fvalxmin': 1e-5},
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 1./6 },
        'Momentum': {
            'learning_rate': 1./6, 
            'momentum': lambda k: (k<10)*0.1 + (k>10)*0.9 },
        'NAG2': {
            'learning_rate': 1./6, 
            'momentum': lambda k: (k<10)*0.1 + (k>10)*0.9 },
        'NAG': {
            'learning_rate': 1./6},
        'LineSearchGD': {
            'learning_rate': 1./6 },
        'ConjugateGD': {
            'learning_rate': 1./6 },
    }
}




quad2_func = Function2dWrapper(
    lambda x,y: 0.05 * (0.02*x**2 + 0.005*y**2),
    lambda x,y: 0.05 * np.array([0.04*x,
                          0.01*y])
)
default_options['quad2'] = {
    'function': quad2_func,
    'accuracy': 1e-6,
    'x_range': [-6, 6],
    'y_range': [-6, 11],
    'z_range': [0, 20],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-5.0, 10.0],
    'projection': 'contourf',
    'view_init': [30,-30],
    'gfx': {'fvalnorm': 0.3, 'fvalscale': 'linear', 'fvalxmin': 0},
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 25.0 },
        'Momentum': {
            'learning_rate': 25.0, 
            'momentum': 0.9 },
        'NAG': {
            'learning_rate': 25.0 },
        'LineSearchGD': {
            'learning_rate': 25.0 },
        'ConjugateGD': {
            'learning_rate': 25.0 },
    }
}




gaussian_func = Function2dWrapper(
    lambda x,y: 1-np.exp(-(x**2)/2-(y**2)/2.9),
    lambda x,y: np.array([x*np.exp(-(x**2)/2-(y**2)/2.9),
                          2.0/2.9*y*np.exp(-(x**2)/2-(y**2)/2.9)])
)
default_options['gaussian'] = {
    'function': gaussian_func,
    'x_range': [-3, 3],
    'y_range': [-3, 3],
    'z_range': [0, 1],
    'min_point': [0.0, 0.0, 0.0],
    'start_theta': [-2.5, 2.5],
    'projection': '3d',
    'view_init': [30,-30],
    'methods_to_use': ['GD', 'Momentum', 'NAG', 'LineSearchGD', 'ConjugateGD'],
    'methods_params': {
        'GD': {
            'learning_rate': 0.5 },
        'Momentum': {
            'learning_rate': 0.5, 
            'momentum': 0.9 },
        'NAG': {
            'learning_rate': 0.5 },
        'LineSearchGD': {
            'learning_rate': 0.5 },
        'ConjugateGD': {
            'learning_rate': 0.5 },
    }
}