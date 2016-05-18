from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from matplotlib import ticker
from matplotlib import cm
import numpy as np

from algorithms.util import ConvergenceError, flatten

class StochasticFunction2dWrapper(object):
    
    def __init__(self, func, gradient, stoch_func, stoch_gradient, n_samples):
        self._func = func
        self._gradient = gradient
        self._stoch_func = stoch_func
        self._stoch_gradient = stoch_gradient
        self.n_samples = n_samples

    def stoch_eval(self, theta, i):
        return self._stoch_func(*theta, i=i)

    def stoch_gradient(self, theta, i):
        return self._stoch_gradient(*theta, i=i)

    def eval(self, theta):
        """Alias for func"""
        return self._func(*theta)

    def gradient(self, theta):
        return self._gradient(*theta)


class Function2dWrapper(object):
    
    def __init__(self, func, gradient):
        self._func = func
        self._gradient = gradient

    def eval(self, theta):
        """Alias for func"""
        return self._func(*theta)

    def gradient(self, theta):
        return self._gradient(*theta)


class VizAlgoWrapper(object):

    def __init__(self, algorithm, color, gfx):
        self.algorithm = algorithm
        self.has_converged = False
        
        self.surface_points = {d:[] for d in ['x','y']}
        self.line_main, = gfx.ax_main.plot([], [], c=color, lw=2, alpha=.9, zorder=-1,
            label=self.algorithm.name, marker='D', markersize=2)
        self.point, = gfx.ax_main.plot([], [], c=color, lw=2, alpha=.9, zorder=10,
            marker='o', markersize=15)
        
        self.fval_points = {d:[] for d in ['x','y']}
        self.line_fval, = gfx.ax_fval.plot([], [], c=color, lw=2, 
            label=self.algorithm.name, alpha=1.0)

        self.gval_points = {d:[] for d in ['x','y']}
        self.line_gval, = gfx.ax_gval.plot([], [], c=color, lw=2, ls='-',
            label=self.algorithm.name, alpha=1.0)

    def update_viz(self):
        # Update surface plot data
        x, y = self.algorithm.theta
        self.surface_points['x'].append(x)
        self.surface_points['y'].append(y)
        self.line_main.set_data(self.surface_points['x'], self.surface_points['y'])
        self.point.set_data([x], [y])
        # Update value plot data
        n = self.algorithm._num_iter
        f = self.algorithm._func.eval(self.algorithm.theta)
        
        if self.algorithm.name.startswith('Ada'):
            g = np.linalg.norm(self.algorithm.adjusted_grad, 2)
        else:
            g = np.linalg.norm(self.algorithm._func.gradient(self.algorithm.theta), 2)

        self.fval_points['x'].append(n)
        self.fval_points['y'].append(f)
        self.line_fval.set_data(self.fval_points['x'], self.fval_points['y'])
        self.gval_points['x'].append(n)
        self.gval_points['y'].append(g)
        self.line_gval.set_data(self.gval_points['x'], self.gval_points['y'])

    def get_viz_objects(self):
        return [self.line_main, self.point, self.line_fval, self.line_gval]

    def get_legend_handle(self):
        return self.line_main

    def update_algo(self):
        self.algorithm.do_iteration()
        if self.algorithm.has_converged:
            self.has_converged = True


class TextPanel(object):

    def __init__(self, ax):
        self.text_dict = dict()
        self.xtext = 0.01
        self.ytext = 0.975
        self.ax = ax

    def add_text(self, method, message=None):
        if method.algorithm.name not in self.text_dict:
            name = method.algorithm.name
            if not message:
                message = '"{0}" converged in {1} steps'.format(name, int(method.algorithm._num_iter))
            text = self.ax.text(self.xtext, self.ytext, message, size=18, 
                                    name='Source Code Pro', weight='bold', style='italic',
                                    transform=self.ax.transAxes)
            self.ytext -= 0.025
            self.text_dict[name] = text


class SurfacePanel(object):

    def __init__(self, ax_main, ax_cbar, function, min_point, x_range, y_range):
        self.ax_main = ax_main  # Axis used for the surface plot
        self.ax_cbar = ax_cbar  # Axis used for the color bar
        self._init_style(x_range, y_range)
        X, Y, Z = self._init_meshgrid(function, x_range, y_range)
        self.plot_surface(X, Y, Z)
        self.plot_min_point(Z, min_point)

    def _init_style(self, x_range, y_range):   
        self.ax_main.set_xlabel(r'$\theta_0$', fontstyle='italic', fontname='Apple Chancery', fontsize=18)
        self.ax_main.set_ylabel(r'$\theta_1$', fontstyle='italic', fontname='Apple Chancery', fontsize=18)
        self.ax_main.set_xlim(*x_range)
        self.ax_main.set_ylim(*y_range)

    def _init_meshgrid(self, function, x_range, y_range):
        self.x = np.linspace(*x_range, num=50)
        self.y = np.linspace(*y_range, num=50)
        X, Y = np.meshgrid(self.x, self.y)
        f = np.vectorize(function._func)
        Z = f(X,Y)
        return X, Y, Z

    def plot_surface(self, X, Y, Z):
        # Init color scale
        lmin = np.floor(np.log10(max(Z.min(),1e-10)))
        lmax = np.ceil(np.log10(Z.max()))
        levels = np.logspace(lmin, lmax, 200)
        cmap = cm.get_cmap('BkBlAqGrYeOrReViWh200')
        # Plot surface
        im = self.ax_main.contourf(X, Y, Z, 
            levels=levels,
            norm=colors.LogNorm(),
            cmap=cmap,
            locator=ticker.LogLocator(),
            linewidth=0.1, alpha=0.8)
        # Set colorbar accordingly
        self.set_colorbar(lmin, lmax)

    def plot_min_point(self, Z, min_point):
        minx, miny, _ = min_point
        self.ax_main.plot([minx], [miny], marker='*', markersize=25, c='w', zorder=5)

    def set_colorbar(self, first_exp, last_exp, n_levels=10):
        levls = []
        for E in np.arange(first_exp,last_exp):
            levls = np.concatenate((levls[:-1],np.linspace(10**E,10**(E+1),n_levels)))
        XC = [np.zeros(len(levls)), np.ones(len(levls))]
        YC = [levls, levls]
        CM = self.ax_cbar.contourf(XC,YC,YC, levels=levls, 
                                   norm=colors.LogNorm(), 
                                   cmap=cm.get_cmap('BkBlAqGrYeOrReViWh200'))
        self.ax_cbar.set_yscale('log')   # log y-scale
        self.ax_cbar.yaxis.tick_right()  # y-labels on the right
        self.ax_cbar.set_xticks([])      # no x-ticks

    def set_legend(self, handles):
        self.ax_main.legend(handles=handles, loc=4)


class GraphicsContainer(object):

    def __init__(self, **options):
        self.fig = plt.figure()

        # Main surface plot
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        self.ax_main = plt.subplot(gs1[0, 0])

        # Colorbar plot
        gs_cbar = gridspec.GridSpec(1, 1)
        gs_cbar.update(left=0.555, right=0.575, wspace=0.0)
        self.ax_cbar = plt.subplot(gs_cbar[0, 0])

        # Plot surface
        self.surface_panel = SurfacePanel(self.ax_main, self.ax_cbar, 
                                          function=options['function'],
                                          min_point=options['min_point'],
                                          x_range=options['x_range'],
                                          y_range=options['y_range'])

        # ax_text = plt.subplot(gs2[2, 0])
        self.text_container = TextPanel(self.ax_main)

        # Secondary plots
        gs2 = gridspec.GridSpec(3, 1)
        gs2.update(left=0.67, right=0.98, hspace=0.2)
        self.ax_fval = plt.subplot(gs2[:2, 0])
        self.ax_gval = plt.subplot(gs2[2, 0])
        
        self.ax_fval.set_xlabel(r'Iteration', fontname='Lora', fontsize=18)
        self.ax_fval.set_ylabel(r'Objective value', fontname='Lora', fontsize=18)
        self.ax_fval.set_xlim([0, options['max_iter']])
       
        fvalnorm = options['gfx'].get('fvalnorm', 0.7)
        fvalscale = options['gfx'].get('fvalscale', 'linear')
        xmax = fvalnorm*options['function'].eval(options['start_theta'])
        xmin = options['gfx'].get('fvalxmin')
        if xmin is None:
            if fvalscale=='log':
                xmin = 1e-5
            else:
                xmin = 0
        self.ax_fval.set_ylim([xmin,xmax])
        self.ax_fval.set_yscale(fvalscale)

        
        self.ax_gval.set_xlabel(r'Iteration', fontname='Lora', fontsize=18)
        self.ax_gval.set_ylabel(r'Gradient L2-norm', fontname='Lora', fontsize=18)
        self.ax_gval.set_xlim([0, options['max_iter']])
        maxg = np.abs(options['function'].gradient(options['start_theta'])).max()
        self.ax_gval.set_ylim([options['accuracy']*0.1, maxg])
        self.ax_gval.set_yscale('log')
 

        # Adjust margins
        self.fig.subplots_adjust(left=0.01, right=0.95, top=0.99, bottom=0.05)

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

    def update(self, method_list):
        for method in method_list:
            method.update_viz()
            try:
                method.update_algo()
                if method.has_converged:
                    self.text_container.add_text(method)
            except ConvergenceError as e:
                self.text_container.add_text(method, message=e.message)

        artists = list()
        artists += flatten([m.get_viz_objects() for m in method_list])
        artists += self.text_container.text_dict.values()
        
        return artists

    def set_legend(self, handles):
        self.surface_panel.set_legend(handles)


