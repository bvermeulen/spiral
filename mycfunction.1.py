import sys
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib.collections import PatchCollection
from descartes.patch import PolygonPatch
from scipy.signal import convolve2d
from convex_hull_alpha_shape import alpha_shape
import json

FIG_SIZE = (10, 10)
PLOT_LIMIT = 1.5
NUM_ITERATIONS = 100
DPU_JULIASET = 40
MAX_ITERATIONS_JULIASET = 50
BORDER_RANGE = range(1, 6)
ALPHA = 0.1


class MainMap():

    @classmethod
    def settings(cls, fig_size, plot_limit):
        # set the plot outline, including axes going through the origin
        cls.fig, cls.ax = plt.subplots(figsize=fig_size)
        cls.plot_limit = plot_limit
        cls.ax.set_xlim(-cls.plot_limit, cls.plot_limit)
        cls.ax.set_ylim(-cls.plot_limit, cls.plot_limit)
        cls.ax.set_aspect(1)
        tick_range = np.arange(round(-cls.plot_limit + (10*cls.plot_limit % 2)/10, 1), cls.plot_limit + 0.1, step=0.2)
        cls.ax.set_xticks(tick_range)
        cls.ax.set_yticks(tick_range)
        cls.ax.tick_params(axis='both', which='major', labelsize=6)
        cls.ax.spines['left'].set_position('zero')
        cls.ax.spines['right'].set_color('none')
        cls.ax.spines['bottom'].set_position('zero')
        cls.ax.spines['top'].set_color('none')

        # plot unit circle
        circle = 1*cls.unit_circle()
        cls.ax.plot(circle.real, circle.imag, c='black')

    @classmethod
    def get_ax(cls):
        return cls.ax

    @staticmethod
    def plot():
        plt.tight_layout()
        plt.show()

    @staticmethod
    def unit_circle():
        radians = np.linspace(-np.pi, np.pi, 1000)
        circle = np.exp(1j * radians)
        return circle


class JuliaSets(MainMap):

    def __init__(self, tolerance=0):
        self.current_object = None
        self.current_dragging = False

        self.julia_constant = mpl_patches.Circle((0.0, 0.3), 0.05, fc='g', alpha=1)
        self.ax.add_patch(self.julia_constant)
        self.julia_constant.set_picker(tolerance)
        cv_point = self.julia_constant.figure.canvas
        cv_point.mpl_connect('button_release_event', self.on_release)
        cv_point.mpl_connect('pick_event', self.on_pick)
        cv_point.mpl_connect('motion_notify_event', self.on_motion)

        self.plot_boundary_juliaset()

    @staticmethod
    def juliaset_func(point, constant):
        z = point
        stable = True
        num_iterations = 1
        while stable and num_iterations < MAX_ITERATIONS_JULIASET:
            z = z**2 + constant
            if abs(z) > max(abs(constant), 2):
                stable = False
                return (stable, num_iterations)
            num_iterations += 1

        return (stable, 0)

    def plot_boundary_juliaset(self):
        # determines the resolution of the boundary of the julia set
        # dots per unit - 50 dots per 1 units means 200 points per 4 units
        intval = 1 / DPU_JULIASET
        r_range = np.arange(-self.plot_limit, self.plot_limit + intval, intval)
        c_range = np.arange(-self.plot_limit, self.plot_limit + intval, intval)
        constant = complex(self.julia_constant.center[0], self.julia_constant.center[1])

        stables = np.array([], dtype='bool')
        for imag in c_range:
            for real in r_range:
                stable, _ = self.juliaset_func(complex(real, imag), constant)
                stables = np.append(stables, stable)

        rows = len(r_range)
        cols = len(c_range)
        start = len(stables)
        stable_field = []
        for _ in range(cols):
            start -= rows
            real_vals = [1 if val == True else 0 for val in stables[start:start+rows]]
            stable_field.append(real_vals)
        stable_field = np.array(stable_field, dtype='int')

        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        stable_boundary = convolve2d(stable_field, kernel, mode='same')

        self.boundary_points = np.array([], dtype='complex')
        for col in range(cols):
            for row in range(rows):
                if stable_boundary[col, row] in BORDER_RANGE:
                    real_val = r_range[row]
                    # invert cols as min imag value is highest col and vice versa
                    imag_val = c_range[cols-1 - col]
                    self.boundary_points = np.append(self.boundary_points, 
                        complex(real_val, imag_val))
                else:
                    pass

        bnd_points = [Point(val.real, val.imag) for val in self.boundary_points]
        bnd_polygon, _ = alpha_shape(bnd_points, ALPHA)

        patches = []
        if bnd_polygon.geom_type == 'Polygon':
            patches.append(PolygonPatch(bnd_polygon))
            ec, lw = 'red', 2
        else:
            for poly in bnd_polygon:
                patches.append(PolygonPatch(poly))
            ec, lw = 'green', 1

        p = PatchCollection(patches, facecolor='none', edgecolor=ec, lw=lw)
        self.juliaset_plot = self.ax.add_collection(p)

    def on_release(self, event):
        self.current_object = None
        self.current_dragging = False

    def on_pick(self, event):
        if event.artist != self.julia_constant:
            return

        self.current_dragging = True
        self.current_object = event.artist

    def on_motion(self, event):
        if not self.current_dragging:
            return
        if self.current_object == None:
            return

        self.remove_juliaset_from_plot()
        self.plot_boundary_juliaset()

        self.julia_constant.center = event.xdata, event.ydata
        self.julia_constant.figure.canvas.draw()

    def remove_juliaset_from_plot(self):
        try:
            self.juliaset_plot.remove()
        except ValueError:
            pass


class MandelbrotPoints(MainMap):

    def __init__(self, start_point, num_iterations, tolerance=0):

        self.num_iterations = num_iterations
        self.current_object = None
        self.currently_dragging = False
        self.plot_types = ['-o', 'o']
        self.plot_type = 0

        # define a complex constant to be used in complex function
        # this point is blue and can be moved interactively. Initial value is
        # the origin
        self.constant = mpl_patches.Circle((start_point.real, start_point.imag), 
            0.05, fc='blue', alpha=1)
        self.ax.add_patch(self.constant)
        self.constant.set_picker(tolerance)
        cv_constant = self.constant.figure.canvas
        cv_constant.mpl_connect('button_release_event', self.on_release)
        cv_constant.mpl_connect('pick_event', self.on_pick)
        cv_constant.mpl_connect('motion_notify_event', self.on_motion)

        # define a starting point in complex plane
        # this point is yellow and can be move interactively
        self.point = mpl_patches.Circle((0, 0), 0.05, fc='yellow', alpha=1)
        self.ax.add_patch(self.point)
        self.point.set_picker(tolerance)
        cv_point = self.point.figure.canvas
        cv_point.mpl_connect('button_release_event', self.on_release)
        cv_point.mpl_connect('pick_event', self.on_pick)
        cv_point.mpl_connect('motion_notify_event', self.on_motion)

        self.plot_mandelbrot_points()

    def plot_mandelbrot_points(self):
        c_constant = complex(self.constant.center[0], self.constant.center[1])
        c = complex(self.point.center[0], self.point.center[1])
        c_numbers = np.array(c)
        for _ in range(self.num_iterations):
            c = c**2 + c_constant
            c_numbers = np.append(c_numbers, c)

        self.mandelbrot_plot, = self.ax.plot(c_numbers.real, c_numbers.imag,
            self.plot_types[self.plot_type], color='r', lw=0.3, markersize=2)

    def on_release(self, event):
        self.current_object = None
        self.currently_dragging = False

    def on_pick(self, event):
        self.currently_dragging = True
        self.current_object = event.artist

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_object == None:
            return

        self.current_object.center = event.xdata, event.ydata
        self.remove_mandelbrot_from_plot()
        self.plot_mandelbrot_points()
        self.point.figure.canvas.draw()

    def remove_mandelbrot_from_plot(self):
        try:
            self.mandelbrot_plot.remove()
        except ValueError:
            pass


class OnKey(MainMap):
    @classmethod
    def start(cls, mp, jp):
        cls.fig.canvas.mpl_connect('key_press_event', cls.on_key)
        cls.mp = mp
        cls.jp = jp
        cls.json_file_count = 2

    @classmethod
    def on_key(cls, event):
        # with 'space' toggle between just points or points connected with
        # lines
        if event.key == ' ':
            cls.mp.plot_type = (cls.mp.plot_type + 1) % 2
            cls.mp.remove_mandelbrot_from_plot()
            cls.mp.plot_mandelbrot_points()
            cls.mp.point.figure.canvas.draw()

        elif event.key in ['j', 'J']:
            cls.json_file_count += 1
            json_file_name = ''.join(['julia_set_', 
                str(cls.json_file_count), '.json']) 
            bnd_pts = [(val.real, val.imag) for val in cls.jp.boundary_points]
            with open(json_file_name, 'wt') as json_file:
                json.dump({'boundary': bnd_pts}, json_file)


def main(start_point):

    MainMap.settings(FIG_SIZE, PLOT_LIMIT)
    mp = MandelbrotPoints(start_point, NUM_ITERATIONS)
    jp = JuliaSets()
    OnKey.start(mp, jp)
    MainMap.plot()


if __name__ == "__main__":
    # a good example for constant = 0.3 + 0.25j
    try:
        start_point = complex(sys.argv[1])
    except IndexError:
        start_point= 0.3 + 0.25j

    main(start_point)
