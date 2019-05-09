import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint


PLOT_LIMIT = 2.5
NUM_ITERATIONS = 100
FIG_SIZE = (10, 10)


class MainMap():

    @classmethod
    def settings(cls, fig_size, plot_limit):
        # set the plot outline, including axes going through the origin
        cls.fig, cls.ax = plt.subplots(figsize=fig_size)
        cls.ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        cls.ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        cls.ax.set_aspect(1)
        tick_range = np.arange(round(-plot_limit + (10*plot_limit % 2)/10, 1), plot_limit + 0.1, step=0.2)
        cls.ax.set_xticks(tick_range)
        cls.ax.set_yticks(tick_range)
        cls.ax.tick_params(axis='both', which='major', labelsize=6)
        cls.ax.spines['left'].set_position('zero')
        cls.ax.spines['right'].set_color('none')
        cls.ax.spines['bottom'].set_position('zero')
        cls.ax.spines['top'].set_color('none')

        # plot unit circle
        c_real, c_imag = zip(*1*cls.unit_circle())
        cls.ax.plot(c_real, c_imag)
    
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
        return np.column_stack((circle.real, circle.imag))


class PlotJuliaSets(MainMap):

    def __init__(self, tolerance=0):

        # self.fig = fig
        # self.ax = ax
        self.current_object = None
        self.currently_dragging = False

        self.point = patches.Circle((0.5, 0.5), 0.05, fc='g', alpha=1)
        self.ax.add_patch(self.point)
        self.point.set_picker(tolerance)
        cv_point = self.point.figure.canvas
        cv_point.mpl_connect('button_release_event', self.on_release)
        cv_point.mpl_connect('pick_event', self.on_pick)
        cv_point.mpl_connect('motion_notify_event', self.on_motion)


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
        self.point.figure.canvas.draw()


class PlotMandelbrotPoints(MainMap):

    def __init__(self, start_point, num_iterations, tolerance=0):

        # self.fig = fig
        # self.ax = ax
        self.num_iterations = num_iterations
        self.current_object = None
        self.currently_dragging = False
        self.plot_types = ['-o', 'o']
        self.plot_type = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # define a complex constant to be used in complex function
        # this point is blue and can be moved interactively. Initial value is
        # the origin
        self.constant = patches.Circle((0, 0), 0.05, fc='b', alpha=1,
            gid='constant')
        self.ax.add_patch(self.constant)
        self.constant.set_picker(tolerance)
        cv_constant = self.constant.figure.canvas
        cv_constant.mpl_connect('button_release_event', self.on_release)
        cv_constant.mpl_connect('pick_event', self.on_pick)
        cv_constant.mpl_connect('motion_notify_event', self.on_motion)

        # define a starting point in complex plane
        # this point is yellow and can be move interactively
        self.point = patches.Circle((start_point.real, start_point.imag),
            0.05, fc='yellow', alpha=1, gid='point')
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

        self.function_plot, = self.ax.plot(c_numbers.real, c_numbers.imag,
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
        self.remove_function_from_plot()
        self.plot_mandelbrot_points()
        self.point.figure.canvas.draw()

    def remove_function_from_plot(self):
        try:
            self.function_plot.remove()
        except ValueError:
            pass

    def on_key(self, event):
        # with 'space' toggle between just points or points connected with
        # lines
        if event.key == ' ':
            self.plot_type = (self.plot_type + 1) % 2
            self.remove_function_from_plot()
            self.plot_mandelbrot_points()
            self.point.figure.canvas.draw()


def main(start_point):

    MainMap.settings(FIG_SIZE, PLOT_LIMIT)
    pmp = PlotMandelbrotPoints(start_point, NUM_ITERATIONS)  #pylint: disable=unused-variable
    pjs = PlotJuliaSets()  #pylint: disable=unused-variable
    MainMap.plot()

if __name__ == "__main__":
    # a good example for constant = 0.3 + 0.25j
    try:
        start_point = complex(sys.argv[1])
    except IndexError:
        start_point= 0.3 + 0.25j

    main(start_point)
